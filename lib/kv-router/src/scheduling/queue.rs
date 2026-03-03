// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use tokio::sync::watch;

use super::selector::WorkerSelector;
use super::types::{SchedulingRequest, SchedulingResponse, WorkerCapacity};
use crate::protocols::{ActiveLoad, WorkerConfigLike, WorkerId, WorkerWithDpRank};
use crate::sequences::{ActiveSequencesMultiWorker, SequencePublisher, SequenceRequest};

/// Large default for max_num_batched_tokens when not configured (effectively disables queueing for that worker)
pub const DEFAULT_MAX_BATCHED_TOKENS: u64 = 10_000_000;

/// Entry in the priority queue, ordered by effective arrival time (lower = higher priority).
/// Effective arrival = elapsed time since queue start minus `priority_jump`.
struct QueueEntry {
    effective_offset: Duration,
    request: SchedulingRequest,
}

impl Eq for QueueEntry {}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.effective_offset == other.effective_offset
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; reverse so lower effective_offset = higher priority
        other.effective_offset.cmp(&self.effective_offset)
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Configuration for virtual capacity reservations.
#[derive(Debug, Clone)]
pub struct VirtualReservationConfig {
    /// Whether virtual reservations are enabled.
    pub enabled: bool,
    /// Default predicted output length in blocks when expected_output_tokens is absent.
    pub default_osl_blocks: u32,
    /// TTL in milliseconds: virtual reservations older than this are considered expired.
    pub ttl_ms: u64,
}

impl Default for VirtualReservationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_osl_blocks: 16,
            ttl_ms: 500,
        }
    }
}

/// Queue that gates scheduling requests behind a capacity check.
/// When all workers exceed `threshold_frac` utilisation the request is parked in `pending`.
/// When capacity frees up (`update()`), pending requests are scheduled in priority order.
/// If queueing is disabled (threshold_frac is None), requests are scheduled immediately.
pub struct SchedulerQueue<P: SequencePublisher, C: WorkerConfigLike> {
    pending: Mutex<BinaryHeap<QueueEntry>>,
    slots: Arc<ActiveSequencesMultiWorker<P>>,
    workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
    /// Cached threshold fraction; None means queueing is disabled.
    threshold_frac: Option<f64>,
    /// Reference instant for computing arrival offsets.
    start_time: Instant,
    block_size: u32,
    selector: Box<dyn WorkerSelector<C> + Send + Sync>,
    /// Per-worker capacity data from ActiveLoad metrics.
    worker_capacities: RwLock<HashMap<WorkerWithDpRank, WorkerCapacity>>,
    /// Virtual reservation configuration.
    virtual_config: VirtualReservationConfig,
    /// Oracle routing metrics for virtual reservation tracking.
    oracle_metrics: Arc<super::oracle_metrics::OracleRoutingMetrics>,
}

impl<P: SequencePublisher + 'static, C: WorkerConfigLike> SchedulerQueue<P, C> {
    pub fn new(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        threshold_frac: Option<f64>,
        block_size: u32,
        selector: Box<dyn WorkerSelector<C> + Send + Sync>,
        virtual_config: VirtualReservationConfig,
        oracle_metrics: Arc<super::oracle_metrics::OracleRoutingMetrics>,
    ) -> Self {
        if let Some(frac) = threshold_frac {
            tracing::info!("Router queue enabled with threshold fraction {frac}");
        }
        if virtual_config.enabled {
            tracing::info!(
                "Virtual capacity reservations enabled (default_osl_blocks={}, ttl_ms={})",
                virtual_config.default_osl_blocks,
                virtual_config.ttl_ms,
            );
        }
        Self {
            pending: Mutex::new(BinaryHeap::new()),
            slots,
            workers_with_configs,
            threshold_frac,
            start_time: Instant::now(),
            block_size,
            selector,
            worker_capacities: RwLock::new(HashMap::new()),
            virtual_config,
            oracle_metrics,
        }
    }

    /// Build a QueueEntry for a request, computing its effective arrival offset.
    fn make_entry(&self, request: SchedulingRequest) -> QueueEntry {
        let arrival_offset = self.start_time.elapsed();
        let jump = Duration::from_secs_f64(request.priority_jump.max(0.0));
        let effective_offset = arrival_offset.saturating_sub(jump);
        QueueEntry {
            effective_offset,
            request,
        }
    }

    /// Enqueue a new request.
    /// If queueing is disabled or workers have capacity, schedule immediately.
    /// Otherwise park in the pending heap.
    pub async fn enqueue(&self, request: SchedulingRequest) {
        let Some(threshold) = self.threshold_frac else {
            self.schedule(request).await;
            return;
        };

        if self.all_workers_busy(threshold) {
            tracing::debug!("all workers busy, queueing request");
            let entry = self.make_entry(request);
            self.pending.lock().await.push(entry);
        } else {
            self.schedule(request).await;
        }
    }

    /// Called on prefill_complete/free. Drains pending requests while workers have capacity.
    /// Each scheduled request updates active_tokens via add_request, so the busy check
    /// sees fresh state on the next iteration.
    pub async fn update(&self) {
        let Some(threshold) = self.threshold_frac else {
            return;
        };

        loop {
            if self.all_workers_busy(threshold) {
                break;
            }
            let Some(entry) = self.pending.lock().await.pop() else {
                break;
            };
            tracing::debug!("scheduling request from pending queue");
            self.schedule(entry.request).await;
        }
    }

    /// Update capacity data for a worker from an ActiveLoad message.
    /// Only stores data when all three capacity fields (free, evictable, total) are present.
    pub fn update_capacity(&self, worker: WorkerWithDpRank, load: &ActiveLoad) {
        if let (Some(free), Some(evictable), Some(total)) = (
            load.free_kv_blocks,
            load.evictable_kv_blocks,
            load.total_kv_blocks,
        ) {
            let cap = WorkerCapacity {
                free_kv_blocks: free,
                evictable_kv_blocks: evictable,
                total_kv_blocks: total,
                num_running_requests: load.num_running_requests.unwrap_or(0),
                last_updated: Instant::now(),
                active_requests: load.active_requests.clone().unwrap_or_default(),
                virtual_reservation_count: 0, // Real data — reset virtual count
            };
            self.worker_capacities.write().unwrap().insert(worker, cap);
        }
    }

    /// Get a snapshot of all worker capacities (for populating SchedulingRequest).
    ///
    /// When virtual reservations are enabled, checks TTL: if a worker's capacity
    /// has virtual reservations AND has not received a real update within
    /// `virtual_reservation_ttl_ms`, the virtual reservation count is reset
    /// in the returned snapshot (stale virtual data is unreliable).
    pub fn get_capacities(&self) -> HashMap<WorkerWithDpRank, WorkerCapacity> {
        let caps = self.worker_capacities.read().unwrap();

        if !self.virtual_config.enabled {
            return caps.clone();
        }

        let ttl = Duration::from_millis(self.virtual_config.ttl_ms);
        caps.iter()
            .map(|(&worker, cap)| {
                if cap.virtual_reservation_count > 0 && cap.last_updated.elapsed() > ttl {
                    tracing::debug!(
                        worker_id = worker.worker_id,
                        dp_rank = worker.dp_rank,
                        virtual_count = cap.virtual_reservation_count,
                        staleness_ms = cap.last_updated.elapsed().as_millis() as u64,
                        "virtual reservations expired without real update"
                    );
                    let mut reset = cap.clone();
                    reset.virtual_reservation_count = 0;
                    (worker, reset)
                } else {
                    (worker, cap.clone())
                }
            })
            .collect()
    }

    /// Run the full scheduling pipeline for a single request:
    /// compute potential load -> select worker -> respond -> book via add_request.
    async fn schedule(&self, mut request: SchedulingRequest) {
        let (decode_blocks, prefill_tokens) = self.slots.potential_blocks_and_tokens(
            request.token_seq.as_deref(),
            request.isl_tokens,
            request.overlaps.clone(),
        );
        request.decode_blocks = decode_blocks;
        request.prefill_tokens = prefill_tokens;
        request.worker_capacities = self.get_capacities();

        let selection = {
            let workers = self.workers_with_configs.borrow();
            self.selector
                .select_worker(&workers, &request, self.block_size)
        };

        let selection = match selection {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("scheduling failed: {e}");
                request.respond(Err(e));
                return;
            }
        };

        request.respond(Ok(SchedulingResponse {
            best_worker: selection.worker,
            overlap_blocks: selection.overlap_blocks,
            transfer_hint: selection.transfer_hint,
        }));

        // Apply virtual reservation so next scheduling decision sees reduced capacity.
        self.apply_virtual_reservation(
            selection.worker,
            request.isl_tokens,
            request.expected_output_tokens,
            request.priority,
        );

        if !request.update_states {
            return;
        }

        let Some(request_id) = request.maybe_request_id else {
            tracing::error!("No request_id provided to add_request to the slot tracker");
            return;
        };

        if let Err(e) = self
            .slots
            .add_request(SequenceRequest {
                request_id: request_id.clone(),
                token_sequence: request.token_seq,
                isl: request.isl_tokens,
                overlap: selection.overlap_blocks,
                expected_output_tokens: request.expected_output_tokens,
                worker: selection.worker,
                lora_name: request.lora_name.clone(),
            })
            .await
        {
            tracing::warn!("Failed to add request {request_id}: {e}");
        }
    }

    /// Get the block size used by this queue.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Observe a request completion for throughput estimation.
    /// Called before `free()` so that the request's metadata is still available.
    pub fn observe_completion(
        &self,
        worker: &WorkerWithDpRank,
        output_tokens: u32,
        decode_secs: f64,
    ) {
        self.selector
            .observe_request_completion(worker, output_tokens, decode_secs);
    }

    /// Apply a virtual capacity reservation for the selected worker.
    ///
    /// Deducts predicted ISL + OSL blocks from `free_kv_blocks` and pushes a synthetic
    /// `ActiveRequestSummary` into the capacity entry so that subsequent scheduling
    /// decisions see the reduced capacity.
    ///
    /// Self-correcting: the next real `ActiveLoad` from `update_capacity()` replaces
    /// the entire entry, wiping virtual adjustments.
    fn apply_virtual_reservation(
        &self,
        worker: WorkerWithDpRank,
        isl_tokens: usize,
        expected_output_tokens: Option<u32>,
        priority: Option<i32>,
    ) {
        if !self.virtual_config.enabled {
            return;
        }

        let isl_blocks = isl_tokens.div_ceil(self.block_size as usize) as u64;
        let osl_blocks = expected_output_tokens
            .map(|osl| (osl as usize).div_ceil(self.block_size as usize) as u64)
            .unwrap_or(self.virtual_config.default_osl_blocks as u64);
        let total_reserved = isl_blocks + osl_blocks;

        let mut caps = self.worker_capacities.write().unwrap();
        if let Some(cap) = caps.get_mut(&worker) {
            cap.free_kv_blocks = cap.free_kv_blocks.saturating_sub(total_reserved);
            cap.num_running_requests += 1;
            cap.active_requests.push(crate::protocols::ActiveRequestSummary {
                isl_tokens: isl_tokens as u32,
                generated_tokens: 0,
                max_new_tokens: expected_output_tokens
                    .unwrap_or(self.virtual_config.default_osl_blocks * self.block_size),
                priority: priority.unwrap_or(i32::MAX),
                is_prefill: true,
            });
            cap.virtual_reservation_count += 1;

            tracing::debug!(
                worker_id = worker.worker_id,
                dp_rank = worker.dp_rank,
                reserved_blocks = total_reserved,
                remaining_free = cap.free_kv_blocks,
                virtual_count = cap.virtual_reservation_count,
                "applied virtual reservation"
            );

            self.oracle_metrics.record_virtual_reservation(total_reserved);
        }
    }

    /// Check if all workers are busy based on threshold.
    /// Returns true only if ALL workers exceed the threshold (no worker has capacity).
    fn all_workers_busy(&self, threshold: f64) -> bool {
        let active_tokens = self.slots.active_tokens();
        let configs = self.workers_with_configs.borrow();

        for (&worker_id, config) in configs.iter() {
            let dp_size = config.data_parallel_size();
            let max_batched = config
                .max_num_batched_tokens()
                .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS);

            for dp_rank in 0..dp_size {
                let worker = WorkerWithDpRank::new(worker_id, dp_rank);
                let tokens = active_tokens.get(&worker).copied().unwrap_or(0);
                if (tokens as f64) <= threshold * (max_batched as f64) {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use tokio::sync::watch;

    use super::*;
    use crate::protocols::OverlapScores;
    use crate::selector::DefaultWorkerSelector;
    use crate::sequences::ActiveSequencesMultiWorker;
    use crate::test_utils::{NoopSequencePublisher, SimpleWorkerConfig};

    fn make_queue(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        threshold_frac: Option<f64>,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let dp_sizes: HashMap<u64, u32> = (0..num_workers as u64).map(|id| (id, 1)).collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_sizes,
            false,
            0,
            "test",
        ));

        let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        for id in 0..num_workers as u64 {
            configs.insert(
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(isl as u64),
                    ..Default::default()
                },
            );
        }
        let (cfg_tx, cfg_rx) = watch::channel(configs);
        std::mem::forget(cfg_tx);

        let selector = Box::new(DefaultWorkerSelector::default());
        let queue = Arc::new(SchedulerQueue::new(
            Arc::clone(&slots),
            cfg_rx,
            threshold_frac,
            block_size,
            selector,
            VirtualReservationConfig::default(),
            Arc::new(crate::scheduling::oracle_metrics::OracleRoutingMetrics::new_unregistered()),
        ));

        (queue, slots)
    }

    fn make_request(
        request_id: &str,
        isl_tokens: usize,
    ) -> (
        SchedulingRequest,
        tokio::sync::oneshot::Receiver<
            Result<SchedulingResponse, crate::scheduling::types::KvSchedulerError>,
        >,
    ) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let req = SchedulingRequest {
            maybe_request_id: Some(request_id.to_string()),
            token_seq: None,
            isl_tokens,
            overlaps: OverlapScores::default(),
            decode_blocks: HashMap::new(),
            prefill_tokens: HashMap::new(),
            router_config_override: None,
            update_states: true,
            lora_name: None,
            priority_jump: 0.0,
            allowed_worker_ids: None,
            expected_output_tokens: None,
            priority: None,
            worker_capacities: HashMap::new(),
            resp_tx: Some(tx),
        };
        (req, rx)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_flood() {
        let block_size = 16;
        let isl = 512;
        let num_workers = 4;
        let num_tasks = 25;

        let (queue, slots) = make_queue(num_workers, block_size, isl, None);

        let mut handles = Vec::new();
        for i in 0..num_tasks {
            let queue = Arc::clone(&queue);
            let slots = Arc::clone(&slots);
            handles.push(tokio::spawn(async move {
                let req_id = format!("req-{i}");
                let (req, rx) = make_request(&req_id, isl);
                queue.enqueue(req).await;
                let resp = rx.await.expect("oneshot dropped");
                let resp = resp.expect("scheduling failed");
                assert!(resp.best_worker.worker_id < num_workers as u64);

                slots.mark_prefill_completed(&req_id).await.unwrap();
                slots.free(&req_id).await.unwrap();
                queue.update().await;
            }));
        }

        for h in handles {
            h.await.expect("task panicked");
        }

        let active = slots.active_tokens();
        for (worker, tokens) in &active {
            assert_eq!(
                *tokens, 0,
                "worker {worker:?} still has {tokens} active tokens"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_queueing_under_pressure() {
        let block_size = 16;
        let isl = 512;
        let num_workers = 2;
        let num_requests = 10;

        let (queue, slots) = make_queue(num_workers, block_size, isl, Some(0.0));

        let mut receivers = Vec::new();
        let mut req_ids = Vec::new();

        for i in 0..num_requests {
            let req_id = format!("pressure-{i}");
            let (req, rx) = make_request(&req_id, isl);
            queue.enqueue(req).await;
            receivers.push(rx);
            req_ids.push(req_id);
        }

        // Drain pending by cycling mark_prefill_completed + free + update
        // on already-scheduled requests until all receivers have a response.
        for _ in 0..num_requests {
            queue.update().await;
            for rid in &req_ids {
                let _ = slots.mark_prefill_completed(rid).await;
                let _ = slots.free(rid).await;
            }
        }
        queue.update().await;

        let mut ok_count = 0;
        for mut rx in receivers {
            if let Ok(result) = rx.try_recv() {
                result.expect("scheduling returned error");
                ok_count += 1;
            }
        }
        assert_eq!(ok_count, num_requests, "not all requests were scheduled");
    }

    #[tokio::test]
    async fn test_no_workers_returns_error() {
        let (queue, _slots) = make_queue(0, 16, 512, None);

        let (req, rx) = make_request("lonely-req", 512);
        queue.enqueue(req).await;

        let resp = rx.await.expect("oneshot dropped");
        assert!(
            matches!(
                resp,
                Err(crate::scheduling::types::KvSchedulerError::NoEndpoints)
            ),
            "expected NoEndpoints, got {resp:?}"
        );
    }

    #[test]
    fn test_capacity_update_from_active_load() {
        let worker = WorkerWithDpRank::from_worker_id(1);
        let load = ActiveLoad {
            worker_id: 1,
            dp_rank: 0,
            active_decode_blocks: Some(100),
            active_prefill_tokens: None,
            free_kv_blocks: Some(500),
            evictable_kv_blocks: Some(200),
            total_kv_blocks: Some(1000),
            num_running_requests: Some(5),
            active_requests: None,
        };

        let (queue, _slots) = make_queue(2, 16, 512, None);
        queue.update_capacity(worker, &load);

        let caps = queue.get_capacities();
        let cap = caps.get(&worker).expect("capacity should be stored");
        assert_eq!(cap.free_kv_blocks, 500);
        assert_eq!(cap.evictable_kv_blocks, 200);
        assert_eq!(cap.total_kv_blocks, 1000);
        assert_eq!(cap.num_running_requests, 5);
    }

    #[test]
    fn test_capacity_not_updated_without_capacity_fields() {
        let worker = WorkerWithDpRank::from_worker_id(1);
        let load = ActiveLoad {
            worker_id: 1,
            dp_rank: 0,
            active_decode_blocks: Some(100),
            active_prefill_tokens: None,
            free_kv_blocks: None,
            evictable_kv_blocks: None,
            total_kv_blocks: None,
            num_running_requests: None,
            active_requests: None,
        };

        let (queue, _slots) = make_queue(2, 16, 512, None);
        queue.update_capacity(worker, &load);

        let caps = queue.get_capacities();
        assert!(
            caps.get(&worker).is_none(),
            "should not store without capacity data"
        );
    }

    fn make_queue_with_virtual(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        threshold_frac: Option<f64>,
        virtual_config: VirtualReservationConfig,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let dp_sizes: HashMap<u64, u32> = (0..num_workers as u64).map(|id| (id, 1)).collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_sizes,
            false,
            0,
            "test",
        ));

        let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        for id in 0..num_workers as u64 {
            configs.insert(
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(isl as u64),
                    ..Default::default()
                },
            );
        }
        let (cfg_tx, cfg_rx) = watch::channel(configs);
        std::mem::forget(cfg_tx);

        let selector = Box::new(DefaultWorkerSelector::default());
        let queue = Arc::new(SchedulerQueue::new(
            Arc::clone(&slots),
            cfg_rx,
            threshold_frac,
            block_size,
            selector,
            virtual_config,
            Arc::new(crate::scheduling::oracle_metrics::OracleRoutingMetrics::new_unregistered()),
        ));

        (queue, slots)
    }

    /// Helper to set up worker capacity on a queue.
    fn setup_capacity(queue: &SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig>, worker_id: u64, free: u64, evictable: u64, total: u64) {
        let worker = WorkerWithDpRank::from_worker_id(worker_id);
        let load = ActiveLoad {
            worker_id,
            dp_rank: 0,
            active_decode_blocks: Some(0),
            active_prefill_tokens: None,
            free_kv_blocks: Some(free),
            evictable_kv_blocks: Some(evictable),
            total_kv_blocks: Some(total),
            num_running_requests: Some(0),
            active_requests: None,
        };
        queue.update_capacity(worker, &load);
    }

    #[test]
    fn test_virtual_reservation_deducts_blocks() {
        let (queue, _slots) = make_queue_with_virtual(
            2, 16, 512, None,
            VirtualReservationConfig { enabled: true, default_osl_blocks: 16, ttl_ms: 500 },
        );
        let worker = WorkerWithDpRank::from_worker_id(0);
        setup_capacity(&queue, 0, 500, 200, 1000);

        // isl_tokens=256, expected_output_tokens=Some(512)
        // isl_blocks = 256/16 = 16, osl_blocks = 512/16 = 32, total = 48
        queue.apply_virtual_reservation(worker, 256, Some(512), None);

        let caps = queue.get_capacities();
        let cap = caps.get(&worker).expect("capacity should exist");
        assert_eq!(cap.free_kv_blocks, 500 - 48, "free blocks should be deducted");
        assert_eq!(cap.active_requests.len(), 1, "synthetic request should be added");
        assert_eq!(cap.virtual_reservation_count, 1);
    }

    #[test]
    fn test_virtual_reservation_overwritten_by_real_update() {
        let (queue, _slots) = make_queue_with_virtual(
            2, 16, 512, None,
            VirtualReservationConfig { enabled: true, default_osl_blocks: 16, ttl_ms: 500 },
        );
        let worker = WorkerWithDpRank::from_worker_id(0);
        setup_capacity(&queue, 0, 500, 200, 1000);

        queue.apply_virtual_reservation(worker, 256, Some(512), None);
        assert_eq!(queue.get_capacities()[&worker].free_kv_blocks, 452);

        // Real update restores capacity
        setup_capacity(&queue, 0, 500, 200, 1000);

        let caps = queue.get_capacities();
        let cap = caps.get(&worker).unwrap();
        assert_eq!(cap.free_kv_blocks, 500, "real update should restore free blocks");
        assert_eq!(cap.virtual_reservation_count, 0, "real update should reset virtual count");
    }

    #[test]
    fn test_virtual_reservation_no_op_when_disabled() {
        let (queue, _slots) = make_queue_with_virtual(
            2, 16, 512, None,
            VirtualReservationConfig { enabled: false, default_osl_blocks: 16, ttl_ms: 500 },
        );
        let worker = WorkerWithDpRank::from_worker_id(0);
        setup_capacity(&queue, 0, 500, 200, 1000);

        queue.apply_virtual_reservation(worker, 256, Some(512), None);

        let caps = queue.get_capacities();
        let cap = caps.get(&worker).unwrap();
        assert_eq!(cap.free_kv_blocks, 500, "disabled reservation should not change free blocks");
    }

    #[test]
    fn test_virtual_reservation_default_osl_blocks() {
        let (queue, _slots) = make_queue_with_virtual(
            2, 16, 512, None,
            VirtualReservationConfig { enabled: true, default_osl_blocks: 32, ttl_ms: 500 },
        );
        let worker = WorkerWithDpRank::from_worker_id(0);
        setup_capacity(&queue, 0, 500, 200, 1000);

        // isl_tokens=256, expected_output_tokens=None (uses default_osl_blocks=32)
        // isl_blocks=16, osl_blocks=32, total=48
        queue.apply_virtual_reservation(worker, 256, None, None);

        let caps = queue.get_capacities();
        let cap = caps.get(&worker).unwrap();
        assert_eq!(cap.free_kv_blocks, 500 - 48, "default OSL blocks should be used");
        // Check synthetic request's max_new_tokens = default_osl_blocks * block_size = 32 * 16 = 512
        assert_eq!(cap.active_requests[0].max_new_tokens, 512);
    }

    #[test]
    fn test_virtual_reservation_cumulative() {
        let (queue, _slots) = make_queue_with_virtual(
            2, 16, 512, None,
            VirtualReservationConfig { enabled: true, default_osl_blocks: 16, ttl_ms: 500 },
        );
        let worker = WorkerWithDpRank::from_worker_id(0);
        setup_capacity(&queue, 0, 500, 200, 1000);

        // Each reservation: isl=256 (16 blocks) + osl=Some(256) (16 blocks) = 32 blocks
        queue.apply_virtual_reservation(worker, 256, Some(256), None);
        queue.apply_virtual_reservation(worker, 256, Some(256), None);
        queue.apply_virtual_reservation(worker, 256, Some(256), None);

        let caps = queue.get_capacities();
        let cap = caps.get(&worker).unwrap();
        assert_eq!(cap.free_kv_blocks, 500 - 96, "3 reservations of 32 blocks each = 96");
        assert_eq!(cap.virtual_reservation_count, 3);
        assert_eq!(cap.active_requests.len(), 3);
    }
}
