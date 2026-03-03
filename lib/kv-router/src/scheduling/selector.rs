// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use rand::Rng;

use super::config::KvRouterConfig;
use super::throughput::ThroughputEstimator;
use super::oracle_metrics::OracleRoutingMetrics;
use super::transfer_metrics::TransferDecisionMetrics;
use super::types::{KvSchedulerError, SchedulingRequest, WorkerCapacity};
use crate::protocols::{
    TransferHint, WorkerConfigLike, WorkerId, WorkerSelectionResult, WorkerWithDpRank,
};

/// A trait that users can implement to define custom selection logic.
///
/// Generic over `C` so that the scheduling layer does not depend on a concrete config type.
pub trait WorkerSelector<C: WorkerConfigLike> {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError>;

    /// Record a completed request for throughput estimation.
    /// Default implementation is a no-op (for selectors that don't track throughput).
    fn observe_request_completion(
        &self,
        _worker: &WorkerWithDpRank,
        _output_tokens: u32,
        _decode_secs: f64,
    ) {
        // default: no-op
    }
}

/// Helper function for softmax sampling.
/// Returns a vec of workers: multiple if tied, single if sampled.
fn softmax_sample(
    logits: &HashMap<WorkerWithDpRank, f64>,
    temperature: f64,
) -> Vec<WorkerWithDpRank> {
    if logits.is_empty() {
        panic!("Empty logits for softmax sampling");
    }

    // Guard: if temperature is 0, return all keys with the smallest logit value (ties)
    if temperature == 0.0 {
        let min_logit = logits.values().fold(f64::INFINITY, |a, &b| a.min(b));

        let min_keys: Vec<_> = logits
            .iter()
            .filter(|&(_, &v)| v == min_logit)
            .map(|(k, _)| *k)
            .collect();

        return min_keys;
    }

    let keys: Vec<_> = logits.keys().copied().collect();
    let values: Vec<_> = logits.values().copied().collect();

    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let probabilities = if min_val == max_val {
        vec![1.0 / keys.len() as f64; keys.len()]
    } else {
        // Fused normalize -> negate -> scale -> exp, then normalize probabilities
        let range = max_val - min_val;
        let scaled: Vec<f64> = values.iter().map(|&v| -(v / range) / temperature).collect();
        let max_scaled = scaled.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut probs: Vec<f64> = scaled.iter().map(|&v| (v - max_scaled).exp()).collect();
        let sum: f64 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        probs
    };

    let mut rng = rand::rng();
    let sample: f64 = rng.random();

    let mut cumsum = 0.0;
    for (i, &prob) in probabilities.iter().enumerate() {
        cumsum += prob;
        if sample <= cumsum {
            return vec![keys[i]];
        }
    }

    // Fallback to last key (shouldn't normally reach here)
    vec![keys[keys.len() - 1]]
}

/// Config for OSL-weighted effective load computation, passed to `compute_worker_logits`.
struct EffectiveLoadConfig {
    high_priority_weight: f64,
    low_priority_weight: f64,
}

/// Configuration for headroom projection in the cost function.
struct HeadroomConfig {
    headroom_weight: f64,
    throughput: ThroughputEstimator,
}

/// Compute effective decode load for a worker based on per-request summaries.
/// Returns `Some(load)` when per-request data is available, `None` otherwise
/// (signaling the caller to fall back to raw decode_blocks).
fn compute_effective_load(
    worker: &WorkerWithDpRank,
    capacities: &HashMap<WorkerWithDpRank, WorkerCapacity>,
    new_request_priority: Option<i32>,
    block_size: u32,
    high_priority_weight: f64,
    low_priority_weight: f64,
) -> Option<f64> {
    let cap = capacities.get(worker)?;
    if cap.active_requests.is_empty() {
        return None; // No per-request data, signal caller to use fallback
    }

    let new_priority = new_request_priority.unwrap_or(i32::MAX);

    let load = cap
        .active_requests
        .iter()
        .map(|req| {
            let remaining_tokens = req.max_new_tokens.saturating_sub(req.generated_tokens);
            let remaining_blocks = remaining_tokens as f64 / block_size as f64;

            let priority_weight = if req.priority < new_priority {
                high_priority_weight // Ahead of us in priority → more contention
            } else if req.priority == new_priority {
                1.0
            } else {
                low_priority_weight // Behind us → less contention
            };

            remaining_blocks * priority_weight
        })
        .sum::<f64>();

    Some(load)
}

/// Project how many KV blocks will be freed on a worker within `horizon_secs`.
///
/// Sums the total blocks (ISL + OSL) of each in-flight request whose estimated
/// remaining decode time is <= `horizon_secs`. These blocks become available when
/// those requests finish.
///
/// Returns `None` when capacity data or active_requests are missing.
fn projected_free_blocks(
    worker: &WorkerWithDpRank,
    capacities: &HashMap<WorkerWithDpRank, WorkerCapacity>,
    throughput: &ThroughputEstimator,
    horizon_secs: f64,
    block_size: u32,
) -> Option<f64> {
    let cap = capacities.get(worker)?;
    if cap.active_requests.is_empty() {
        return None;
    }

    let mut freed = 0.0f64;
    for req in &cap.active_requests {
        if req.is_prefill {
            continue; // prefill requests haven't started generating yet
        }
        let remaining_tokens = req.max_new_tokens.saturating_sub(req.generated_tokens);
        let time_to_finish = throughput.estimate_remaining_seconds(worker, remaining_tokens);

        if time_to_finish <= horizon_secs {
            // This request finishes within the horizon → its KV blocks will be freed
            let total_request_blocks =
                (req.isl_tokens as f64 + req.max_new_tokens as f64) / block_size as f64;
            freed += total_request_blocks;
        }
    }

    Some(cap.free_kv_blocks as f64 + freed)
}

/// Adjust overlap and headroom weights based on request priority.
///
/// High-priority requests (priority <= threshold): boost headroom weight (fast start),
/// reduce overlap weight (less cache dependency).
/// Normal/background requests: use base weights unchanged.
fn priority_adjusted_weights(
    base_overlap_weight: f64,
    base_headroom_weight: f64,
    priority: Option<i32>,
    high_priority_threshold: i32,
) -> (f64, f64) {
    let p = match priority {
        Some(p) => p,
        None => return (base_overlap_weight, base_headroom_weight),
    };

    if p <= high_priority_threshold {
        // Urgency: 1.0 for priority=0 (or negative), 0.0 at threshold
        let urgency = if high_priority_threshold > 0 {
            (1.0 - (p.max(0) as f64 / high_priority_threshold as f64)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let overlap_w = base_overlap_weight * (1.0 - 0.5 * urgency);
        let headroom_w = base_headroom_weight * (1.0 + 2.0 * urgency);
        (overlap_w, headroom_w)
    } else {
        (base_overlap_weight, base_headroom_weight)
    }
}

/// Compute per-worker logits (cost scores) for routing.
///
/// Returns a map of (worker, logit) where lower logit = better choice.
/// The cost function is: logit = overlap_weight * prefill_blocks + load_term
/// where load_term is either effective load (OSL-weighted) or raw decode_blocks.
fn compute_worker_logits<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    block_size: u32,
    overlap_weight: f64,
    effective_load_config: Option<&EffectiveLoadConfig>,
    headroom_config: Option<&HeadroomConfig>,
    oracle_metrics: Option<&OracleRoutingMetrics>,
) -> HashMap<WorkerWithDpRank, f64> {
    let allowed_ids = request.allowed_worker_ids.as_ref();
    let isl = request.isl_tokens;
    let overlaps = &request.overlaps.scores;
    let decode_blocks = &request.decode_blocks;
    let prefill_tokens = &request.prefill_tokens;

    let mut worker_logits = HashMap::new();

    for (worker_id, config) in workers
        .iter()
        .filter(|(wid, _)| allowed_ids.is_none_or(|ids| ids.contains(wid)))
    {
        let data_parallel_size = config.data_parallel_size();

        for dp_rank in 0..data_parallel_size {
            let worker = WorkerWithDpRank::new(*worker_id, dp_rank);

            let overlap = *overlaps.get(&worker).unwrap_or(&0);

            let prefill_token = *prefill_tokens.get(&worker).unwrap_or(&isl);
            let potential_prefill_block = (prefill_token as f64) / (block_size as f64);

            let decode_block = *decode_blocks
                .get(&worker)
                .unwrap_or(&(potential_prefill_block.floor() as usize))
                as f64;

            // When effective load is enabled and per-request data is available,
            // use OSL-weighted load instead of raw decode blocks.
            let effective_load_value = if let Some(elc) = effective_load_config {
                compute_effective_load(
                    &worker,
                    &request.worker_capacities,
                    request.priority,
                    block_size,
                    elc.high_priority_weight,
                    elc.low_priority_weight,
                )
            } else {
                None
            };
            let load_term = effective_load_value.unwrap_or(decode_block);

            // Headroom bonus: workers where in-flight requests are about to finish
            // get a lower logit (more attractive).
            let (headroom_bonus, headroom_ratio_value) = if let Some(hc) = headroom_config {
                let prefill_time_secs = request.isl_tokens as f64 / 5000.0;
                if let Some(projected) = projected_free_blocks(
                    &worker,
                    &request.worker_capacities,
                    &hc.throughput,
                    prefill_time_secs,
                    block_size,
                ) {
                    let total = request
                        .worker_capacities
                        .get(&worker)
                        .map(|c| c.total_kv_blocks)
                        .unwrap_or(1)
                        .max(1) as f64;
                    let ratio = (projected / total).clamp(0.0, 1.0);
                    (hc.headroom_weight * ratio, ratio)
                } else {
                    (0.0, 0.0)
                }
            } else {
                (0.0, 0.0)
            };

            let logit = overlap_weight * potential_prefill_block + load_term - headroom_bonus;

            // Emit oracle routing metrics
            if let Some(m) = oracle_metrics {
                m.record_logit(logit);
                if let Some(eff) = effective_load_value {
                    m.record_effective_load(eff);
                }
                m.record_headroom_ratio(headroom_ratio_value);
            }

            worker_logits.insert(worker, logit);

            let virtual_count = request
                .worker_capacities
                .get(&worker)
                .map(|c| c.virtual_reservation_count)
                .unwrap_or(0);

            tracing::info!(
                "Formula for worker_id={} dp_rank={:?} with {overlap} cached blocks: {logit:.3} \
                 = {overlap_weight:.1} * {potential_prefill_block:.3} + {load_term:.3} - {headroom_bonus:.3} \
                 (raw_decode={decode_block:.3}, eff_load={}, \
                 headroom_ratio={headroom_ratio_value:.3}, virtual_count={virtual_count})",
                worker.worker_id,
                worker.dp_rank,
                effective_load_value
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "n/a".to_string()),
            );
        }
    }

    worker_logits
}

/// Select the best worker from logits using softmax sampling and tree-size tie-breaking.
fn select_from_logits(
    worker_logits: &HashMap<WorkerWithDpRank, f64>,
    request: &SchedulingRequest,
    temperature: f64,
) -> WorkerWithDpRank {
    let candidates = softmax_sample(worker_logits, temperature);

    if candidates.len() > 1 {
        tracing::info!("Multiple workers tied with same logit, using tree size as tie-breaker");
        let tree_sizes: Vec<(usize, &WorkerWithDpRank)> = candidates
            .iter()
            .map(|w| (request.overlaps.tree_sizes.get(w).copied().unwrap_or(0), w))
            .collect();

        if tree_sizes.iter().all(|(s, _)| *s == tree_sizes[0].0) {
            let idx = rand::rng().random_range(0..candidates.len());
            candidates[idx]
        } else {
            *tree_sizes.iter().min_by_key(|(s, _)| *s).unwrap().1
        }
    } else {
        candidates[0]
    }
}

/// Log the selected worker with useful debug info.
fn log_selection<C: WorkerConfigLike>(
    best_worker: WorkerWithDpRank,
    worker_logits: &HashMap<WorkerWithDpRank, f64>,
    overlaps: &rustc_hash::FxHashMap<WorkerWithDpRank, u32>,
    request: &SchedulingRequest,
    workers: &HashMap<WorkerId, C>,
) {
    let best_logit = worker_logits[&best_worker];
    let best_overlap = *overlaps.get(&best_worker).unwrap_or(&0);
    let total_blocks_info = workers
        .get(&best_worker.worker_id)
        .and_then(|cfg| cfg.total_kv_blocks())
        .map(|blocks| format!(", total blocks: {}", blocks))
        .unwrap_or_default();
    let tree_size = request
        .overlaps
        .tree_sizes
        .get(&best_worker)
        .copied()
        .unwrap_or(0);

    tracing::info!(
        "Selected worker: worker_id={} dp_rank={:?}, logit: {:.3}, cached blocks: {}, tree size: {}{}",
        best_worker.worker_id,
        best_worker.dp_rank,
        best_logit,
        best_overlap,
        tree_size,
        total_blocks_info
    );
}

/// Default implementation matching the Python _cost_function.
#[derive(Debug, Clone, Default)]
pub struct DefaultWorkerSelector {
    pub kv_router_config: KvRouterConfig,
}

impl DefaultWorkerSelector {
    pub fn new(kv_router_config: Option<KvRouterConfig>) -> Self {
        Self {
            kv_router_config: kv_router_config.unwrap_or_default(),
        }
    }
}

impl<C: WorkerConfigLike> WorkerSelector<C> for DefaultWorkerSelector {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        assert!(request.isl_tokens > 0);

        let allowed_ids = request.allowed_worker_ids.as_ref();
        if allowed_ids.map_or(workers.is_empty(), |ids| {
            !workers.keys().any(|wid| ids.contains(wid))
        }) {
            return Err(KvSchedulerError::NoEndpoints);
        }

        let request_blocks = request.isl_tokens.div_ceil(block_size as usize);
        let overlaps = &request.overlaps.scores;

        let overlap_weight = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.overlap_score_weight)
            .unwrap_or(self.kv_router_config.overlap_score_weight);

        let worker_logits =
            compute_worker_logits(workers, request, block_size, overlap_weight, None, None, None);

        let temperature = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.router_temperature)
            .unwrap_or(self.kv_router_config.router_temperature);

        let best_worker = select_from_logits(&worker_logits, request, temperature);

        log_selection(best_worker, &worker_logits, overlaps, request, workers);

        Ok(WorkerSelectionResult {
            worker: best_worker,
            required_blocks: request_blocks as u64,
            overlap_blocks: overlaps.get(&best_worker).copied().unwrap_or(0),
            transfer_hint: None,
        })
    }
}

/// Worker selector that considers cross-worker KV cache transfer.
///
/// When enabled, this selector evaluates whether routing to a less-loaded
/// worker WITH a transfer hint produces a lower total cost than routing
/// to the cache-optimal worker.
///
/// Cost without transfer (same as DefaultWorkerSelector):
///   cost(w) = alpha * prefill_blocks(w) + decode_blocks(w)
///
/// Cost with transfer (target=t, source=s):
///   cost_transfer(t, s) = alpha * remaining_prefill(t,s) + decode_blocks(t) + beta * transfer_blocks
///   where remaining_prefill = prefill_blocks(t) - transferable_blocks
///   and   transfer_blocks = min(overlap(s) - overlap(t), max_transfer_blocks)
///
/// Transfer is chosen when:
///   cost_transfer(best_target, best_source) < cost(best_cache_worker)
///   AND decode_blocks(best_cache_worker) - decode_blocks(best_target) >= min_queue_advantage
#[derive(Debug)]
pub struct TransferAwareWorkerSelector {
    pub kv_router_config: KvRouterConfig,
    pub metrics: Arc<TransferDecisionMetrics>,
    pub oracle_metrics: Arc<OracleRoutingMetrics>,
    pub throughput: Arc<RwLock<ThroughputEstimator>>,
}

impl Default for TransferAwareWorkerSelector {
    fn default() -> Self {
        let cfg = KvRouterConfig::default();
        Self {
            throughput: Arc::new(RwLock::new(ThroughputEstimator::new(
                cfg.default_decode_tps,
                cfg.throughput_ema_alpha,
            ))),
            kv_router_config: cfg,
            metrics: Arc::new(TransferDecisionMetrics::new_unregistered()),
            oracle_metrics: Arc::new(OracleRoutingMetrics::new_unregistered()),
        }
    }
}

impl TransferAwareWorkerSelector {
    pub fn new(kv_router_config: Option<KvRouterConfig>) -> Self {
        let cfg = kv_router_config.unwrap_or_default();
        let throughput = ThroughputEstimator::new(cfg.default_decode_tps, cfg.throughput_ema_alpha);
        Self {
            kv_router_config: cfg,
            metrics: Arc::new(TransferDecisionMetrics::new_unregistered()),
            oracle_metrics: Arc::new(OracleRoutingMetrics::new_unregistered()),
            throughput: Arc::new(RwLock::new(throughput)),
        }
    }

    #[cfg(feature = "metrics")]
    pub fn with_component(
        kv_router_config: Option<KvRouterConfig>,
        component: &dynamo_runtime::component::Component,
    ) -> Self {
        let cfg = kv_router_config.unwrap_or_default();
        let throughput = ThroughputEstimator::new(cfg.default_decode_tps, cfg.throughput_ema_alpha);
        Self {
            kv_router_config: cfg,
            metrics: TransferDecisionMetrics::from_component(component),
            oracle_metrics: OracleRoutingMetrics::from_component(component),
            throughput: Arc::new(RwLock::new(throughput)),
        }
    }

    /// Estimate total KV blocks needed for a transfer to the target worker.
    /// Includes transferred blocks + remaining ISL blocks + predicted OSL blocks.
    fn estimate_transfer_blocks_needed(
        request: &SchedulingRequest,
        target_overlap: u32,
        transferable: u32,
        block_size: u32,
    ) -> u64 {
        let isl = request.isl_tokens;
        let transfer_blocks = transferable as u64;
        let remaining_isl_blocks = isl
            .saturating_sub((target_overlap + transferable) as usize * block_size as usize)
            .div_ceil(block_size as usize) as u64;
        let predicted_osl_blocks = request
            .expected_output_tokens
            .map(|osl| (osl as usize).div_ceil(block_size as usize) as u64)
            .unwrap_or(0);
        transfer_blocks + remaining_isl_blocks + predicted_osl_blocks
    }

    /// Evaluate whether a cross-worker transfer would be beneficial.
    /// Returns Ok((target, source, transferable_blocks, transfer_cost, cache_cost))
    /// if transfer is cheaper, or Err(reason) explaining why not.
    fn evaluate_transfer(
        &self,
        worker_logits: &HashMap<WorkerWithDpRank, f64>,
        request: &SchedulingRequest,
        block_size: u32,
        overlap_weight: f64,
    ) -> Result<(WorkerWithDpRank, WorkerWithDpRank, u32, f64, f64), &'static str> {
        let cfg = &self.kv_router_config;
        let overlaps = &request.overlaps.scores;
        let decode_blocks = &request.decode_blocks;

        // Find the best-cache worker (highest overlap)
        let (&best_cache_worker, &best_cache_overlap) = overlaps
            .iter()
            .filter(|(w, _)| worker_logits.contains_key(w))
            .max_by_key(|&(_, score)| *score)
            .ok_or("no_overlap")?;

        if best_cache_overlap == 0 {
            return Err("no_overlap");// No cached blocks anywhere
        }

        let source_decode = *decode_blocks.get(&best_cache_worker).unwrap_or(&0);

        // Find the best target: worker with lowest logit that is NOT the best-cache worker.
        // Among workers with the same lowest logit, prefer the one with fewest decode blocks.
        let best_target = worker_logits
            .iter()
            .filter(|(w, _)| *w != &best_cache_worker)
            .min_by(|(w_a, logit_a), (w_b, logit_b)| {
                logit_a.partial_cmp(logit_b).unwrap_or(std::cmp::Ordering::Equal).then_with(|| {
                    let da = decode_blocks.get(w_a).unwrap_or(&0);
                    let db = decode_blocks.get(w_b).unwrap_or(&0);
                    da.cmp(db)
                })
            })
            .map(|(w, _)| *w)
            .ok_or("no_target")?;

        let target_decode = *decode_blocks.get(&best_target).unwrap_or(&0);
        let target_overlap = *overlaps.get(&best_target).unwrap_or(&0);

        // Check queue advantage: source must be sufficiently more loaded than target
        if source_decode.saturating_sub(target_decode) < cfg.min_transfer_queue_advantage as usize {
            return Err("queue_advantage_low");
        }

        // Check the target is actually missing blocks the source has
        if target_overlap >= best_cache_overlap {
            return Err("target_has_cache");
        }

        // Compute transferable blocks (capped by max_transfer_blocks)
        let transferable = (best_cache_overlap - target_overlap).min(cfg.max_transfer_blocks);

        // Memory feasibility check: ensure target has enough free + evictable blocks
        // for the transferred blocks + remaining ISL blocks + predicted OSL blocks.
        if !request.worker_capacities.is_empty() {
            if let Some(target_cap) = request.worker_capacities.get(&best_target) {
                let total_needed = Self::estimate_transfer_blocks_needed(
                    request, target_overlap, transferable, block_size,
                );
                let available = target_cap.free_kv_blocks + target_cap.evictable_kv_blocks;
                if total_needed > available {
                    return Err("target_insufficient_memory");
                }
            }
        }

        // Compute transfer cost:
        // After transfer, the target will have (target_overlap + transferable) cached blocks.
        // Remaining prefill = total request tokens that still need prefill after transfer.
        let isl = request.isl_tokens;
        let tokens_covered_by_transfer = (target_overlap + transferable) as usize * block_size as usize;
        let remaining_prefill_tokens = isl.saturating_sub(tokens_covered_by_transfer);
        let remaining_prefill_blocks = (remaining_prefill_tokens as f64) / (block_size as f64);

        let mut transfer_cost = overlap_weight * remaining_prefill_blocks
            + target_decode as f64
            + cfg.transfer_cost_weight * transferable as f64;

        // Memory pressure: soft penalty based on how much of the target's capacity is needed.
        if !request.worker_capacities.is_empty() {
            let memory_pressure = if let Some(target_cap) = request.worker_capacities.get(&best_target) {
                let total_needed = Self::estimate_transfer_blocks_needed(
                    request, target_overlap, transferable, block_size,
                ) as f64;
                let available = (target_cap.free_kv_blocks + target_cap.evictable_kv_blocks) as f64;
                if available > 0.0 {
                    (total_needed / available).min(2.0)
                } else {
                    2.0
                }
            } else {
                0.5 // Unknown capacity for this worker, moderate assumption
            };
            // Emit memory pressure metric
            self.oracle_metrics.record_memory_pressure(memory_pressure);

            transfer_cost += cfg.memory_pressure_weight * memory_pressure;
        }

        // Headroom bonus for transfer target: lower transfer cost when target has capacity
        // about to free up (requests finishing soon).
        if cfg.headroom_weight > 0.0 {
            if let Ok(throughput) = self.throughput.read() {
                let prefill_time_secs = request.isl_tokens as f64 / 5000.0;
                if let Some(projected) = projected_free_blocks(
                    &best_target,
                    &request.worker_capacities,
                    &throughput,
                    prefill_time_secs,
                    block_size,
                ) {
                    if let Some(target_cap) = request.worker_capacities.get(&best_target) {
                        let total = target_cap.total_kv_blocks.max(1) as f64;
                        let headroom_ratio = (projected / total).clamp(0.0, 1.0);
                        transfer_cost -= cfg.headroom_weight * headroom_ratio;
                    }
                }
            }
        }

        // Compare with best-cache worker's cost (the default choice)
        let cache_cost = *worker_logits.get(&best_cache_worker).ok_or("no_overlap")?;

        if transfer_cost < cache_cost {
            tracing::info!(
                "Transfer beneficial: target=worker_{} (cost={:.3}) vs cache_worker_{} (cost={:.3}), \
                 transferring {} blocks from worker_{}",
                best_target.worker_id,
                transfer_cost,
                best_cache_worker.worker_id,
                cache_cost,
                transferable,
                best_cache_worker.worker_id,
            );
            Ok((best_target, best_cache_worker, transferable, transfer_cost, cache_cost))
        } else {
            Err("not_cheaper")
        }
    }
}

impl<C: WorkerConfigLike> WorkerSelector<C> for TransferAwareWorkerSelector {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        assert!(request.isl_tokens > 0);

        let allowed_ids = request.allowed_worker_ids.as_ref();
        if allowed_ids.map_or(workers.is_empty(), |ids| {
            !workers.keys().any(|wid| ids.contains(wid))
        }) {
            return Err(KvSchedulerError::NoEndpoints);
        }

        let request_blocks = request.isl_tokens.div_ceil(block_size as usize);
        let overlaps = &request.overlaps.scores;

        let base_overlap_weight = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.overlap_score_weight)
            .unwrap_or(self.kv_router_config.overlap_score_weight);
        let base_headroom_weight = self.kv_router_config.headroom_weight;

        // Priority-adaptive weight adjustment
        let (overlap_weight, headroom_weight) =
            if self.kv_router_config.priority_adaptive_weights {
                priority_adjusted_weights(
                    base_overlap_weight,
                    base_headroom_weight,
                    request.priority,
                    self.kv_router_config.high_priority_threshold,
                )
            } else {
                (base_overlap_weight, base_headroom_weight)
            };

        let effective_load_config = if self.kv_router_config.use_effective_load {
            Some(EffectiveLoadConfig {
                high_priority_weight: self.kv_router_config.high_priority_contention_weight,
                low_priority_weight: self.kv_router_config.low_priority_contention_weight,
            })
        } else {
            None
        };

        let headroom_config = if headroom_weight > 0.0 {
            self.throughput
                .read()
                .ok()
                .map(|t| HeadroomConfig {
                    headroom_weight,
                    throughput: t.clone(),
                })
        } else {
            None
        };

        let worker_logits = compute_worker_logits(
            workers,
            request,
            block_size,
            overlap_weight,
            effective_load_config.as_ref(),
            headroom_config.as_ref(),
            Some(&self.oracle_metrics),
        );

        // Evaluate transfer opportunity
        if self.kv_router_config.enable_kv_transfer {
            match self.evaluate_transfer(&worker_logits, request, block_size, overlap_weight) {
                Ok((target, source, transferable, transfer_cost, cache_cost)) => {
                    self.metrics.record_decision("transfer");
                    self.metrics
                        .record_transfer_chosen(transferable, transfer_cost, cache_cost);

                    log_selection(target, &worker_logits, overlaps, request, workers);

                    return Ok(WorkerSelectionResult {
                        worker: target,
                        required_blocks: request_blocks as u64,
                        overlap_blocks: overlaps.get(&target).copied().unwrap_or(0),
                        transfer_hint: Some(TransferHint {
                            source_worker: source,
                            num_blocks: transferable,
                        }),
                    });
                }
                Err(reason) => {
                    self.metrics.record_no_transfer_reason(reason);
                }
            }
        } else {
            self.metrics.record_no_transfer_reason("disabled");
        }

        // Fall through to default selection (no transfer)
        self.metrics.record_decision("no_transfer");
        // Record the best (lowest) logit as the baseline cost
        if let Some(&best_cost) = worker_logits.values().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)) {
            self.metrics.record_no_transfer_cost(best_cost);
        }

        let temperature = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.router_temperature)
            .unwrap_or(self.kv_router_config.router_temperature);

        let best_worker = select_from_logits(&worker_logits, request, temperature);

        log_selection(best_worker, &worker_logits, overlaps, request, workers);

        Ok(WorkerSelectionResult {
            worker: best_worker,
            required_blocks: request_blocks as u64,
            overlap_blocks: overlaps.get(&best_worker).copied().unwrap_or(0),
            transfer_hint: None,
        })
    }

    fn observe_request_completion(
        &self,
        worker: &WorkerWithDpRank,
        output_tokens: u32,
        decode_secs: f64,
    ) {
        if let Ok(mut throughput) = self.throughput.write() {
            throughput.observe_completion(worker, output_tokens, decode_secs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sample_single_key() {
        let mut logits = HashMap::new();
        let worker = WorkerWithDpRank::from_worker_id(42);
        logits.insert(worker, 0.5);

        for temperature in &[0.1, 1.0, 10.0] {
            let result = softmax_sample(&logits, *temperature);
            assert_eq!(result.len(), 1, "Should return exactly one worker");
            assert_eq!(result[0], worker, "Should return the only available worker");
        }

        logits.clear();
        logits.insert(worker, -100.0);
        let result = softmax_sample(&logits, 1.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], worker);

        logits.clear();
        logits.insert(worker, 100.0);
        let result = softmax_sample(&logits, 1.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], worker);

        logits.clear();
        logits.insert(worker, 0.0);
        let result = softmax_sample(&logits, 1.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], worker);
    }

    use crate::test_utils::SimpleWorkerConfig;

    /// Helper to build a SchedulingRequest for selector tests.
    fn make_scheduling_request(
        isl_tokens: usize,
        overlaps: rustc_hash::FxHashMap<WorkerWithDpRank, u32>,
    ) -> SchedulingRequest {
        SchedulingRequest {
            maybe_request_id: None,
            token_seq: None,
            isl_tokens,
            overlaps: crate::protocols::OverlapScores {
                scores: overlaps,
                frequencies: Vec::new(),
                tree_sizes: Default::default(),
            },
            decode_blocks: HashMap::new(),
            prefill_tokens: HashMap::new(),
            router_config_override: None,
            update_states: false,
            lora_name: None,
            priority_jump: 0.0,
            allowed_worker_ids: None,
            expected_output_tokens: None,
            priority: None,
            worker_capacities: HashMap::new(),
            resp_tx: None,
        }
    }

    #[test]
    fn test_default_selector_picks_best_worker() {
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w1, 5);
        overlaps.insert(w2, 10); // w2 has more overlap = fewer prefill blocks

        let mut request = make_scheduling_request(256, overlaps);
        // Set decode_blocks to be equal so the tie-breaker is prefill
        request.decode_blocks.insert(w1, 0);
        request.decode_blocks.insert(w2, 0);
        // Set prefill_tokens to reflect overlap (isl - overlap * block_size)
        // so the logit computation correctly reflects fewer prefill blocks for w2.
        request.prefill_tokens.insert(w1, 256 - 5 * 16); // 176
        request.prefill_tokens.insert(w2, 256 - 10 * 16); // 96

        let selector = DefaultWorkerSelector::new(None);
        let result = selector.select_worker(&workers, &request, 16).unwrap();

        // w2 has 10 overlap blocks out of 16 total (256/16), so fewer prefill blocks
        assert_eq!(result.worker, w2);
        assert_eq!(result.overlap_blocks, 10);
        assert!(result.transfer_hint.is_none());
    }

    #[test]
    fn test_default_selector_no_endpoints() {
        let workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        let overlaps = rustc_hash::FxHashMap::default();
        let request = make_scheduling_request(256, overlaps);

        let selector = DefaultWorkerSelector::new(None);
        let result = selector.select_worker(&workers, &request, 16);
        assert!(matches!(result, Err(KvSchedulerError::NoEndpoints)));
    }

    #[test]
    fn test_transfer_selector_no_transfer_when_disabled() {
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w1, 0);
        overlaps.insert(w2, 10);

        let request = make_scheduling_request(256, overlaps);

        // Transfer disabled
        let config = KvRouterConfig {
            enable_kv_transfer: false,
            ..Default::default()
        };
        let selector = TransferAwareWorkerSelector::new(Some(config));
        let result = selector.select_worker(&workers, &request, 16).unwrap();
        assert!(result.transfer_hint.is_none());
    }

    #[test]
    fn test_transfer_selector_triggers_transfer() {
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        // w2 has all the cache, w1 has none
        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w1, 0);
        overlaps.insert(w2, 14); // 14 out of 16 blocks cached

        let mut request = make_scheduling_request(256, overlaps);
        // w2 is heavily loaded, w1 is empty
        request.decode_blocks.insert(w1, 0);
        request.decode_blocks.insert(w2, 100);

        let config = KvRouterConfig {
            enable_kv_transfer: true,
            overlap_score_weight: 1.0,
            transfer_cost_weight: 0.1,
            min_transfer_queue_advantage: 8,
            max_transfer_blocks: 256,
            ..Default::default()
        };
        let selector = TransferAwareWorkerSelector::new(Some(config));
        let result = selector.select_worker(&workers, &request, 16).unwrap();

        // Should pick w1 (less loaded) with a transfer hint from w2
        assert_eq!(result.worker, w1);
        let hint = result.transfer_hint.expect("transfer hint should be present");
        assert_eq!(hint.source_worker, w2);
        assert_eq!(hint.num_blocks, 14); // min(14 - 0, 256)
    }

    #[test]
    fn test_transfer_selector_no_transfer_insufficient_queue_advantage() {
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w1, 0);
        overlaps.insert(w2, 14);

        let mut request = make_scheduling_request(256, overlaps);
        // Queues are similar (difference < min_transfer_queue_advantage=8)
        request.decode_blocks.insert(w1, 5);
        request.decode_blocks.insert(w2, 10);

        let config = KvRouterConfig {
            enable_kv_transfer: true,
            overlap_score_weight: 1.0,
            transfer_cost_weight: 0.1,
            min_transfer_queue_advantage: 8,
            max_transfer_blocks: 256,
            ..Default::default()
        };
        let selector = TransferAwareWorkerSelector::new(Some(config));
        let result = selector.select_worker(&workers, &request, 16).unwrap();

        // Should not trigger transfer because queue advantage (10-5=5) < 8
        assert!(result.transfer_hint.is_none());
    }

    #[test]
    fn test_transfer_selector_respects_max_transfer_blocks() {
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w1, 0);
        overlaps.insert(w2, 14);

        let mut request = make_scheduling_request(256, overlaps);
        request.decode_blocks.insert(w1, 0);
        request.decode_blocks.insert(w2, 100);

        let config = KvRouterConfig {
            enable_kv_transfer: true,
            overlap_score_weight: 1.0,
            transfer_cost_weight: 0.1,
            min_transfer_queue_advantage: 8,
            max_transfer_blocks: 5, // Cap at 5 blocks
            ..Default::default()
        };
        let selector = TransferAwareWorkerSelector::new(Some(config));
        let result = selector.select_worker(&workers, &request, 16).unwrap();

        if let Some(hint) = &result.transfer_hint {
            assert!(hint.num_blocks <= 5, "Should be capped at max_transfer_blocks");
        }
    }

    #[test]
    fn test_transfer_selector_no_transfer_when_no_cache() {
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        // No cache on either worker
        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w1, 0);
        overlaps.insert(w2, 0);

        let mut request = make_scheduling_request(256, overlaps);
        request.decode_blocks.insert(w1, 0);
        request.decode_blocks.insert(w2, 100);

        let config = KvRouterConfig {
            enable_kv_transfer: true,
            ..Default::default()
        };
        let selector = TransferAwareWorkerSelector::new(Some(config));
        let result = selector.select_worker(&workers, &request, 16).unwrap();

        assert!(result.transfer_hint.is_none(), "No transfer when no cache exists");
    }

    #[test]
    fn test_softmax_sample_zero_temperature() {
        let mut logits = HashMap::new();
        let worker1 = WorkerWithDpRank::from_worker_id(1);
        let worker2 = WorkerWithDpRank::from_worker_id(2);
        let worker3 = WorkerWithDpRank::from_worker_id(3);
        let worker4 = WorkerWithDpRank::from_worker_id(4);
        logits.insert(worker1, 5.0);
        logits.insert(worker2, 3.0);
        logits.insert(worker3, 7.0);
        logits.insert(worker4, 3.5);

        let result = softmax_sample(&logits, 0.0);
        assert_eq!(
            result.len(),
            1,
            "Should return one worker when there's no tie"
        );
        assert_eq!(
            result[0], worker2,
            "Should return worker with smallest logit when temperature is 0"
        );

        logits.clear();
        let worker5 = WorkerWithDpRank::from_worker_id(5);
        let worker6 = WorkerWithDpRank::from_worker_id(6);
        logits.insert(worker1, 5.0);
        logits.insert(worker2, 3.0);
        logits.insert(worker5, 3.0);
        logits.insert(worker6, 7.0);

        let result = softmax_sample(&logits, 0.0);
        assert_eq!(
            result.len(),
            2,
            "Should return all workers with smallest logit when tied"
        );
        assert!(
            result.contains(&worker2) && result.contains(&worker5),
            "Should contain both tied workers"
        );

        logits.clear();
        let worker10 = WorkerWithDpRank::from_worker_id(10);
        let worker20 = WorkerWithDpRank::from_worker_id(20);
        let worker30 = WorkerWithDpRank::from_worker_id(30);
        logits.insert(worker10, -1.0);
        logits.insert(worker20, -5.0);
        logits.insert(worker30, 0.0);

        let result = softmax_sample(&logits, 0.0);
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0], worker20,
            "Should handle negative logits correctly"
        );
    }

    #[test]
    fn test_transfer_metrics_recorded_on_transfer() {
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w1, 0);
        overlaps.insert(w2, 14);

        let mut request = make_scheduling_request(256, overlaps);
        request.decode_blocks.insert(w1, 0);
        request.decode_blocks.insert(w2, 100);

        let config = KvRouterConfig {
            enable_kv_transfer: true,
            overlap_score_weight: 1.0,
            transfer_cost_weight: 0.1,
            min_transfer_queue_advantage: 8,
            max_transfer_blocks: 256,
            ..Default::default()
        };
        let selector = TransferAwareWorkerSelector::new(Some(config));
        let result = selector.select_worker(&workers, &request, 16).unwrap();

        assert!(result.transfer_hint.is_some(), "Transfer should be triggered");

        // Verify metrics were recorded
        let transfer_count = selector
            .metrics
            .decisions_total
            .with_label_values(&["transfer"])
            .get();
        assert_eq!(transfer_count, 1, "Should record one transfer decision");

        let no_transfer_count = selector
            .metrics
            .decisions_total
            .with_label_values(&["no_transfer"])
            .get();
        assert_eq!(no_transfer_count, 0, "Should not record no_transfer");

        assert_eq!(
            selector.metrics.blocks_routed_total.get(),
            14,
            "Should record 14 blocks routed"
        );

        assert_eq!(
            selector.metrics.cost_with_transfer.get_sample_count(),
            1,
            "Should record cost_with_transfer"
        );
        assert_eq!(
            selector.metrics.cost_without_transfer.get_sample_count(),
            1,
            "Should record cost_without_transfer"
        );
        assert_eq!(
            selector.metrics.transfer_advantage.get_sample_count(),
            1,
            "Should record transfer_advantage"
        );
    }

    #[test]
    fn test_transfer_metrics_recorded_on_no_transfer() {
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        // No cache = no transfer opportunity
        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w1, 0);
        overlaps.insert(w2, 0);

        let mut request = make_scheduling_request(256, overlaps);
        request.decode_blocks.insert(w1, 0);
        request.decode_blocks.insert(w2, 100);

        let config = KvRouterConfig {
            enable_kv_transfer: true,
            ..Default::default()
        };
        let selector = TransferAwareWorkerSelector::new(Some(config));
        let result = selector.select_worker(&workers, &request, 16).unwrap();

        assert!(result.transfer_hint.is_none());

        let no_transfer_count = selector
            .metrics
            .decisions_total
            .with_label_values(&["no_transfer"])
            .get();
        assert_eq!(no_transfer_count, 1, "Should record no_transfer decision");

        let transfer_count = selector
            .metrics
            .decisions_total
            .with_label_values(&["transfer"])
            .get();
        assert_eq!(transfer_count, 0, "Should not record transfer");

        // cost_without_transfer should be recorded as baseline
        assert_eq!(
            selector.metrics.cost_without_transfer.get_sample_count(),
            1,
            "Should record baseline cost"
        );
    }

    #[test]
    fn test_no_transfer_reason_disabled() {
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w1, 0);
        overlaps.insert(w2, 10);

        let request = make_scheduling_request(256, overlaps);

        // Transfer disabled
        let config = KvRouterConfig {
            enable_kv_transfer: false,
            ..Default::default()
        };
        let selector = TransferAwareWorkerSelector::new(Some(config));
        let _result = selector.select_worker(&workers, &request, 16).unwrap();

        assert_eq!(
            selector.metrics.no_transfer_reasons.with_label_values(&["disabled"]).get(),
            1,
            "Should record 'disabled' reason when transfer is not enabled"
        );
    }

    #[test]
    fn test_no_transfer_reason_no_overlap() {
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        // No cache on either worker
        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w1, 0);
        overlaps.insert(w2, 0);

        let mut request = make_scheduling_request(256, overlaps);
        request.decode_blocks.insert(w1, 0);
        request.decode_blocks.insert(w2, 100);

        let config = KvRouterConfig {
            enable_kv_transfer: true,
            ..Default::default()
        };
        let selector = TransferAwareWorkerSelector::new(Some(config));
        let _result = selector.select_worker(&workers, &request, 16).unwrap();

        assert_eq!(
            selector.metrics.no_transfer_reasons.with_label_values(&["no_overlap"]).get(),
            1,
            "Should record 'no_overlap' reason when no blocks are cached"
        );
    }

    #[test]
    fn test_no_transfer_reason_queue_advantage_low() {
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w1, 0);
        overlaps.insert(w2, 14);

        let mut request = make_scheduling_request(256, overlaps);
        // Queues are similar (difference < min_transfer_queue_advantage=8)
        request.decode_blocks.insert(w1, 5);
        request.decode_blocks.insert(w2, 10);

        let config = KvRouterConfig {
            enable_kv_transfer: true,
            overlap_score_weight: 1.0,
            transfer_cost_weight: 0.1,
            min_transfer_queue_advantage: 8,
            max_transfer_blocks: 256,
            ..Default::default()
        };
        let selector = TransferAwareWorkerSelector::new(Some(config));
        let _result = selector.select_worker(&workers, &request, 16).unwrap();

        assert_eq!(
            selector.metrics.no_transfer_reasons.with_label_values(&["queue_advantage_low"]).get(),
            1,
            "Should record 'queue_advantage_low' when queue delta is below threshold"
        );
    }

    #[test]
    fn test_no_transfer_reason_not_cheaper() {
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        // w2 has some cache, but transfer cost weight is very high
        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w1, 0);
        overlaps.insert(w2, 4); // small overlap

        let mut request = make_scheduling_request(256, overlaps);
        request.decode_blocks.insert(w1, 0);
        request.decode_blocks.insert(w2, 20); // moderate queue difference

        let config = KvRouterConfig {
            enable_kv_transfer: true,
            overlap_score_weight: 1.0,
            transfer_cost_weight: 100.0, // absurdly high — transfer never wins
            min_transfer_queue_advantage: 8,
            max_transfer_blocks: 256,
            ..Default::default()
        };
        let selector = TransferAwareWorkerSelector::new(Some(config));
        let _result = selector.select_worker(&workers, &request, 16).unwrap();

        assert_eq!(
            selector.metrics.no_transfer_reasons.with_label_values(&["not_cheaper"]).get(),
            1,
            "Should record 'not_cheaper' when transfer cost exceeds cache cost"
        );
    }

    #[test]
    fn test_no_transfer_reason_no_target_single_worker() {
        let w1 = WorkerWithDpRank::from_worker_id(1);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());

        // Single worker with some cache
        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w1, 10);

        let request = make_scheduling_request(256, overlaps);

        let config = KvRouterConfig {
            enable_kv_transfer: true,
            ..Default::default()
        };
        let selector = TransferAwareWorkerSelector::new(Some(config));
        let _result = selector.select_worker(&workers, &request, 16).unwrap();

        assert_eq!(
            selector.metrics.no_transfer_reasons.with_label_values(&["no_target"]).get(),
            1,
            "Should record 'no_target' when only one worker is available"
        );
    }

    #[test]
    fn test_no_transfer_reason_target_has_cache() {
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        // Both workers have equal cache — target already has everything source has
        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w1, 10);
        overlaps.insert(w2, 10);

        let mut request = make_scheduling_request(256, overlaps);
        request.decode_blocks.insert(w1, 0);
        request.decode_blocks.insert(w2, 100); // w2 is heavily loaded

        let config = KvRouterConfig {
            enable_kv_transfer: true,
            overlap_score_weight: 1.0,
            transfer_cost_weight: 0.1,
            min_transfer_queue_advantage: 8,
            max_transfer_blocks: 256,
            ..Default::default()
        };
        let selector = TransferAwareWorkerSelector::new(Some(config));
        let _result = selector.select_worker(&workers, &request, 16).unwrap();

        assert_eq!(
            selector.metrics.no_transfer_reasons.with_label_values(&["target_has_cache"]).get(),
            1,
            "Should record 'target_has_cache' when target has >= source's overlap"
        );
    }

    #[test]
    fn test_transfer_metrics_multiple_decisions() {
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        let config = KvRouterConfig {
            enable_kv_transfer: true,
            overlap_score_weight: 1.0,
            transfer_cost_weight: 0.1,
            min_transfer_queue_advantage: 8,
            max_transfer_blocks: 256,
            ..Default::default()
        };
        let selector = TransferAwareWorkerSelector::new(Some(config));

        // First request: triggers transfer (w2 has cache, w1 is empty)
        let mut overlaps1 = rustc_hash::FxHashMap::default();
        overlaps1.insert(w1, 0);
        overlaps1.insert(w2, 14);
        let mut req1 = make_scheduling_request(256, overlaps1);
        req1.decode_blocks.insert(w1, 0);
        req1.decode_blocks.insert(w2, 100);
        let r1 = selector.select_worker(&workers, &req1, 16).unwrap();
        assert!(r1.transfer_hint.is_some());

        // Second request: no transfer (no cache)
        let mut overlaps2 = rustc_hash::FxHashMap::default();
        overlaps2.insert(w1, 0);
        overlaps2.insert(w2, 0);
        let mut req2 = make_scheduling_request(128, overlaps2);
        req2.decode_blocks.insert(w1, 0);
        req2.decode_blocks.insert(w2, 50);
        let r2 = selector.select_worker(&workers, &req2, 16).unwrap();
        assert!(r2.transfer_hint.is_none());

        // Verify cumulative metrics
        let transfer_count = selector
            .metrics
            .decisions_total
            .with_label_values(&["transfer"])
            .get();
        let no_transfer_count = selector
            .metrics
            .decisions_total
            .with_label_values(&["no_transfer"])
            .get();
        assert_eq!(transfer_count, 1);
        assert_eq!(no_transfer_count, 1);
    }

    #[test]
    fn test_transfer_rejected_insufficient_memory() {
        let w0 = WorkerWithDpRank::from_worker_id(0);
        let w1 = WorkerWithDpRank::from_worker_id(1);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(0, SimpleWorkerConfig::default());
        workers.insert(1, SimpleWorkerConfig::default());

        let config = KvRouterConfig {
            enable_kv_transfer: true,
            transfer_cost_weight: 0.1,
            min_transfer_queue_advantage: 0,
            max_transfer_blocks: 256,
            memory_pressure_weight: 1.0,
            ..Default::default()
        };
        let selector = TransferAwareWorkerSelector::new(Some(config));

        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w0, 20); // worker_0 has 20 cached blocks
        overlaps.insert(w1, 0); // worker_1 has no cache

        let mut req = make_scheduling_request(512, overlaps);
        req.expected_output_tokens = Some(256); // ~16 blocks at block_size=16
        req.decode_blocks.insert(w0, 100); // heavily loaded
        req.decode_blocks.insert(w1, 0); // empty

        // Target (worker_1) has very limited memory: 10 free + 5 evictable = 15 total
        // But needs ~20 transfer + ~12 remaining ISL + 16 OSL = ~48 blocks
        req.worker_capacities.insert(
            w1,
            WorkerCapacity {
                free_kv_blocks: 10,
                evictable_kv_blocks: 5,
                total_kv_blocks: 1000,
                num_running_requests: 5,
                last_updated: std::time::Instant::now(),
                active_requests: vec![],
                virtual_reservation_count: 0,
            },
        );

        let result = selector.select_worker(&workers, &req, 16).unwrap();
        assert!(
            result.transfer_hint.is_none(),
            "Expected no transfer due to insufficient memory"
        );
    }

    #[test]
    fn test_transfer_allowed_sufficient_memory() {
        let w0 = WorkerWithDpRank::from_worker_id(0);
        let w1 = WorkerWithDpRank::from_worker_id(1);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(0, SimpleWorkerConfig::default());
        workers.insert(1, SimpleWorkerConfig::default());

        let config = KvRouterConfig {
            enable_kv_transfer: true,
            transfer_cost_weight: 0.1,
            min_transfer_queue_advantage: 0,
            max_transfer_blocks: 256,
            memory_pressure_weight: 1.0,
            ..Default::default()
        };
        let selector = TransferAwareWorkerSelector::new(Some(config));

        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w0, 20);
        overlaps.insert(w1, 0);

        let mut req = make_scheduling_request(512, overlaps);
        req.expected_output_tokens = Some(256);
        req.decode_blocks.insert(w0, 100);
        req.decode_blocks.insert(w1, 0);

        // Target has plenty of memory
        req.worker_capacities.insert(
            w1,
            WorkerCapacity {
                free_kv_blocks: 100,
                evictable_kv_blocks: 50,
                total_kv_blocks: 1000,
                num_running_requests: 2,
                last_updated: std::time::Instant::now(),
                active_requests: vec![],
                virtual_reservation_count: 0,
            },
        );

        let result = selector.select_worker(&workers, &req, 16).unwrap();
        assert!(
            result.transfer_hint.is_some(),
            "Expected transfer to proceed with sufficient memory"
        );
    }

    #[test]
    fn test_transfer_no_capacity_data_backward_compat() {
        let w0 = WorkerWithDpRank::from_worker_id(0);
        let w1 = WorkerWithDpRank::from_worker_id(1);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(0, SimpleWorkerConfig::default());
        workers.insert(1, SimpleWorkerConfig::default());

        let config = KvRouterConfig {
            enable_kv_transfer: true,
            transfer_cost_weight: 0.1,
            min_transfer_queue_advantage: 0,
            max_transfer_blocks: 256,
            memory_pressure_weight: 1.0,
            ..Default::default()
        };
        let selector = TransferAwareWorkerSelector::new(Some(config));

        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w0, 20);
        overlaps.insert(w1, 0);

        let mut req = make_scheduling_request(512, overlaps);
        req.decode_blocks.insert(w0, 100);
        req.decode_blocks.insert(w1, 0);
        // worker_capacities is empty (default) → no memory check

        let result = selector.select_worker(&workers, &req, 16).unwrap();
        assert!(
            result.transfer_hint.is_some(),
            "Expected transfer to proceed when no capacity data is available"
        );
    }

    #[test]
    fn test_scheduling_request_carries_osl_and_priority() {
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w1, 5);

        let mut request = make_scheduling_request(256, overlaps);
        request.expected_output_tokens = Some(500);
        request.priority = Some(3);

        // Verify fields are accessible and correct
        assert_eq!(request.expected_output_tokens, Some(500));
        assert_eq!(request.priority, Some(3));

        // Verify selector still works with new fields present
        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        request.decode_blocks.insert(w1, 0);

        let selector = DefaultWorkerSelector::new(None);
        let result = selector.select_worker(&workers, &request, 16);
        assert!(result.is_ok());
    }

    // ─── Phase 3: Effective load tests ───

    use crate::protocols::ActiveRequestSummary;

    /// Worker A has 5 requests nearly done (generated 490/500 tokens each).
    /// Worker B has 5 requests just started (generated 10/500 tokens each).
    /// Effective load for A should be much lower than B.
    #[test]
    fn test_effective_load_lower_for_nearly_done_workers() {
        let wa = WorkerWithDpRank::from_worker_id(1);
        let wb = WorkerWithDpRank::from_worker_id(2);
        let block_size: u32 = 16;

        let mut capacities = HashMap::new();

        // Worker A: 5 requests nearly done (10 tokens remaining each)
        capacities.insert(
            wa,
            WorkerCapacity {
                free_kv_blocks: 50,
                evictable_kv_blocks: 0,
                total_kv_blocks: 100,
                num_running_requests: 5,
                last_updated: std::time::Instant::now(),
                active_requests: (0..5)
                    .map(|_| ActiveRequestSummary {
                        isl_tokens: 100,
                        generated_tokens: 490,
                        max_new_tokens: 500,
                        priority: 0,
                        is_prefill: false,
                    })
                    .collect(),
                virtual_reservation_count: 0,
            },
        );

        // Worker B: 5 requests just started (490 tokens remaining each)
        capacities.insert(
            wb,
            WorkerCapacity {
                free_kv_blocks: 50,
                evictable_kv_blocks: 0,
                total_kv_blocks: 100,
                num_running_requests: 5,
                last_updated: std::time::Instant::now(),
                active_requests: (0..5)
                    .map(|_| ActiveRequestSummary {
                        isl_tokens: 100,
                        generated_tokens: 10,
                        max_new_tokens: 500,
                        priority: 0,
                        is_prefill: false,
                    })
                    .collect(),
                virtual_reservation_count: 0,
            },
        );

        let load_a = compute_effective_load(&wa, &capacities, Some(0), block_size, 1.3, 0.7);
        let load_b = compute_effective_load(&wb, &capacities, Some(0), block_size, 1.3, 0.7);

        let load_a = load_a.expect("Worker A should have effective load data");
        let load_b = load_b.expect("Worker B should have effective load data");

        // Worker A: 5 * (500-490)/16 = 5 * 0.625 = 3.125
        // Worker B: 5 * (500-10)/16  = 5 * 30.625 = 153.125
        assert!(
            load_a < load_b,
            "Nearly-done worker A (load={load_a:.3}) should have much lower effective load than just-started worker B (load={load_b:.3})"
        );
        assert!(load_a < 5.0, "Worker A effective load should be small: {load_a:.3}");
        assert!(load_b > 100.0, "Worker B effective load should be large: {load_b:.3}");
    }

    /// High-priority in-flight requests should increase effective load more
    /// for a lower-priority incoming request than vice versa.
    #[test]
    fn test_effective_load_priority_contention() {
        let w = WorkerWithDpRank::from_worker_id(1);
        let block_size: u32 = 16;

        let mut capacities = HashMap::new();
        capacities.insert(
            w,
            WorkerCapacity {
                free_kv_blocks: 50,
                evictable_kv_blocks: 0,
                total_kv_blocks: 100,
                num_running_requests: 2,
                last_updated: std::time::Instant::now(),
                active_requests: vec![
                    // One high-priority request (priority=0, 200 tokens remaining)
                    ActiveRequestSummary {
                        isl_tokens: 100,
                        generated_tokens: 100,
                        max_new_tokens: 300,
                        priority: 0, // high priority (lower number = higher priority)
                        is_prefill: false,
                    },
                    // One low-priority request (priority=10, 200 tokens remaining)
                    ActiveRequestSummary {
                        isl_tokens: 100,
                        generated_tokens: 100,
                        max_new_tokens: 300,
                        priority: 10,
                        is_prefill: false,
                    },
                ],
                virtual_reservation_count: 0,
            },
        );

        // Incoming request has mid-priority (5)
        let load_mid = compute_effective_load(&w, &capacities, Some(5), block_size, 1.3, 0.7)
            .expect("should compute");

        // The high-priority req (priority=0 < 5) gets weight 1.3
        // The low-priority req  (priority=10 > 5) gets weight 0.7
        // remaining = (300-100)/16 = 12.5 blocks each
        // load = 12.5 * 1.3 + 12.5 * 0.7 = 16.25 + 8.75 = 25.0
        let expected = 12.5 * 1.3 + 12.5 * 0.7;
        assert!(
            (load_mid - expected).abs() < 0.01,
            "Expected effective load ~{expected:.3}, got {load_mid:.3}"
        );

        // Compare: if incoming request is highest priority (priority=-1),
        // both in-flight requests are lower priority → both get weight 0.7
        let load_high = compute_effective_load(&w, &capacities, Some(-1), block_size, 1.3, 0.7)
            .expect("should compute");
        let expected_high = 12.5 * 0.7 + 12.5 * 0.7; // 17.5
        assert!(
            (load_high - expected_high).abs() < 0.01,
            "Expected effective load ~{expected_high:.3} for high-priority incoming, got {load_high:.3}"
        );

        // Higher-priority incoming request sees LESS contention
        assert!(
            load_high < load_mid,
            "High-priority incoming (load={load_high:.3}) should see less contention than mid-priority (load={load_mid:.3})"
        );
    }

    /// Returns None when worker has no capacity data (backward compat).
    #[test]
    fn test_effective_load_fallback_no_data() {
        let w = WorkerWithDpRank::from_worker_id(1);
        let capacities = HashMap::new(); // No capacity data for any worker

        let result = compute_effective_load(&w, &capacities, Some(0), 16, 1.3, 0.7);
        assert!(
            result.is_none(),
            "Should return None when no capacity data is available"
        );
    }

    /// Returns None when active_requests is empty (signals caller to use fallback).
    #[test]
    fn test_effective_load_empty_requests_falls_back() {
        let w = WorkerWithDpRank::from_worker_id(1);
        let mut capacities = HashMap::new();
        capacities.insert(
            w,
            WorkerCapacity {
                free_kv_blocks: 50,
                evictable_kv_blocks: 0,
                total_kv_blocks: 100,
                num_running_requests: 0,
                last_updated: std::time::Instant::now(),
                active_requests: vec![], // Empty — no per-request data
                virtual_reservation_count: 0,
            },
        );

        let result = compute_effective_load(&w, &capacities, Some(0), 16, 1.3, 0.7);
        assert!(
            result.is_none(),
            "Should return None when active_requests is empty"
        );
    }

    /// Verify that compute_worker_logits uses effective load when config is provided,
    /// and falls back to raw decode_blocks otherwise.
    #[test]
    fn test_compute_worker_logits_with_effective_load() {
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w1, 0);
        overlaps.insert(w2, 0);

        let mut request = make_scheduling_request(256, overlaps);
        // Both have same raw decode blocks
        request.decode_blocks.insert(w1, 50);
        request.decode_blocks.insert(w2, 50);

        // But worker_2 has nearly-done requests, worker_1 has just-started
        request.worker_capacities.insert(
            w1,
            WorkerCapacity {
                free_kv_blocks: 50,
                evictable_kv_blocks: 0,
                total_kv_blocks: 100,
                num_running_requests: 3,
                last_updated: std::time::Instant::now(),
                active_requests: vec![
                    ActiveRequestSummary {
                        isl_tokens: 100,
                        generated_tokens: 10,
                        max_new_tokens: 500,
                        priority: 0,
                        is_prefill: false,
                    },
                ],
                virtual_reservation_count: 0,
            },
        );
        request.worker_capacities.insert(
            w2,
            WorkerCapacity {
                free_kv_blocks: 50,
                evictable_kv_blocks: 0,
                total_kv_blocks: 100,
                num_running_requests: 3,
                last_updated: std::time::Instant::now(),
                active_requests: vec![
                    ActiveRequestSummary {
                        isl_tokens: 100,
                        generated_tokens: 490,
                        max_new_tokens: 500,
                        priority: 0,
                        is_prefill: false,
                    },
                ],
                virtual_reservation_count: 0,
            },
        );

        // Without effective load: both should have the same logit (same decode_blocks)
        let logits_without = compute_worker_logits(&workers, &request, 16, 1.0, None, None, None);
        assert!(
            (logits_without[&w1] - logits_without[&w2]).abs() < 0.001,
            "Without effective load, logits should be equal: w1={:.3}, w2={:.3}",
            logits_without[&w1], logits_without[&w2],
        );

        // With effective load: w2 should have lower logit (nearly-done → less load)
        let elc = EffectiveLoadConfig {
            high_priority_weight: 1.3,
            low_priority_weight: 0.7,
        };
        let logits_with = compute_worker_logits(&workers, &request, 16, 1.0, Some(&elc), None, None);
        assert!(
            logits_with[&w2] < logits_with[&w1],
            "With effective load, nearly-done w2 (logit={:.3}) should have lower logit than just-started w1 (logit={:.3})",
            logits_with[&w2], logits_with[&w1],
        );
    }

    // ─── Phase 4: Headroom projection tests ───

    use crate::scheduling::throughput::ThroughputEstimator;
    use crate::protocols::OverlapScores;

    #[test]
    fn test_projected_free_blocks_requests_finishing_soon() {
        // Worker with 3 active requests:
        // - req1: 490/500 tokens done → 10 remaining → finishes in 0.33s at 30 tok/s
        // - req2: 100/500 tokens done → 400 remaining → finishes in 13.3s
        // - req3: 480/500 tokens done → 20 remaining → finishes in 0.67s
        // Horizon: 1.0s
        let throughput = ThroughputEstimator::new(30.0, 0.1);
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let mut capacities = HashMap::new();
        capacities.insert(
            w1,
            WorkerCapacity {
                free_kv_blocks: 50,
                evictable_kv_blocks: 0,
                total_kv_blocks: 1000,
                num_running_requests: 3,
                last_updated: std::time::Instant::now(),
                active_requests: vec![
                    ActiveRequestSummary {
                        isl_tokens: 100,
                        generated_tokens: 490,
                        max_new_tokens: 500,
                        priority: 0,
                        is_prefill: false,
                    },
                    ActiveRequestSummary {
                        isl_tokens: 200,
                        generated_tokens: 100,
                        max_new_tokens: 500,
                        priority: 0,
                        is_prefill: false,
                    },
                    ActiveRequestSummary {
                        isl_tokens: 150,
                        generated_tokens: 480,
                        max_new_tokens: 500,
                        priority: 0,
                        is_prefill: false,
                    },
                ],
                virtual_reservation_count: 0,
            },
        );

        let result = projected_free_blocks(&w1, &capacities, &throughput, 1.0, 16);
        assert!(result.is_some());
        let freed = result.unwrap();
        // req1 finishes: (100+500)/16 = 37.5
        // req3 finishes: (150+500)/16 = 40.625
        // total = 50 + 37.5 + 40.625 = 128.125
        assert!(
            freed > 120.0 && freed < 140.0,
            "expected ~128.125, got {freed}"
        );
    }

    #[test]
    fn test_projected_free_blocks_no_finishing() {
        // All requests have >100s remaining → nothing finishes in 1s horizon
        let throughput = ThroughputEstimator::new(30.0, 0.1);
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let mut capacities = HashMap::new();
        capacities.insert(
            w1,
            WorkerCapacity {
                free_kv_blocks: 50,
                evictable_kv_blocks: 0,
                total_kv_blocks: 1000,
                num_running_requests: 1,
                last_updated: std::time::Instant::now(),
                active_requests: vec![ActiveRequestSummary {
                    isl_tokens: 100,
                    generated_tokens: 10,
                    max_new_tokens: 5000,
                    priority: 0,
                    is_prefill: false,
                }],
                virtual_reservation_count: 0,
            },
        );

        let result = projected_free_blocks(&w1, &capacities, &throughput, 1.0, 16);
        // Only free_kv_blocks, no requests finish
        assert!((result.unwrap() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_projected_free_blocks_skips_prefill() {
        let throughput = ThroughputEstimator::new(30.0, 0.1);
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let mut capacities = HashMap::new();
        capacities.insert(
            w1,
            WorkerCapacity {
                free_kv_blocks: 50,
                evictable_kv_blocks: 0,
                total_kv_blocks: 1000,
                num_running_requests: 1,
                last_updated: std::time::Instant::now(),
                active_requests: vec![ActiveRequestSummary {
                    isl_tokens: 100,
                    generated_tokens: 0,
                    max_new_tokens: 10,
                    priority: 0,
                    is_prefill: true,
                }],
                virtual_reservation_count: 0,
            },
        );

        let result = projected_free_blocks(&w1, &capacities, &throughput, 100.0, 16);
        // Prefill request skipped, only free blocks
        assert!((result.unwrap() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_projected_free_blocks_no_capacity_data() {
        let throughput = ThroughputEstimator::new(30.0, 0.1);
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let capacities = HashMap::new();
        assert!(projected_free_blocks(&w1, &capacities, &throughput, 1.0, 16).is_none());
    }

    #[test]
    fn test_headroom_bonus_in_logits() {
        // Worker A: many requests about to finish → high headroom → lower logit
        // Worker B: requests just started → low headroom → higher logit
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        let mut request = SchedulingRequest {
            maybe_request_id: None,
            token_seq: None,
            isl_tokens: 256,
            overlaps: OverlapScores {
                scores: {
                    let mut s = rustc_hash::FxHashMap::default();
                    s.insert(w1, 8);
                    s.insert(w2, 8);
                    s
                },
                ..Default::default()
            },
            decode_blocks: [(w1, 50), (w2, 50)].into(),
            prefill_tokens: [(w1, 128), (w2, 128)].into(),
            router_config_override: None,
            update_states: false,
            lora_name: None,
            priority_jump: 0.0,
            allowed_worker_ids: None,
            expected_output_tokens: None,
            priority: None,
            worker_capacities: HashMap::new(),
            resp_tx: None,
        };

        // W1: 5 requests nearly done (499/500) — only 1 token remaining each
        // At 30 TPS, each finishes in 0.033s < prefill horizon of 0.051s
        request.worker_capacities.insert(
            w1,
            WorkerCapacity {
                free_kv_blocks: 100,
                evictable_kv_blocks: 0,
                total_kv_blocks: 1000,
                num_running_requests: 5,
                last_updated: std::time::Instant::now(),
                active_requests: (0..5)
                    .map(|_| ActiveRequestSummary {
                        isl_tokens: 100,
                        generated_tokens: 499,
                        max_new_tokens: 500,
                        priority: 0,
                        is_prefill: false,
                    })
                    .collect(),
                virtual_reservation_count: 0,
            },
        );
        // W2: 5 requests just started (10/500) — won't finish within horizon
        request.worker_capacities.insert(
            w2,
            WorkerCapacity {
                free_kv_blocks: 100,
                evictable_kv_blocks: 0,
                total_kv_blocks: 1000,
                num_running_requests: 5,
                last_updated: std::time::Instant::now(),
                active_requests: (0..5)
                    .map(|_| ActiveRequestSummary {
                        isl_tokens: 100,
                        generated_tokens: 10,
                        max_new_tokens: 500,
                        priority: 0,
                        is_prefill: false,
                    })
                    .collect(),
                virtual_reservation_count: 0,
            },
        );

        let throughput = ThroughputEstimator::new(30.0, 0.1);
        let hc = HeadroomConfig {
            headroom_weight: 5.0,
            throughput,
        };

        let logits = compute_worker_logits(&workers, &request, block_size, 1.0, None, Some(&hc), None);

        let logit_w1 = logits[&w1];
        let logit_w2 = logits[&w2];
        assert!(
            logit_w1 < logit_w2,
            "w1 (nearly done requests) should have lower logit: w1={logit_w1}, w2={logit_w2}"
        );
    }

    // ─── Phase 5: Priority-aware routing tests ───

    #[test]
    fn test_priority_adjusted_weights_high_priority() {
        // priority=0, threshold=5 → urgency=1.0
        // overlap: base * (1.0 - 0.5*1.0) = base * 0.5
        // headroom: base * (1.0 + 2.0*1.0) = base * 3.0
        let (ow, hw) = priority_adjusted_weights(1.0, 1.0, Some(0), 5);
        assert!((ow - 0.5).abs() < 0.01, "ow={ow}");
        assert!((hw - 3.0).abs() < 0.01, "hw={hw}");
    }

    #[test]
    fn test_priority_adjusted_weights_mid_priority() {
        // priority=3, threshold=5 → urgency = 1 - 3/5 = 0.4
        let (ow, hw) = priority_adjusted_weights(1.0, 1.0, Some(3), 5);
        assert!((ow - 0.8).abs() < 0.01, "ow={ow}");
        assert!((hw - 1.8).abs() < 0.01, "hw={hw}");
    }

    #[test]
    fn test_priority_adjusted_weights_at_threshold() {
        // priority=5, threshold=5 → urgency = 0.0
        let (ow, hw) = priority_adjusted_weights(1.0, 1.0, Some(5), 5);
        assert!((ow - 1.0).abs() < 0.01, "ow={ow}");
        assert!((hw - 1.0).abs() < 0.01, "hw={hw}");
    }

    #[test]
    fn test_priority_adjusted_weights_low_priority() {
        // priority=50 > threshold=5 → default weights
        let (ow, hw) = priority_adjusted_weights(1.0, 2.0, Some(50), 5);
        assert!((ow - 1.0).abs() < 0.01);
        assert!((hw - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_priority_adjusted_weights_none_priority() {
        let (ow, hw) = priority_adjusted_weights(1.0, 2.0, None, 5);
        assert!((ow - 1.0).abs() < 0.01);
        assert!((hw - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_priority_adjusted_weights_negative_priority() {
        // Negative priority → clamped urgency = 1.0
        let (ow, hw) = priority_adjusted_weights(1.0, 1.0, Some(-5), 5);
        assert!((ow - 0.5).abs() < 0.01, "ow={ow}");
        assert!((hw - 3.0).abs() < 0.01, "hw={hw}");
    }

    #[test]
    fn test_high_priority_prefers_empty_worker_over_cache() {
        // W1: high cache overlap but loaded (many in-flight, low headroom)
        // W2: no cache overlap but empty (high headroom)
        // With priority_adaptive_weights and priority=1:
        //   overlap_weight halved, headroom_weight tripled
        //   → W2 should win
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        let mut request = SchedulingRequest {
            maybe_request_id: None,
            token_seq: None,
            isl_tokens: 256,
            overlaps: OverlapScores {
                scores: {
                    let mut s = rustc_hash::FxHashMap::default();
                    s.insert(w1, 12);
                    s.insert(w2, 0);
                    s
                },
                ..Default::default()
            },
            decode_blocks: [(w1, 80), (w2, 10)].into(),
            prefill_tokens: [(w1, 64), (w2, 256)].into(),
            router_config_override: None,
            update_states: false,
            lora_name: None,
            priority_jump: 0.0,
            allowed_worker_ids: None,
            expected_output_tokens: Some(500),
            priority: Some(1),
            worker_capacities: HashMap::new(),
            resp_tx: None,
        };

        // W1: loaded, no headroom
        request.worker_capacities.insert(
            w1,
            WorkerCapacity {
                free_kv_blocks: 50,
                evictable_kv_blocks: 0,
                total_kv_blocks: 1000,
                num_running_requests: 10,
                last_updated: std::time::Instant::now(),
                active_requests: (0..10)
                    .map(|_| ActiveRequestSummary {
                        isl_tokens: 100,
                        generated_tokens: 10,
                        max_new_tokens: 2000,
                        priority: 0,
                        is_prefill: false,
                    })
                    .collect(),
                virtual_reservation_count: 0,
            },
        );
        // W2: empty, lots of headroom
        request.worker_capacities.insert(
            w2,
            WorkerCapacity {
                free_kv_blocks: 800,
                evictable_kv_blocks: 0,
                total_kv_blocks: 1000,
                num_running_requests: 0,
                last_updated: std::time::Instant::now(),
                active_requests: vec![],
                virtual_reservation_count: 0,
            },
        );

        // urgency = 1 - 1/5 = 0.8
        let (overlap_w, headroom_w) = priority_adjusted_weights(1.0, 5.0, Some(1), 5);
        // overlap_w = 1.0 * (1 - 0.5*0.8) = 0.6
        assert!((overlap_w - 0.6).abs() < 0.01);
        // headroom_w = 5.0 * (1 + 2.0*0.8) = 13.0
        assert!((headroom_w - 13.0).abs() < 0.01);

        let throughput = ThroughputEstimator::new(30.0, 0.1);
        let hc = HeadroomConfig {
            headroom_weight: headroom_w,
            throughput,
        };

        let logits =
            compute_worker_logits(&workers, &request, block_size, overlap_w, None, Some(&hc), None);

        // W2 (empty, high headroom) should have lower logit than W1 (cached but loaded)
        assert!(
            logits[&w2] < logits[&w1],
            "high-priority should prefer empty worker: w1={}, w2={}",
            logits[&w1],
            logits[&w2]
        );
    }

    #[test]
    fn test_low_priority_prefers_cache_over_headroom() {
        // Same setup but priority=50 (above threshold=5) → default weights
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        let mut request = SchedulingRequest {
            maybe_request_id: None,
            token_seq: None,
            isl_tokens: 256,
            overlaps: OverlapScores {
                scores: {
                    let mut s = rustc_hash::FxHashMap::default();
                    s.insert(w1, 12);
                    s.insert(w2, 0);
                    s
                },
                ..Default::default()
            },
            decode_blocks: [(w1, 20), (w2, 10)].into(),
            prefill_tokens: [(w1, 64), (w2, 256)].into(),
            router_config_override: None,
            update_states: false,
            lora_name: None,
            priority_jump: 0.0,
            allowed_worker_ids: None,
            expected_output_tokens: Some(500),
            priority: Some(50),
            worker_capacities: HashMap::new(),
            resp_tx: None,
        };

        // W1: moderate load
        request.worker_capacities.insert(
            w1,
            WorkerCapacity {
                free_kv_blocks: 200,
                evictable_kv_blocks: 0,
                total_kv_blocks: 1000,
                num_running_requests: 3,
                last_updated: std::time::Instant::now(),
                active_requests: (0..3)
                    .map(|_| ActiveRequestSummary {
                        isl_tokens: 100,
                        generated_tokens: 10,
                        max_new_tokens: 2000,
                        priority: 0,
                        is_prefill: false,
                    })
                    .collect(),
                virtual_reservation_count: 0,
            },
        );
        // W2: empty
        request.worker_capacities.insert(
            w2,
            WorkerCapacity {
                free_kv_blocks: 800,
                evictable_kv_blocks: 0,
                total_kv_blocks: 1000,
                num_running_requests: 0,
                last_updated: std::time::Instant::now(),
                active_requests: vec![],
                virtual_reservation_count: 0,
            },
        );

        // No priority adjustment (above threshold)
        let (overlap_w, headroom_w) = priority_adjusted_weights(1.0, 5.0, Some(50), 5);
        assert!((overlap_w - 1.0).abs() < 0.01);
        assert!((headroom_w - 5.0).abs() < 0.01);

        let throughput = ThroughputEstimator::new(30.0, 0.1);
        let hc = HeadroomConfig {
            headroom_weight: headroom_w,
            throughput,
        };

        let logits =
            compute_worker_logits(&workers, &request, block_size, overlap_w, None, Some(&hc), None);

        // W1 (cached, overlap=12) should have lower logit than W2
        // W1 prefill: (64/16)=4.0, W2 prefill: (256/16)=16.0
        // Cache advantage: 1.0 * 4 vs 1.0 * 16 = 12 block difference
        assert!(
            logits[&w1] < logits[&w2],
            "low-priority should prefer cache-hot worker: w1={}, w2={}",
            logits[&w1],
            logits[&w2]
        );
    }

    #[test]
    fn test_oracle_metrics_emitted_during_routing() {
        let selector = TransferAwareWorkerSelector::new(Some(KvRouterConfig {
            use_effective_load: true,
            headroom_weight: 1.0,
            ..KvRouterConfig::default()
        }));

        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);

        let mut workers: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        workers.insert(1, SimpleWorkerConfig::default());
        workers.insert(2, SimpleWorkerConfig::default());

        let mut overlaps = rustc_hash::FxHashMap::default();
        overlaps.insert(w1, 5);
        overlaps.insert(w2, 10);

        let mut request = make_scheduling_request(256, overlaps);
        request.decode_blocks.insert(w1, 10);
        request.decode_blocks.insert(w2, 5);
        request.prefill_tokens.insert(w1, 176);
        request.prefill_tokens.insert(w2, 96);

        let _result = selector.select_worker(&workers, &request, 16).unwrap();

        // Verify oracle metrics were recorded (logit histogram should have observations)
        assert!(
            selector.oracle_metrics.routing_decision_logit.get_sample_count() > 0,
            "routing_decision_logit should have observations after select_worker"
        );
    }
}
