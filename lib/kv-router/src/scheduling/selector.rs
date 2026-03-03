// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use rand::Rng;

use super::config::KvRouterConfig;
use super::transfer_metrics::TransferDecisionMetrics;
use super::types::{KvSchedulerError, SchedulingRequest};
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

/// Compute per-worker logits (cost scores) for routing.
///
/// Returns a map of (worker, logit) where lower logit = better choice.
/// The cost function is: logit = overlap_weight * prefill_blocks + decode_blocks
fn compute_worker_logits<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    block_size: u32,
    overlap_weight: f64,
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

            let logit = overlap_weight * potential_prefill_block + decode_block;

            worker_logits.insert(worker, logit);

            tracing::info!(
                "Formula for worker_id={} dp_rank={:?} with {overlap} cached blocks: {logit:.3} \
                 = {overlap_weight:.1} * prefill_blocks + decode_blocks \
                 = {overlap_weight:.1} * {potential_prefill_block:.3} + {decode_block:.3}",
                worker.worker_id,
                worker.dp_rank
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

        let worker_logits = compute_worker_logits(workers, request, block_size, overlap_weight);

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
#[derive(Debug, Clone)]
pub struct TransferAwareWorkerSelector {
    pub kv_router_config: KvRouterConfig,
    pub metrics: Arc<TransferDecisionMetrics>,
}

impl Default for TransferAwareWorkerSelector {
    fn default() -> Self {
        Self {
            kv_router_config: KvRouterConfig::default(),
            metrics: Arc::new(TransferDecisionMetrics::new_unregistered()),
        }
    }
}

impl TransferAwareWorkerSelector {
    pub fn new(kv_router_config: Option<KvRouterConfig>) -> Self {
        Self {
            kv_router_config: kv_router_config.unwrap_or_default(),
            metrics: Arc::new(TransferDecisionMetrics::new_unregistered()),
        }
    }

    #[cfg(feature = "metrics")]
    pub fn with_component(
        kv_router_config: Option<KvRouterConfig>,
        component: &dynamo_runtime::component::Component,
    ) -> Self {
        Self {
            kv_router_config: kv_router_config.unwrap_or_default(),
            metrics: TransferDecisionMetrics::from_component(component),
        }
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

        // Compute transfer cost:
        // After transfer, the target will have (target_overlap + transferable) cached blocks.
        // Remaining prefill = total request tokens that still need prefill after transfer.
        let isl = request.isl_tokens;
        let tokens_covered_by_transfer = (target_overlap + transferable) as usize * block_size as usize;
        let remaining_prefill_tokens = isl.saturating_sub(tokens_covered_by_transfer);
        let remaining_prefill_blocks = (remaining_prefill_tokens as f64) / (block_size as f64);

        let transfer_cost = overlap_weight * remaining_prefill_blocks
            + target_decode as f64
            + cfg.transfer_cost_weight * transferable as f64;

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

        let overlap_weight = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.overlap_score_weight)
            .unwrap_or(self.kv_router_config.overlap_score_weight);

        let worker_logits = compute_worker_logits(workers, request, block_size, overlap_weight);

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
}
