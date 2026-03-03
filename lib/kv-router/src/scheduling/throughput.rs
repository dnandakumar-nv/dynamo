// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EMA-based per-worker decode throughput estimator.
//!
//! Tracks observed tokens/sec/request for each worker via an exponential moving
//! average and provides remaining-time estimates for in-flight requests.  Used by
//! the headroom projection in [`super::selector`] to predict when workers will
//! have free capacity.

use std::collections::HashMap;

use crate::protocols::WorkerWithDpRank;

/// EMA-based per-worker decode throughput estimator.
///
/// Tracks observed decode tokens/sec per request for each worker and provides
/// remaining-time estimates for in-flight requests.
#[derive(Debug, Clone)]
pub struct ThroughputEstimator {
    /// EMA of observed decode tokens/sec per request on each worker.
    worker_tps: HashMap<WorkerWithDpRank, f64>,
    /// EMA smoothing factor (0.01..1.0). Higher = more responsive to recent observations.
    ema_alpha: f64,
    /// Default estimate when no observations exist for a worker.
    default_tps: f64,
}

impl ThroughputEstimator {
    pub fn new(default_tps: f64, ema_alpha: f64) -> Self {
        Self {
            worker_tps: HashMap::new(),
            ema_alpha: ema_alpha.clamp(0.01, 1.0),
            default_tps: default_tps.max(1.0),
        }
    }

    /// Get the current throughput estimate for a worker (tokens/sec/request).
    pub fn get_tps(&self, worker: &WorkerWithDpRank) -> f64 {
        self.worker_tps
            .get(worker)
            .copied()
            .unwrap_or(self.default_tps)
    }

    /// Estimate how many seconds until a request with `remaining_tokens` finishes.
    pub fn estimate_remaining_seconds(
        &self,
        worker: &WorkerWithDpRank,
        remaining_tokens: u32,
    ) -> f64 {
        let tps = self.get_tps(worker);
        remaining_tokens as f64 / tps
    }

    /// Called when a request completes — update EMA for that worker.
    ///
    /// `output_tokens`: number of tokens generated during the decode phase.
    /// `decode_secs`: wall-clock time spent in decode phase (prefill excluded).
    pub fn observe_completion(
        &mut self,
        worker: &WorkerWithDpRank,
        output_tokens: u32,
        decode_secs: f64,
    ) {
        if decode_secs <= 0.0 || output_tokens == 0 {
            return;
        }
        let observed_tps = output_tokens as f64 / decode_secs;
        let current = self
            .worker_tps
            .entry(*worker)
            .or_insert(self.default_tps);
        *current = self.ema_alpha * observed_tps + (1.0 - self.ema_alpha) * *current;
    }

    /// Remove throughput data for a worker that is no longer available.
    pub fn remove_worker(&mut self, worker: &WorkerWithDpRank) {
        self.worker_tps.remove(worker);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_tps() {
        let est = ThroughputEstimator::new(30.0, 0.1);
        let w = WorkerWithDpRank::from_worker_id(1);
        assert_eq!(est.get_tps(&w), 30.0);
        assert!((est.estimate_remaining_seconds(&w, 300) - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_observe_updates_ema() {
        let mut est = ThroughputEstimator::new(30.0, 0.1);
        let w = WorkerWithDpRank::from_worker_id(1);
        // Observe 60 tok/s
        est.observe_completion(&w, 600, 10.0);
        // EMA: 0.1 * 60 + 0.9 * 30 = 33.0
        assert!((est.get_tps(&w) - 33.0).abs() < 0.01);
    }

    #[test]
    fn test_ema_converges() {
        let mut est = ThroughputEstimator::new(30.0, 0.1);
        let w = WorkerWithDpRank::from_worker_id(1);
        // 20 observations of 50 tok/s
        for _ in 0..20 {
            est.observe_completion(&w, 500, 10.0);
        }
        // Should converge close to 50 (EMA with alpha=0.1 from 30 after 20 steps ≈ 47.6)
        assert!(
            (est.get_tps(&w) - 50.0).abs() < 3.0,
            "expected ~50.0, got {}",
            est.get_tps(&w)
        );
    }

    #[test]
    fn test_zero_duration_ignored() {
        let mut est = ThroughputEstimator::new(30.0, 0.1);
        let w = WorkerWithDpRank::from_worker_id(1);
        est.observe_completion(&w, 100, 0.0);
        assert_eq!(est.get_tps(&w), 30.0); // unchanged
    }

    #[test]
    fn test_zero_tokens_ignored() {
        let mut est = ThroughputEstimator::new(30.0, 0.1);
        let w = WorkerWithDpRank::from_worker_id(1);
        est.observe_completion(&w, 0, 5.0);
        assert_eq!(est.get_tps(&w), 30.0); // unchanged
    }

    #[test]
    fn test_remove_worker() {
        let mut est = ThroughputEstimator::new(30.0, 0.1);
        let w = WorkerWithDpRank::from_worker_id(1);
        est.observe_completion(&w, 600, 10.0);
        assert_ne!(est.get_tps(&w), 30.0);
        est.remove_worker(&w);
        assert_eq!(est.get_tps(&w), 30.0); // back to default
    }

    #[test]
    fn test_multiple_workers_independent() {
        let mut est = ThroughputEstimator::new(30.0, 0.1);
        let w1 = WorkerWithDpRank::from_worker_id(1);
        let w2 = WorkerWithDpRank::from_worker_id(2);
        est.observe_completion(&w1, 600, 10.0); // 60 tok/s
        est.observe_completion(&w2, 300, 10.0); // 30 tok/s
        // w1: 0.1*60 + 0.9*30 = 33
        assert!((est.get_tps(&w1) - 33.0).abs() < 0.01);
        // w2: 0.1*30 + 0.9*30 = 30
        assert!((est.get_tps(&w2) - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_negative_duration_ignored() {
        let mut est = ThroughputEstimator::new(30.0, 0.1);
        let w = WorkerWithDpRank::from_worker_id(1);
        est.observe_completion(&w, 100, -1.0);
        assert_eq!(est.get_tps(&w), 30.0);
    }
}
