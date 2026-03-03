// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for transfer routing decisions.
//!
//! Follows the `KvIndexerMetrics` pattern (indexer.rs): feature-gated
//! `Component`-based initialization with unregistered fallback for tests.

#[cfg(feature = "metrics")]
use dynamo_runtime::{
    component::Component,
    metrics::{MetricsHierarchy, prometheus_names::kvrouter},
};
use prometheus::{Histogram, HistogramOpts, IntCounter, IntCounterVec, Opts};
#[cfg(feature = "metrics")]
use std::sync::{Arc, OnceLock};

/// Prometheus metrics for transfer routing decisions.
#[derive(Debug)]
pub struct TransferDecisionMetrics {
    /// dynamo_kvrouter_transfer_decisions_total{result="transfer|no_transfer"}
    pub decisions_total: IntCounterVec,
    /// dynamo_kvrouter_transfer_blocks_routed_total
    pub blocks_routed_total: IntCounter,
    /// dynamo_kvrouter_transfer_estimated_ms (unused until we add latency estimation)
    pub estimated_ms: Histogram,
    /// dynamo_kvrouter_cost_with_transfer
    pub cost_with_transfer: Histogram,
    /// dynamo_kvrouter_cost_without_transfer
    pub cost_without_transfer: Histogram,
    /// dynamo_kvrouter_transfer_advantage (cost_normal - cost_transfer)
    pub transfer_advantage: Histogram,
    /// dynamo_kvrouter_no_transfer_reasons_total{reason="..."}
    pub no_transfer_reasons: IntCounterVec,
}

#[cfg(feature = "metrics")]
static TRANSFER_DECISION_METRICS: OnceLock<Arc<TransferDecisionMetrics>> = OnceLock::new();

impl TransferDecisionMetrics {
    #[cfg(feature = "metrics")]
    pub fn from_component(component: &Component) -> Arc<Self> {
        TRANSFER_DECISION_METRICS
            .get_or_init(|| {
                match Self::try_from_component(component) {
                    Ok(m) => Arc::new(m),
                    Err(e) => {
                        tracing::warn!(
                            "Failed to create transfer decision metrics from component: {}. \
                             Using unregistered metrics as fallback.",
                            e
                        );
                        Arc::new(Self::new_unregistered())
                    }
                }
            })
            .clone()
    }

    #[cfg(feature = "metrics")]
    fn try_from_component(
        component: &Component,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let metrics = component.metrics();
        let decisions_total = metrics.create_intcountervec(
            kvrouter::TRANSFER_DECISIONS_TOTAL,
            "Transfer routing decisions by result",
            &["result"],
            &[],
        )?;
        let blocks_routed_total = metrics.create_intcounter(
            kvrouter::TRANSFER_BLOCKS_ROUTED_TOTAL,
            "Total KV blocks routed for transfer",
            &[],
        )?;
        let estimated_ms = metrics.create_histogram(
            kvrouter::TRANSFER_ESTIMATED_MS,
            "Estimated transfer time in ms",
            &[],
            Some(vec![0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]),
        )?;
        let cost_with_transfer = metrics.create_histogram(
            kvrouter::COST_WITH_TRANSFER,
            "Cost score when transfer is chosen",
            &[],
            Some(vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]),
        )?;
        let cost_without_transfer = metrics.create_histogram(
            kvrouter::COST_WITHOUT_TRANSFER,
            "Cost score of cache-optimal worker (baseline)",
            &[],
            Some(vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]),
        )?;
        let transfer_advantage = metrics.create_histogram(
            kvrouter::TRANSFER_ADVANTAGE,
            "Cost advantage of transfer (normal_cost - transfer_cost)",
            &[],
            Some(vec![-10.0, -5.0, 0.0, 5.0, 10.0, 25.0, 50.0, 100.0]),
        )?;
        let no_transfer_reasons = metrics.create_intcountervec(
            kvrouter::NO_TRANSFER_REASONS_TOTAL,
            "Reasons why transfer was not chosen",
            &["reason"],
            &[],
        )?;
        Ok(Self {
            decisions_total,
            blocks_routed_total,
            estimated_ms,
            cost_with_transfer,
            cost_without_transfer,
            transfer_advantage,
            no_transfer_reasons,
        })
    }

    /// Creates metrics not registered with any MetricsRegistry.
    /// Used for tests and as fallback.
    pub fn new_unregistered() -> Self {
        Self {
            decisions_total: IntCounterVec::new(
                Opts::new(
                    "dynamo_kvrouter_transfer_decisions_total",
                    "Transfer routing decisions by result",
                ),
                &["result"],
            )
            .unwrap(),
            blocks_routed_total: IntCounter::new(
                "dynamo_kvrouter_transfer_blocks_routed_total",
                "Total KV blocks routed for transfer",
            )
            .unwrap(),
            estimated_ms: Histogram::with_opts(
                HistogramOpts::new(
                    "dynamo_kvrouter_transfer_estimated_ms",
                    "Estimated transfer time in ms",
                )
                .buckets(vec![0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]),
            )
            .unwrap(),
            cost_with_transfer: Histogram::with_opts(
                HistogramOpts::new(
                    "dynamo_kvrouter_cost_with_transfer",
                    "Cost score when transfer is chosen",
                )
                .buckets(vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]),
            )
            .unwrap(),
            cost_without_transfer: Histogram::with_opts(
                HistogramOpts::new(
                    "dynamo_kvrouter_cost_without_transfer",
                    "Cost score of cache-optimal worker (baseline)",
                )
                .buckets(vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]),
            )
            .unwrap(),
            transfer_advantage: Histogram::with_opts(
                HistogramOpts::new(
                    "dynamo_kvrouter_transfer_advantage",
                    "Cost advantage of transfer (normal_cost - transfer_cost)",
                )
                .buckets(vec![-10.0, -5.0, 0.0, 5.0, 10.0, 25.0, 50.0, 100.0]),
            )
            .unwrap(),
            no_transfer_reasons: IntCounterVec::new(
                Opts::new(
                    "dynamo_kvrouter_no_transfer_reasons_total",
                    "Reasons why transfer was not chosen",
                ),
                &["reason"],
            )
            .unwrap(),
        }
    }

    /// Record a transfer decision.
    pub fn record_decision(&self, result: &str) {
        self.decisions_total.with_label_values(&[result]).inc();
    }

    /// Record transfer cost metrics when transfer IS chosen.
    pub fn record_transfer_chosen(&self, blocks: u32, cost_transfer: f64, cost_normal: f64) {
        self.blocks_routed_total.inc_by(blocks as u64);
        self.cost_with_transfer.observe(cost_transfer);
        self.cost_without_transfer.observe(cost_normal);
        self.transfer_advantage.observe(cost_normal - cost_transfer);
    }

    /// Record cost when transfer is NOT chosen (for visibility).
    pub fn record_no_transfer_cost(&self, cost_normal: f64) {
        self.cost_without_transfer.observe(cost_normal);
    }

    /// Record the reason why a transfer was not chosen.
    pub fn record_no_transfer_reason(&self, reason: &str) {
        self.no_transfer_reasons.with_label_values(&[reason]).inc();
    }
}
