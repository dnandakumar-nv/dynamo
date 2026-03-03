// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for oracle routing decisions.
//!
//! Feature-gated Component-based initialization with unregistered fallback.

#[cfg(feature = "metrics")]
use dynamo_runtime::{
    component::Component,
    metrics::{MetricsHierarchy, prometheus_names::kvrouter},
};
use prometheus::{Histogram, HistogramOpts, IntCounter};
#[cfg(feature = "metrics")]
use std::sync::{Arc, OnceLock};

/// Prometheus metrics for oracle routing factors.
#[derive(Debug)]
pub struct OracleRoutingMetrics {
    /// Effective load (OSL-weighted) per worker per routing decision.
    pub effective_load: Histogram,
    /// Memory pressure per worker per transfer evaluation.
    pub memory_pressure: Histogram,
    /// Projected headroom ratio per worker per routing decision.
    pub headroom_ratio: Histogram,
    /// Total KV blocks reserved virtually (counter).
    pub virtual_reservation_blocks_total: IntCounter,
    /// Number of virtual reservation events (counter).
    pub virtual_reservation_count: IntCounter,
    /// Final logit value per worker per routing decision.
    pub routing_decision_logit: Histogram,
}

#[cfg(feature = "metrics")]
static ORACLE_ROUTING_METRICS: OnceLock<Arc<OracleRoutingMetrics>> = OnceLock::new();

impl OracleRoutingMetrics {
    #[cfg(feature = "metrics")]
    pub fn from_component(component: &Component) -> Arc<Self> {
        ORACLE_ROUTING_METRICS
            .get_or_init(|| {
                match Self::try_from_component(component) {
                    Ok(m) => Arc::new(m),
                    Err(e) => {
                        tracing::warn!(
                            "Failed to create oracle routing metrics from component: {}. \
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
        let effective_load = metrics.create_histogram(
            kvrouter::EFFECTIVE_LOAD,
            "Effective load (OSL-weighted) per worker per routing decision",
            &[],
            Some(vec![0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]),
        )?;
        let memory_pressure = metrics.create_histogram(
            kvrouter::MEMORY_PRESSURE,
            "Memory pressure per worker per transfer evaluation",
            &[],
            Some(vec![0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]),
        )?;
        let headroom_ratio = metrics.create_histogram(
            kvrouter::HEADROOM_RATIO,
            "Projected headroom ratio per worker per routing decision",
            &[],
            Some(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        )?;
        let virtual_reservation_blocks_total = metrics.create_intcounter(
            kvrouter::VIRTUAL_RESERVATION_BLOCKS_TOTAL,
            "Total KV blocks reserved virtually",
            &[],
        )?;
        let virtual_reservation_count = metrics.create_intcounter(
            kvrouter::VIRTUAL_RESERVATION_COUNT,
            "Number of virtual reservation events",
            &[],
        )?;
        let routing_decision_logit = metrics.create_histogram(
            kvrouter::ROUTING_DECISION_LOGIT,
            "Final logit value per worker per routing decision",
            &[],
            Some(vec![0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]),
        )?;
        Ok(Self {
            effective_load,
            memory_pressure,
            headroom_ratio,
            virtual_reservation_blocks_total,
            virtual_reservation_count,
            routing_decision_logit,
        })
    }

    /// Creates metrics not registered with any MetricsRegistry.
    /// Used for tests and as fallback.
    pub fn new_unregistered() -> Self {
        Self {
            effective_load: Histogram::with_opts(
                HistogramOpts::new(
                    "dynamo_kvrouter_effective_load",
                    "Effective load (OSL-weighted) per worker per routing decision",
                )
                .buckets(vec![0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]),
            )
            .unwrap(),
            memory_pressure: Histogram::with_opts(
                HistogramOpts::new(
                    "dynamo_kvrouter_memory_pressure",
                    "Memory pressure per worker per transfer evaluation",
                )
                .buckets(vec![0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]),
            )
            .unwrap(),
            headroom_ratio: Histogram::with_opts(
                HistogramOpts::new(
                    "dynamo_kvrouter_headroom_ratio",
                    "Projected headroom ratio per worker per routing decision",
                )
                .buckets(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            )
            .unwrap(),
            virtual_reservation_blocks_total: IntCounter::new(
                "dynamo_kvrouter_virtual_reservation_blocks_total",
                "Total KV blocks reserved virtually",
            )
            .unwrap(),
            virtual_reservation_count: IntCounter::new(
                "dynamo_kvrouter_virtual_reservation_count",
                "Number of virtual reservation events",
            )
            .unwrap(),
            routing_decision_logit: Histogram::with_opts(
                HistogramOpts::new(
                    "dynamo_kvrouter_routing_decision_logit",
                    "Final logit value per worker per routing decision",
                )
                .buckets(vec![0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]),
            )
            .unwrap(),
        }
    }

    pub fn record_effective_load(&self, load: f64) {
        self.effective_load.observe(load);
    }

    pub fn record_memory_pressure(&self, pressure: f64) {
        self.memory_pressure.observe(pressure);
    }

    pub fn record_headroom_ratio(&self, ratio: f64) {
        self.headroom_ratio.observe(ratio);
    }

    pub fn record_virtual_reservation(&self, blocks: u64) {
        self.virtual_reservation_blocks_total.inc_by(blocks);
        self.virtual_reservation_count.inc();
    }

    pub fn record_logit(&self, logit: f64) {
        self.routing_decision_logit.observe(logit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oracle_metrics_unregistered_no_panic() {
        let m = OracleRoutingMetrics::new_unregistered();
        m.record_effective_load(42.0);
        m.record_memory_pressure(0.8);
        m.record_headroom_ratio(0.5);
        m.record_virtual_reservation(32);
        m.record_logit(15.3);
        // No panic = pass. Metrics are recorded to unregistered collectors.
    }

    #[test]
    fn test_oracle_metrics_virtual_reservation_counter() {
        let m = OracleRoutingMetrics::new_unregistered();
        m.record_virtual_reservation(10);
        m.record_virtual_reservation(20);
        assert_eq!(m.virtual_reservation_blocks_total.get(), 30);
        assert_eq!(m.virtual_reservation_count.get(), 2);
    }
}
