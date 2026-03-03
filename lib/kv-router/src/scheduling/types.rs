// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use dynamo_tokens::SequenceHash;
use serde::{Deserialize, Serialize};

use super::config::RouterConfigOverride;
use crate::protocols::{
    ActiveRequestSummary, DpRank, OverlapScores, TransferHint, WorkerId, WorkerWithDpRank,
};

/// Real-time capacity data for a worker, updated from ActiveLoad metrics.
#[derive(Debug, Clone)]
pub struct WorkerCapacity {
    /// Free KV cache blocks (from allocator).
    pub free_kv_blocks: u64,
    /// Evictable KV cache blocks (from tree cache).
    pub evictable_kv_blocks: u64,
    /// Total KV cache block capacity.
    pub total_kv_blocks: u64,
    /// Number of currently running requests.
    pub num_running_requests: u32,
    /// When this capacity data was last updated.
    pub last_updated: Instant,
    /// Per-request summaries for effective load computation.
    pub active_requests: Vec<ActiveRequestSummary>,
    /// Number of virtual reservations applied since the last real ActiveLoad update.
    /// Reset to 0 when update_capacity() replaces the entry with real data.
    pub virtual_reservation_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PotentialLoad {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub potential_prefill_tokens: usize,
    pub potential_decode_blocks: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum KvSchedulerError {
    #[error("no endpoints available to route work")]
    NoEndpoints,

    #[error("endpoint subscriber shutdown")]
    SubscriberShutdown,

    #[error("failed to initialize event publisher: {0}")]
    InitFailed(String),
}

#[derive(Debug)]
pub struct SchedulingResponse {
    pub best_worker: WorkerWithDpRank,
    pub overlap_blocks: u32,
    /// Transfer hint from the selector, if a cross-worker transfer
    /// was deemed beneficial.
    pub transfer_hint: Option<TransferHint>,
}

pub struct SchedulingRequest {
    pub maybe_request_id: Option<String>,
    pub token_seq: Option<Vec<SequenceHash>>,
    pub isl_tokens: usize,
    pub overlaps: OverlapScores,
    pub decode_blocks: HashMap<WorkerWithDpRank, usize>,
    pub prefill_tokens: HashMap<WorkerWithDpRank, usize>,
    pub router_config_override: Option<RouterConfigOverride>,
    pub update_states: bool,
    pub lora_name: Option<String>,
    /// Priority jump in seconds; decreases effective arrival time in the queue.
    pub priority_jump: f64,
    /// Optional set of allowed worker IDs to restrict routing decisions (EPP).
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
    /// Predicted output sequence length from agent_hints.osl.
    /// Used for memory-aware routing and bin-packing.
    pub expected_output_tokens: Option<u32>,
    /// Request priority from agent_hints.priority (lower = higher priority).
    /// Used for priority-aware worker selection.
    pub priority: Option<i32>,
    /// Per-worker capacity data, populated from ActiveLoad metrics.
    /// Used by the selector for memory-aware routing decisions.
    pub worker_capacities: HashMap<WorkerWithDpRank, WorkerCapacity>,
    pub resp_tx: Option<tokio::sync::oneshot::Sender<Result<SchedulingResponse, KvSchedulerError>>>,
}

impl SchedulingRequest {
    pub fn respond(&mut self, result: Result<SchedulingResponse, KvSchedulerError>) {
        let Some(tx) = self.resp_tx.take() else {
            tracing::error!("respond called multiple times on same request");
            return;
        };
        if tx.send(result).is_err() {
            tracing::error!("failed to send response to requestor");
        }
    }
}
