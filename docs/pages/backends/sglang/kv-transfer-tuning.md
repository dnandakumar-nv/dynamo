<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# KV Cache Transfer Tuning Guide

## Overview

Cross-worker KV cache transfer allows the KV-aware router to make smarter
scheduling decisions by decoupling cache locality from load balancing. Without
KV transfer, the router must choose between two options: route a request to the
worker that holds the best KV cache hit (risking overloading that worker), or
route to a less-loaded worker and pay the full prefill cost from scratch.

With KV transfer enabled, a third option becomes available. The router can
route a request to a less-loaded worker **and** transfer the cached KV blocks
from the source worker via RDMA (using NIXL). This means the target worker
receives the relevant KV cache data without having to recompute it, while the
source worker avoids taking on additional decode load.

KV transfer is most beneficial when:

- Multiple workers serve the same model with overlapping prompt patterns.
- Load is unevenly distributed across workers.
- The hardware topology supports fast GPU-to-GPU transfers (PCIe peer-to-peer
  or NVLink within a node, RDMA across nodes).
- Prompt prefixes are long enough that transferring cached blocks is cheaper
  than recomputing them.

KV transfer adds minimal overhead when idle. The system uses dedicated ZMQ
channels (one per TP rank) for transfer initiation, bypassing the higher-latency
`collective_rpc` path used for metadata operations.

## Configuration Parameters

The following parameters control KV transfer behavior. All are set on the
SGLang worker process via CLI flags or the Dynamo component configuration.

| Parameter | CLI Flag | Default | Tuning Guidance |
|-----------|----------|---------|-----------------|
| `enable_kv_transfer` | `--enable-kv-transfer` | `false` | Enable to activate cross-worker KV cache transfer. Requires NIXL to be installed and GPU-to-GPU connectivity between workers. |
| `transfer_cost_weight` (beta) | `--transfer-cost-weight` | `0.1` | Ratio of actual transfer time to prefill time per block. Measure on your hardware: run a transfer benchmark and divide the per-block transfer latency by the per-block prefill latency. Lower values mean the router favors transfers more aggressively. |
| `min_transfer_queue_advantage` | `--min-transfer-queue-advantage` | `8` | Minimum difference in decode queue depth (in blocks) between the source and target workers before the router will consider a transfer. Higher values make the system more conservative, triggering transfers only when the load imbalance is significant. |
| `max_transfer_blocks` | `--max-transfer-blocks` | `256` | Maximum number of blocks that can be transferred in a single operation. This cap is driven by NIXL descriptor list sizes and latency budget. If your transfers are consistently hitting this cap, consider whether the workload would benefit from a higher limit or whether the prompt sharing pattern should be reevaluated. |
| `transfer_timeout_ms` | `--transfer-timeout-ms` | `5000` | Timeout in milliseconds for a single transfer operation. Set this to 2-3x the expected worst-case RDMA transfer time for `max_transfer_blocks`. If transfers time out frequently, check RDMA health and network congestion. |
| `kv_transfer_min_blocks` | `--kv-transfer-min-blocks` | `4` | Minimum number of blocks required to initiate a transfer. Transfers of 1-3 blocks have high overhead relative to the prefill savings. Increase this if you observe many small transfers that do not improve latency. |

### Tuning `transfer_cost_weight` (beta)

The `transfer_cost_weight` parameter is the most important tuning knob. It
represents the ratio:

```
beta = actual_transfer_time_per_block / prefill_time_per_block
```

To measure this on your hardware:

1. Run a KV transfer benchmark with a known block count and record the
   end-to-end transfer latency.
2. Run a prefill benchmark for the same number of tokens (blocks x block size)
   and record the prefill latency.
3. Divide the per-block transfer time by the per-block prefill time.

For example, if transferring 64 blocks takes 50 ms and prefilling the equivalent
tokens takes 500 ms, then `beta = (50/64) / (500/64) = 0.1`.

A beta value of `0.1` means transfers are 10x cheaper than prefill. Values
closer to `1.0` mean transfers provide diminishing benefit. Values above `1.0`
mean transfers are slower than prefill and should generally be avoided.

## PromQL Query Examples

Use these queries to monitor KV transfer performance in Prometheus or Grafana.

```promql
# Transfer decision rate by outcome
sum(rate(dynamo_component_transfer_decisions_total[5m])) by (result)

# Transfer success rate (percentage)
100 * sum(rate(dynamo_kv_transfer_requests_total{role="target",result="success"}[5m]))
    / sum(rate(dynamo_kv_transfer_requests_total{role="target"}[5m]))

# P99 transfer latency (seconds)
histogram_quantile(0.99,
  sum by (le) (rate(dynamo_kv_transfer_duration_seconds_bucket{role="target"}[5m])))

# NIXL throughput (MB/s)
sum(rate(dynamo_kv_transfer_nixl_read_bytes_total[5m])) / 1048576

# Average cost advantage (positive means transfer was cheaper)
sum(rate(dynamo_component_transfer_advantage_sum[5m]))
  / sum(rate(dynamo_component_transfer_advantage_count[5m]))

# Fallback breakdown by reason
sum by (reason) (rate(dynamo_kv_transfer_fallback_total[5m]))

# Transfer error breakdown (excludes deliberate skips like below_threshold)
sum by (reason) (rate(dynamo_kv_transfer_fallback_total{reason!="below_threshold"}[5m]))

# Router no-transfer reason breakdown
sum(rate(dynamo_component_no_transfer_reasons_total[5m])) by (reason)

# Bytes transferred per second (target side)
sum(rate(dynamo_kv_transfer_bytes_total{role="target"}[5m]))

# NIXL read operation P99 latency
histogram_quantile(0.99,
  sum by (le) (rate(dynamo_kv_transfer_nixl_read_duration_seconds_bucket[5m])))
```

### Key Metrics to Watch

- **Transfer success rate** should stay above 90% in steady state. Rates below
  80% indicate systematic issues (timeout, NIXL errors, block eviction races).
- **P99 transfer latency** depends on hardware but should be well below the
  prefill time for the equivalent token count. If P99 approaches prefill time,
  transfers are not providing benefit.
- **Cost advantage** should be consistently positive. A negative average means
  the system is spending more time on transfers than it would on prefill.
- **Fallback rate** tracks how often the target worker falls back to local
  prefill after a transfer was initiated. Common reasons include source block
  eviction, timeout, and NIXL errors.

### Interpreting Transfer Error Reasons

The `dynamo_kv_transfer_fallback_total` metric now classifies errors into
specific reason labels instead of a generic `"error"` bucket:

| Reason | Meaning | Remediation |
|--------|---------|-------------|
| `allocation_failed` | Target worker could not allocate local KV pages for the incoming transfer. | Reduce `max_transfer_blocks`, increase GPU memory, or lower `max_num_seqs` to free KV cache capacity. |
| `timeout` | The NIXL transfer or an RPC timed out before completion. | Increase `transfer_timeout_ms`, check RDMA health, or investigate GPU memory pressure. |
| `nixl_error` | NIXL returned an error during transfer setup or execution. | Check NIXL agent logs, verify RDMA connectivity, ensure NIXL is correctly initialized. |
| `too_many_pending` | Too many concurrent transfers are already in flight. | Wait for backlog to clear, or reduce the transfer rate by increasing `min_transfer_queue_advantage`. |
| `not_initialized` | The KV transfer subsystem has not finished initialization. | This is transient during startup. If persistent, check scheduler initialization logs. |
| `error` | Unclassified error (catch-all). | Check worker logs for the full exception message. |

### Interpreting Router No-Transfer Reasons

The `dynamo_component_no_transfer_reasons_total` metric shows why the router
decided not to transfer, broken down by reason:

| Reason | Meaning | Remediation |
|--------|---------|-------------|
| `disabled` | KV transfer is not enabled in the router config. | Set `enable_kv_transfer: true` if transfers are desired. |
| `no_overlap` | No worker has any cached blocks for this request. | Expected for cold-start or unique prompts. No action needed. |
| `no_target` | Only one worker is available (no alternative target). | Add more workers to the deployment. |
| `queue_advantage_low` | The load difference between source and target is below the threshold. | Lower `min_transfer_queue_advantage` if transfers are desired at lower load deltas. |
| `target_has_cache` | The target worker already has all the cached blocks the source has. | Expected when cache is well-distributed. No action needed. |
| `not_cheaper` | Transfer cost exceeds the cost of routing to the cache-optimal worker. | Re-measure `transfer_cost_weight` or accept that transfers are not beneficial for this workload. |

## Recommended Alert Thresholds

| Alert | Condition | Severity | Description |
|-------|-----------|----------|-------------|
| Transfer Success Rate Low | < 80% over 5m | Warning | More than 20% of transfers are failing. Check NIXL connectivity and transfer timeout settings. |
| Transfer P99 Latency High | > 2s over 5m | Warning | Transfers are taking too long. Investigate RDMA path health, PCIe bandwidth contention, or GPU memory pressure. |
| NIXL Read Duration P99 High | > 1s over 5m | Critical | The RDMA layer itself is slow. This typically indicates hardware-level issues: link degradation, incorrect NUMA placement, or driver problems. |
| Transfer Advantage Negative | Avg < 0 over 10m | Info | Transfers are not helping on average. Consider increasing `min_transfer_queue_advantage` to reduce unnecessary transfers, or re-measure `transfer_cost_weight` for your hardware. |
| High Fallback Rate | > 50% of decisions over 5m | Warning | Most initiated transfers are falling back to local prefill. Common causes: source blocks evicted before transfer completes, or transfer timeout too aggressive. |
| Allocation Failure Rate High | `allocation_failed` > 20% of fallbacks over 5m | Warning | Target workers cannot allocate KV pages. Indicates GPU memory pressure -- consider reducing `max_num_seqs` or `max_transfer_blocks`. |

### Example Alertmanager Rules

```yaml
groups:
  - name: kv_transfer
    rules:
      - alert: KVTransferSuccessRateLow
        expr: |
          100 * sum(rate(dynamo_kv_transfer_requests_total{role="target",result="success"}[5m]))
              / sum(rate(dynamo_kv_transfer_requests_total{role="target"}[5m])) < 80
        for: 5m
        labels:
          severity: warning

      - alert: KVTransferP99LatencyHigh
        expr: |
          histogram_quantile(0.99,
            sum by (le) (rate(dynamo_kv_transfer_duration_seconds_bucket{role="target"}[5m]))) > 2
        for: 5m
        labels:
          severity: warning

      - alert: NIXLReadP99High
        expr: |
          histogram_quantile(0.99,
            sum by (le) (rate(dynamo_kv_transfer_nixl_read_duration_seconds_bucket[5m]))) > 1
        for: 5m
        labels:
          severity: critical

      - alert: KVTransferAllocationFailureRateHigh
        expr: |
          sum(rate(dynamo_kv_transfer_fallback_total{reason="allocation_failed"}[5m]))
              / sum(rate(dynamo_kv_transfer_fallback_total[5m])) > 0.2
        for: 5m
        labels:
          severity: warning
```

## Hardware-Specific Notes

KV transfer performance is heavily influenced by the GPU interconnect topology.
Use `nvidia-smi topo -m` to understand your system layout.

### PCIe Topology Labels

| Label | Meaning | Typical Latency Impact |
|-------|---------|----------------------|
| **PIX** | Same PCIe switch | Lowest latency, highest bandwidth |
| **PHB** | Same NUMA node, traverses host bridge | Low latency, good bandwidth |
| **SYS** | Cross-NUMA, traverses CPU interconnect | Higher latency, reduced bandwidth |
| **NODE** | Same node but different CPU sockets | Similar to SYS |

### Single-Node Recommendations

- **Within NUMA node (PHB)**: Transfers are fast. The default
  `transfer_cost_weight` of 0.1 is typically appropriate. Workers sharing a NUMA
  node make the best transfer pairs.
- **Cross-NUMA (SYS)**: Transfers incur additional latency from crossing the CPU
  interconnect. You may need to increase `transfer_cost_weight` to 0.15-0.25
  depending on the interconnect bandwidth (e.g., AMD Infinity Fabric vs Intel
  UPI). Measure with the benchmark script to get accurate numbers.
- **GPU placement**: When possible, configure your deployment so that workers
  likely to exchange KV blocks (e.g., serving the same model with overlapping
  prompt patterns) are placed on GPUs within the same NUMA node.

### Multi-Node Considerations

Cross-node transfers require NIXL network transport configuration (typically
InfiniBand or RoCE). Additional considerations:

- Network latency dominates over PCIe topology within each node.
- Set `transfer_timeout_ms` higher (10000-15000 ms) to account for network
  variability.
- `transfer_cost_weight` will be significantly higher (0.3-0.8 depending on
  network bandwidth relative to GPU compute throughput).
- Ensure RDMA interfaces are correctly configured and reachable between all
  workers. NIXL will fail silently if the RDMA path is not established.
- Monitor `dynamo_kv_transfer_nixl_read_duration_seconds` closely -- network
  issues will show up as increased NIXL read latency before they cause
  transfer failures.

### Example: 8x L40S with PHB/SYS Topology

On a system with 8 L40S GPUs split across two NUMA nodes (GPUs 0-3 on NUMA 0,
GPUs 4-7 on NUMA 1):

- Intra-NUMA transfers (e.g., GPU 0 to GPU 2): ~0.5-1 ms per 64 blocks.
- Cross-NUMA transfers (e.g., GPU 1 to GPU 5): ~1.5-3 ms per 64 blocks.
- Recommended: set `transfer_cost_weight` to 0.1 for intra-NUMA pairs and
  consider 0.15-0.2 if cross-NUMA transfers are common.

## Dashboard Access

A pre-built Grafana dashboard for KV cache transfer monitoring is
auto-provisioned when the observability stack is deployed. Access it at:

- **Dashboard name**: KV Cache Transfer
- **Dashboard UID**: `kv-transfer`
- **Path**: Grafana > Dashboards > KV Cache Transfer

The dashboard includes panels for:

- Transfer decision rate and outcome breakdown
- Transfer success rate over time
- P99 and P50 transfer latency
- NIXL read throughput and latency
- Cost advantage distribution
- Fallback rate by reason
- Per-worker transfer activity heatmap

For instructions on deploying the observability stack (Prometheus + Grafana),
see the [SGLang Observability](sglang-observability.md) guide.
