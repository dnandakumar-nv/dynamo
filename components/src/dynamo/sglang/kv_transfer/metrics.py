# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV Transfer metrics (Prometheus).

Metrics are created lazily to avoid import-time Prometheus initialization
issues with SGLang's multiprocess setup (PROMETHEUS_MULTIPROC_DIR).

All record_* functions are safe to call even if Prometheus is unavailable;
they silently no-op.
"""

import logging

logger = logging.getLogger(__name__)

_metrics_initialized = False
_transfer_requests = None
_transfer_blocks = None
_transfer_duration = None
_transfer_fallback = None
_transfer_events_published = None
_transfer_bytes = None
_nixl_read_ops = None
_nixl_read_bytes = None
_nixl_read_duration = None


def _ensure_initialized():
    """Initialize Prometheus metrics on first use."""
    global _metrics_initialized, _transfer_requests, _transfer_blocks
    global _transfer_duration, _transfer_fallback, _transfer_events_published
    global _transfer_bytes, _nixl_read_ops, _nixl_read_bytes, _nixl_read_duration
    if _metrics_initialized:
        return
    try:
        from prometheus_client import Counter, Histogram

        _transfer_requests = Counter(
            "dynamo_kv_transfer_requests_total",
            "KV transfer requests",
            ["role", "result"],
        )
        _transfer_blocks = Counter(
            "dynamo_kv_transfer_blocks_total",
            "KV blocks transferred",
            ["role"],
        )
        _transfer_duration = Histogram(
            "dynamo_kv_transfer_duration_seconds",
            "KV transfer duration",
            ["role"],
            buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 5.0],
        )
        _transfer_fallback = Counter(
            "dynamo_kv_transfer_fallback_total",
            "KV transfer fallback reasons",
            ["reason"],
        )
        _transfer_events_published = Counter(
            "dynamo_kv_transfer_events_published_total",
            "KV events published after transfer",
        )
        _transfer_bytes = Counter(
            "dynamo_kv_transfer_bytes_total",
            "Total bytes transferred",
            ["role"],
        )
        _nixl_read_ops = Counter(
            "dynamo_kv_transfer_nixl_read_ops_total",
            "Total NIXL read operations",
        )
        _nixl_read_bytes = Counter(
            "dynamo_kv_transfer_nixl_read_bytes_total",
            "Total bytes read via NIXL RDMA",
        )
        _nixl_read_duration = Histogram(
            "dynamo_kv_transfer_nixl_read_duration_seconds",
            "NIXL RDMA read operation duration",
            buckets=[
                0.0005, 0.001, 0.002, 0.005, 0.01,
                0.02, 0.05, 0.1, 0.5, 1.0, 5.0,
            ],
        )
        _metrics_initialized = True
    except Exception as e:
        logger.debug(f"Prometheus metrics not available: {e}")


def record_transfer_request(role: str, result: str):
    """Record a transfer request (role=target|source, result=success|failed|timeout)."""
    _ensure_initialized()
    if _transfer_requests:
        _transfer_requests.labels(role=role, result=result).inc()


def record_transfer_blocks(role: str, count: int):
    """Record number of blocks transferred."""
    _ensure_initialized()
    if _transfer_blocks:
        _transfer_blocks.labels(role=role).inc(count)


def record_transfer_duration(role: str, seconds: float):
    """Record transfer duration in seconds."""
    _ensure_initialized()
    if _transfer_duration:
        _transfer_duration.labels(role=role).observe(seconds)


def record_transfer_fallback(reason: str):
    """Record a transfer fallback with reason."""
    _ensure_initialized()
    if _transfer_fallback:
        _transfer_fallback.labels(reason=reason).inc()


def record_events_published():
    """Record that KV events were published after transfer."""
    _ensure_initialized()
    if _transfer_events_published:
        _transfer_events_published.inc()


def record_transfer_bytes(role: str, byte_count: int):
    """Record bytes transferred (role=source|target)."""
    _ensure_initialized()
    if _transfer_bytes:
        _transfer_bytes.labels(role=role).inc(byte_count)


def record_nixl_read_op(byte_count: int, duration_seconds: float):
    """Record a NIXL RDMA read operation with bytes and duration."""
    _ensure_initialized()
    if _nixl_read_ops:
        _nixl_read_ops.inc()
    if _nixl_read_bytes:
        _nixl_read_bytes.inc(byte_count)
    if _nixl_read_duration:
        _nixl_read_duration.observe(duration_seconds)
