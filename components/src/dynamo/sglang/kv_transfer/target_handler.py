# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target-side handler for KV transfer orchestration.

Coordinates the full transfer flow when a request arrives with a TransferHint:
1. Fetch source worker's NIXL metadata via Dynamo RPC (cached)
2. Query source for which KV blocks are available via Dynamo RPC
3. Execute the transfer via local scheduler RPC (receive_kv_blocks)
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional


def _classify_transfer_error(error_msg: str) -> str:
    """Classify a transfer exception message into a reason label for metrics.

    Maps known error patterns to specific reason labels so Prometheus/Grafana
    can show a breakdown instead of a generic "error" bucket.
    """
    msg_lower = error_msg.lower()
    if "could not allocate" in msg_lower:
        return "allocation_failed"
    if "timeout" in msg_lower:
        return "timeout"
    if "nixl" in msg_lower:
        return "nixl_error"
    if "too many pending" in msg_lower:
        return "too_many_pending"
    if "not initialized" in msg_lower:
        return "not_initialized"
    return "error"


class TransferResult:
    """Result of a KV transfer attempt."""

    __slots__ = ("success", "num_blocks", "transferred_tokens", "error")

    def __init__(
        self,
        success: bool,
        num_blocks: int = 0,
        transferred_tokens: int = 0,
        error: str = "",
    ):
        self.success = success
        self.num_blocks = num_blocks
        self.transferred_tokens = transferred_tokens
        self.error = error

    @classmethod
    def succeeded(cls, num_blocks: int, transferred_tokens: int):
        return cls(True, num_blocks, transferred_tokens)

    @classmethod
    def failed(cls, error: str):
        return cls(False, error=error)


class KvTransferTargetHandler:
    """Orchestrates KV block transfer from a remote source worker.

    Runs in the main process. Coordinates:
    1. Fetching source worker's metadata via Dynamo RPC
    2. Querying source for block locations via Dynamo RPC
    3. Executing the transfer via scheduler RPC (receive_kv_blocks)

    All GPU operations happen inside the scheduler process.
    """

    def __init__(self, manager: "KvTransferManager"):
        self.manager = manager
        self._transfer_client = None
        # Block query cache: (worker_id, prefix_hash, num_blocks) -> (result, expiry)
        self._block_query_cache = {}
        self._block_query_cache_ttl = 2.0  # seconds

    async def set_transfer_endpoint(self, runtime, namespace: str, component: str):
        """Set up the Dynamo endpoint client for calling remote workers."""
        endpoint_path = f"{namespace}.{component}.kv_transfer"
        endpoint = runtime.endpoint(endpoint_path)
        self._transfer_client = await endpoint.client()
        logging.info(f"KV Transfer target: endpoint client ready ({endpoint_path})")

    async def execute_transfer(
        self,
        transfer_hint: dict,
        token_ids: list,
    ) -> TransferResult:
        """Execute the full KV block transfer from source to local worker.

        Args:
            transfer_hint: {
                "source_worker": {"worker_id": X, "dp_rank": Y},
                "num_blocks": N
            }
            token_ids: Full token sequence for this request

        Returns:
            TransferResult with success/failure status
        """
        source_worker = transfer_hint.get("source_worker", {})
        source_worker_id = source_worker.get("worker_id")
        num_blocks = transfer_hint.get("num_blocks", 0)

        if source_worker_id is None or num_blocks == 0:
            from dynamo.sglang.kv_transfer.metrics import (
                record_transfer_fallback,
                record_transfer_request,
            )

            record_transfer_request("target", "failed")
            record_transfer_fallback("invalid_hint")
            return TransferResult.failed("Invalid transfer hint")

        # Skip transfers below the minimum block threshold. Small transfers
        # (1-3 blocks) cost ~200ms+ in overhead but save <5ms of prefill,
        # making them a net loss even considering cache population benefit.
        min_blocks = getattr(
            self.manager.config.dynamo_args, "kv_transfer_min_blocks", 4
        )
        if num_blocks < min_blocks:
            from dynamo.sglang.kv_transfer.metrics import (
                record_transfer_fallback,
                record_transfer_request,
            )

            record_transfer_request("target", "skipped")
            record_transfer_fallback("below_threshold")
            return TransferResult.failed(
                f"Below minimum block threshold ({num_blocks} < {min_blocks})"
            )

        try:
            transfer_start = time.monotonic()

            logging.info(
                "KV transfer initiated",
                extra={
                    "event": "kv_transfer_initiated",
                    "source_worker": source_worker_id,
                    "num_hint_blocks": num_blocks,
                },
            )

            # Trim token_ids to only the prefix needed for transfer.
            # Avoids serializing thousands of extra tokens through RPCs.
            page_size = self.manager.page_size
            prefix_len = min(num_blocks * page_size, len(token_ids))
            prefix_token_ids = list(token_ids[:prefix_len])

            # Steps 1 & 2: Fetch metadata and query blocks in parallel.
            # These are independent — metadata for RDMA setup, block query
            # for source KV page locations. Block query uses a short-TTL
            # cache to skip the Dynamo RPC for repeated prefixes.
            source_metadata, block_info = await asyncio.gather(
                self._get_source_metadata(source_worker_id),
                self._query_source_blocks_cached(
                    source_worker_id, prefix_token_ids, num_blocks
                ),
            )
            t_lookup = time.monotonic()

            if source_metadata is None:
                from dynamo.sglang.kv_transfer.metrics import (
                    record_transfer_fallback,
                    record_transfer_request,
                )

                record_transfer_request("target", "failed")
                record_transfer_fallback("metadata_fetch")
                return TransferResult.failed("Could not fetch source metadata")

            if block_info is None or block_info.get("status") != "ok":
                from dynamo.sglang.kv_transfer.metrics import (
                    record_transfer_fallback,
                    record_transfer_request,
                )

                msg = (
                    block_info.get("message", "unknown")
                    if block_info
                    else "RPC failed"
                )
                record_transfer_request("target", "failed")
                record_transfer_fallback("block_query")
                return TransferResult.failed(f"Block query failed: {msg}")

            source_kv_indices = block_info.get("kv_indices", [])
            actual_blocks = block_info.get("num_matched_blocks", 0)

            if actual_blocks == 0:
                from dynamo.sglang.kv_transfer.metrics import (
                    record_transfer_fallback,
                    record_transfer_request,
                )

                record_transfer_request("target", "failed")
                record_transfer_fallback("no_blocks")
                return TransferResult.failed("No blocks available on source")

            # Retrim prefix if fewer blocks matched than requested
            actual_prefix_len = actual_blocks * page_size
            if actual_prefix_len < len(prefix_token_ids):
                prefix_token_ids = prefix_token_ids[:actual_prefix_len]

            # Step 3: Execute transfer via local scheduler RPC (non-blocking)
            await self.manager.execute_receive_transfer_async(
                source_worker_id=source_worker_id,
                source_metadata_per_rank=source_metadata,
                source_kv_indices=source_kv_indices,
                token_ids=prefix_token_ids,
                num_blocks=actual_blocks,
            )
            t_transfer = time.monotonic()

            transferred_tokens = actual_blocks * page_size
            transfer_elapsed = t_transfer - transfer_start

            from dynamo.sglang.kv_transfer.metrics import (
                record_transfer_blocks,
                record_transfer_bytes,
                record_transfer_duration,
                record_transfer_request,
            )

            record_transfer_request("target", "success")
            record_transfer_blocks("target", actual_blocks)
            record_transfer_duration("target", transfer_elapsed)
            bpb = getattr(self.manager, "bytes_per_block", 0) or 0
            if isinstance(bpb, int) and bpb > 0:
                record_transfer_bytes("target", actual_blocks * bpb)

            elapsed_ms = transfer_elapsed * 1000
            lookup_ms = (t_lookup - transfer_start) * 1000
            rdma_ms = (t_transfer - t_lookup) * 1000
            logging.info(
                f"KV transfer completed: {actual_blocks} blocks "
                f"({transferred_tokens} tokens) in {elapsed_ms:.1f}ms "
                f"[lookup={lookup_ms:.0f}ms rdma={rdma_ms:.0f}ms]",
                extra={
                    "event": "kv_transfer_completed",
                    "source_worker": source_worker_id,
                    "num_blocks": actual_blocks,
                    "transferred_tokens": transferred_tokens,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "lookup_ms": round(lookup_ms, 2),
                    "rdma_ms": round(rdma_ms, 2),
                    "result": "success",
                },
            )
            return TransferResult.succeeded(actual_blocks, transferred_tokens)

        except Exception as e:
            from dynamo.sglang.kv_transfer.metrics import (
                record_transfer_fallback,
                record_transfer_request,
            )

            record_transfer_request("target", "failed")
            record_transfer_fallback(_classify_transfer_error(str(e)))
            logging.warning(
                f"KV transfer failed, falling back to full prefill: {e}",
                extra={
                    "event": "kv_transfer_failed",
                    "source_worker": source_worker_id,
                    "error": str(e),
                    "result": "fallback_to_prefill",
                },
            )
            return TransferResult.failed(str(e))

    async def _get_source_metadata(
        self, source_worker_id: int
    ) -> Optional[List[dict]]:
        """Fetch source worker's per-rank NIXL metadata, with caching.

        Returns a list of metadata dicts, one per TP rank of the source
        worker. Each dict contains the NIXL agent metadata and KV buffer
        pointers for that rank's GPU.
        """
        cached = self.manager.get_cached_metadata(source_worker_id)
        if cached is not None:
            return cached

        try:
            response = await self._rpc_to_source(
                source_worker_id,
                {"action": "get_metadata"},
            )
            if response and response.get("status") == "ok":
                per_rank = response["per_rank"]
                self.manager.cache_metadata(source_worker_id, per_rank)
                return per_rank
        except Exception as e:
            logging.warning(
                f"Failed to fetch metadata from worker {source_worker_id}: {e}"
            )
        return None

    async def _query_source_blocks(
        self, source_worker_id: int, token_ids: list, num_blocks: int
    ) -> Optional[dict]:
        """Query source worker for KV page indices via Dynamo RPC."""
        try:
            return await self._rpc_to_source(
                source_worker_id,
                {
                    "action": "query_blocks",
                    "token_ids": token_ids,
                    "num_blocks": num_blocks,
                },
            )
        except Exception as e:
            logging.warning(
                f"Block query to worker {source_worker_id} failed: {e}"
            )
            return None

    async def _query_source_blocks_cached(
        self, source_worker_id: int, prefix_token_ids: list, num_blocks: int
    ) -> Optional[dict]:
        """Query source worker for KV page indices, with short-TTL caching.

        For burst workloads where many requests share the same prefix,
        this avoids redundant Dynamo RPCs to the source worker. The cache
        key includes the prefix tokens and source worker ID, with a 2-second
        TTL to bound the staleness of physical page indices.
        """
        cache_key = (source_worker_id, hash(tuple(prefix_token_ids)), num_blocks)
        now = time.monotonic()

        cached = self._block_query_cache.get(cache_key)
        if cached is not None:
            result, expiry = cached
            if now < expiry:
                return result
            del self._block_query_cache[cache_key]

        result = await self._query_source_blocks(
            source_worker_id, prefix_token_ids, num_blocks
        )
        if result is not None and result.get("status") == "ok":
            self._block_query_cache[cache_key] = (
                result, now + self._block_query_cache_ttl
            )
        return result

    async def _rpc_to_source(
        self, worker_id: int, request: dict
    ) -> Optional[dict]:
        """Make a Dynamo RPC call to a specific worker's kv_transfer endpoint.

        Uses client.direct() to route to a specific worker instance.
        """
        if self._transfer_client is None:
            logging.error("Transfer endpoint client not initialized")
            return None

        try:
            response_stream = await self._transfer_client.direct(
                request, instance_id=worker_id
            )
            async for response in response_stream:
                # Response may be wrapped (has .data() method) or raw dict
                if hasattr(response, "data"):
                    return response.data()
                return response
            return None
        except Exception as e:
            logging.warning(f"RPC to worker {worker_id} failed: {e}")
            return None
