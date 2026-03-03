# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-side handler for KV transfer Dynamo endpoint.

Handles incoming requests from remote target workers that want to:
1. Fetch this worker's NIXL metadata (for RDMA connection setup)
2. Query which KV blocks are cached for a given token sequence
"""

import logging
import time
from typing import Any, AsyncGenerator, Dict


class KvTransferSourceHandler:
    """Dynamo endpoint handler for source-side KV transfer RPCs.

    This is registered as the kv_transfer endpoint handler. Remote target
    workers call this endpoint to get metadata and block locations before
    executing NIXL RDMA reads.

    Matches the async generator pattern used by cache_control endpoint.
    """

    def __init__(self, manager: "KvTransferManager"):
        self.manager = manager
        # Block query cache: avoids repeated collective_rpc calls when
        # multiple target workers query for the same prefix simultaneously.
        # Since query_local_blocks blocks the event loop, requests are
        # serialized — the 2nd+ queries for the same prefix hit this cache.
        self._block_query_cache = {}
        self._block_query_cache_ttl = 2.0  # seconds

    async def handle_request(
        self, request: Dict[str, Any], context=None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Dispatch incoming kv_transfer RPC to appropriate handler.

        Actions:
        - get_metadata: Return NIXL RDMA metadata + KV layout info
        - query_blocks: Look up block locations in the radix tree
        """
        from dynamo.sglang.kv_transfer.metrics import record_transfer_request

        action = request.get("action")

        if action == "get_metadata":
            per_rank_metadata = self.manager.get_local_metadata()
            result = {"status": "ok", "per_rank": per_rank_metadata}
            record_transfer_request("source", "success")
        elif action == "query_blocks":
            token_ids = request.get("token_ids", [])
            num_blocks = request.get("num_blocks", 0)
            result = self._query_blocks_cached(token_ids, num_blocks)
            record_transfer_request("source", "success")
            if result.get("status") == "ok":
                matched = result.get("num_matched_blocks", 0)
                bpb = getattr(self.manager, "bytes_per_block", 0) or 0
                if isinstance(bpb, int) and bpb > 0:
                    from dynamo.sglang.kv_transfer.metrics import (
                        record_transfer_bytes,
                    )

                    record_transfer_bytes("source", matched * bpb)
                logging.info(
                    f"Source served {matched} blocks for transfer",
                    extra={
                        "event": "kv_transfer_source_served",
                        "num_blocks": matched,
                        "action": "query_blocks",
                    },
                )
        else:
            result = {"status": "error", "message": f"Unknown action: {action}"}
            record_transfer_request("source", "failed")

        yield result

    def _query_blocks_cached(self, token_ids: list, num_blocks: int) -> dict:
        """Query block locations with short-TTL caching.

        When multiple target workers simultaneously query for the same
        prefix (common in burst workloads), only the first query goes
        through collective_rpc. Subsequent queries hit the cache, avoiding
        the ~200ms collective_rpc overhead per redundant query.
        """
        cache_key = (hash(tuple(token_ids)), num_blocks)
        now = time.monotonic()

        cached = self._block_query_cache.get(cache_key)
        if cached is not None:
            result, expiry = cached
            if now < expiry:
                return result
            del self._block_query_cache[cache_key]

        result = self.manager.query_local_blocks(token_ids, num_blocks)
        if result.get("status") == "ok":
            self._block_query_cache[cache_key] = (
                result, now + self._block_query_cache_ttl
            )
        return result
