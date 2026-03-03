# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV Transfer Manager — orchestrates cross-worker KV cache transfers.

Runs in the main process. Read-only operations (metadata, block queries)
use Engine.collective_rpc() with the result_file pattern. Transfer
initiation uses dedicated per-rank ZMQ PUSH sockets for fire-and-forget
sends (~0ms), bypassing collective_rpc's ~530ms round-trip latency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from dynamo.sglang.args import Config

if TYPE_CHECKING:
    import sglang as sgl


class KvTransferManager:
    """Manages KV cache transfer operations for a single worker.

    Runs in the main process. Read-only operations use collective_rpc;
    transfer initiation uses dedicated per-rank ZMQ PUSH→PULL sockets.

    Owns:
    - Source handler (serves Dynamo RPC for remote workers)
    - Target handler (orchestrates block pull from remote workers)
    - Metadata cache for remote workers' NIXL/KV info
    - Per-rank ZMQ PUSH sockets for fire-and-forget transfer initiation

    TP support
    ----------
    Each TP rank has its own ZMQ PULL socket (bound in the scheduler)
    and a matching PUSH socket (connected here). Transfer requests are
    sent per-rank with that rank's source metadata, so no TP broadcast
    is needed.

    Transfer / prefill overlap
    --------------------------
    SGLang does not currently expose a mechanism for injecting externally
    transferred KV blocks into a running prefill.  The transfer flow is
    therefore intentionally serial: transfer blocks -> insert into radix
    tree -> issue generate request.  This avoids complexity without a
    measurable penalty, since the transferred blocks are cache hits that
    skip prefill computation entirely.
    """

    def __init__(
        self,
        engine: sgl.Engine,
        config: Config,
        metadata_cache_ttl_s: int = 300,
        transfer_timeout_ms: int = 5000,
        max_pending_kv_transfers: int = 4,
        max_nixl_starts_per_poll: int = 2,
    ):
        self.engine = engine
        self.config = config
        self.page_size = config.server_args.page_size
        self.transfer_timeout_ms = transfer_timeout_ms
        self.max_pending_kv_transfers = max_pending_kv_transfers
        self.max_nixl_starts_per_poll = max_nixl_starts_per_poll
        self.target_handler = None  # Set by init_decode
        # Bytes per KV block, computed after init_kv_transfer.
        # Used for byte-level transfer metrics.
        self.bytes_per_block = 0

        # Lock to serialize collective_rpc calls. SGLang's Engine uses ZMQ
        # sockets internally which are NOT thread-safe. Without this lock,
        # concurrent collective_rpc calls (e.g. source handler serving
        # metadata + block queries simultaneously) corrupt the ZMQ state
        # and cause recv_pyobj to hang forever.
        self._collective_rpc_lock = threading.Lock()

        # Cache this worker's own NIXL metadata (per-rank list). Populated
        # on first call to get_local_metadata(). This data is immutable
        # after init_kv_transfer, so we cache it permanently to avoid
        # repeated collective_rpc calls to the scheduler.
        self._local_metadata: Optional[List[dict]] = None

        # Cache of source worker metadata: worker_id -> (metadata_dict, expiry)
        self._metadata_cache: Dict[int, Tuple[dict, float]] = {}
        self._metadata_cache_ttl = metadata_cache_ttl_s

        # Track which source worker IDs have their metadata cached in the
        # scheduler process (to avoid resending large NIXL blobs via ZMQ).
        self._scheduler_cached_sources: set = set()

    async def initialize(self):
        """Initialize transfer infrastructure inside the scheduler process.

        Calls scheduler RPC to register KV cache memory with NIXL and
        create per-rank ZMQ PULL sockets. Then connects per-rank PUSH
        sockets from the main process for fire-and-forget transfer
        initiation (bypassing collective_rpc on the hot path).
        """
        try:
            def _init():
                # Use _rpc_with_result to get per-rank IPC paths back
                results = self._rpc_with_result(
                    "init_kv_transfer",
                    per_rank=True,
                    page_size=self.page_size,
                    max_pending_kv_transfers=self.max_pending_kv_transfers,
                    max_nixl_starts_per_poll=self.max_nixl_starts_per_poll,
                )

                # Create per-rank PUSH sockets connected to each rank's
                # PULL socket. These are used by execute_receive_transfer_async
                # for fire-and-forget transfer initiation (~0ms send).
                import zmq

                ctx = zmq.Context(1)
                self._kv_transfer_sockets = []
                bytes_per_block = 0
                for rank_result in results:
                    if rank_result.get("status") != "ok":
                        raise RuntimeError(
                            f"init_kv_transfer failed on rank: {rank_result}"
                        )
                    ipc_path = rank_result["ipc_path"]
                    if not bytes_per_block:
                        bytes_per_block = rank_result.get(
                            "bytes_per_block", 0
                        )
                    sock = ctx.socket(zmq.PUSH)
                    sock.setsockopt(zmq.SNDHWM, 64)
                    sock.connect(f"ipc://{ipc_path}")
                    self._kv_transfer_sockets.append(sock)
                self._kv_zmq_ctx = ctx
                return bytes_per_block

            bpb = await asyncio.to_thread(_init)
            self.bytes_per_block = bpb
            logging.info(
                f"KV Transfer: initialized with "
                f"{len(self._kv_transfer_sockets)} rank sockets"
            )
        except Exception as e:
            logging.error(f"KV Transfer: initialization failed: {e}")
            raise

    def _rpc_with_result(
        self, method: str, per_rank: bool = False, **kwargs
    ) -> Union[dict, List[dict]]:
        """Call a scheduler RPC method and read back JSON results via temp file.

        Since collective_rpc doesn't return data, we pass a result_file
        path. The scheduler writes JSON to it, we read and delete it.

        Args:
            per_rank: If True, read per-rank result files (result_file.rank0,
                .rank1, ...) written by each TP rank and return a list of
                dicts. If False (default), read a single result file written
                by rank 0 and return a single dict.
        """
        fd, result_file = tempfile.mkstemp(
            prefix="kv_xfer_", suffix=".json"
        )
        os.close(fd)
        try:
            with self._collective_rpc_lock:
                self.engine.collective_rpc(
                    method, result_file=result_file, **kwargs
                )
            if per_rank:
                tp_size = getattr(self.config.server_args, "tp_size", 1)
                results = []
                for rank in range(tp_size):
                    rank_file = f"{result_file}.rank{rank}"
                    with open(rank_file, "r") as f:
                        results.append(json.load(f))
                return results
            else:
                with open(result_file, "r") as f:
                    return json.load(f)
        finally:
            if per_rank:
                tp_size = getattr(self.config.server_args, "tp_size", 1)
                for rank in range(tp_size):
                    try:
                        os.unlink(f"{result_file}.rank{rank}")
                    except OSError:
                        pass
            try:
                os.unlink(result_file)
            except OSError:
                pass

    def get_local_metadata(self) -> List[dict]:
        """Get this worker's NIXL metadata from all TP ranks.

        Returns a list of metadata dicts, one per TP rank. Each dict
        contains the NIXL agent metadata and KV buffer pointers for
        that rank's GPU.

        Results are cached permanently since NIXL metadata is immutable
        after init_kv_transfer.
        """
        if self._local_metadata is not None:
            return self._local_metadata
        try:
            result = self._rpc_with_result(
                "get_kv_transfer_metadata", per_rank=True
            )
            self._local_metadata = result
            return result
        except Exception as e:
            logging.error(f"Failed to get local KV metadata: {e}")
            return [{"status": "error", "message": str(e)}]

    def query_local_blocks(self, token_ids: list, num_blocks: int) -> dict:
        """Query this worker's radix tree for block locations."""
        try:
            return self._rpc_with_result(
                "query_kv_blocks_by_tokens",
                token_ids=token_ids,
                num_blocks=num_blocks,
            )
        except Exception as e:
            logging.error(f"Block query failed: {e}")
            return {"status": "error", "message": str(e)}

    async def execute_receive_transfer(
        self,
        source_metadata_per_rank: list,
        source_kv_indices: list,
        token_ids: list,
        num_blocks: int,
        source_worker_id: Optional[int] = None,
    ) -> bool:
        """Execute a KV block receive on this worker (target side).

        Delegates to the scheduler process which allocates local pages,
        performs NIXL RDMA read, and inserts blocks into radix tree.

        Args:
            source_metadata_per_rank: List of metadata dicts, one per TP
                rank of the source worker. Passed through to
                receive_kv_blocks where each rank picks its own entry.
                Omitted (None sent to scheduler) when the scheduler already
                has this source's metadata cached.
            source_worker_id: Stable ID for the source worker. Used to
                cache metadata in the scheduler process so subsequent
                transfers skip sending the full metadata through ZMQ.

        Returns True on success, raises on failure.
        """
        def _locked_receive():
            # Only send full metadata on the first transfer from this source.
            # After that, the scheduler has it cached and we save KBs of
            # ZMQ serialization per call.
            if source_worker_id is not None and source_worker_id in self._scheduler_cached_sources:
                send_metadata = None
            else:
                send_metadata = source_metadata_per_rank

            t0 = time.time()
            with self._collective_rpc_lock:
                t_lock = time.time()
                self.engine.collective_rpc(
                    "receive_kv_blocks",
                    source_worker_id=source_worker_id,
                    source_metadata_per_rank=send_metadata,
                    source_kv_indices=source_kv_indices,
                    token_ids=token_ids,
                    num_blocks=num_blocks,
                    timeout_ms=self.transfer_timeout_ms,
                )
                t_done = time.time()

            # Mark as cached after successful transfer
            if source_worker_id is not None:
                self._scheduler_cached_sources.add(source_worker_id)

            logging.info(
                f"receive_transfer timing: "
                f"lock_wait={(t_lock - t0) * 1000:.0f}ms "
                f"rpc={(t_done - t_lock) * 1000:.0f}ms"
            )

        await asyncio.to_thread(_locked_receive)
        return True

    async def execute_receive_transfer_async(
        self,
        source_metadata_per_rank: list,
        source_kv_indices: list,
        token_ids: list,
        num_blocks: int,
        source_worker_id: Optional[int] = None,
    ) -> bool:
        """Execute a non-blocking KV block receive on this worker.

        Sends transfer requests directly to each TP rank's dedicated ZMQ
        PULL socket (fire-and-forget, ~0ms). The scheduler drains these
        in process_kv_transfers(), starts RDMA, and writes completion files.

        This completely bypasses collective_rpc and its ~530ms latency.
        No _collective_rpc_lock is needed — PUSH sockets are independent
        per rank and multiple transfers can be sent concurrently.

        Returns True on success, raises on failure.
        """
        fd, completion_file = tempfile.mkstemp(
            prefix="kv_completion_", suffix=".json"
        )
        os.close(fd)

        t_start = time.time()

        # Determine whether to send full metadata or rely on cache
        use_cache = (
            source_worker_id is not None
            and source_worker_id in self._scheduler_cached_sources
        )

        # Send to each rank's PUSH socket (fire-and-forget, ~0ms each)
        tp_size = len(self._kv_transfer_sockets)
        deadline = time.time() + self.transfer_timeout_ms / 1000.0

        for rank in range(tp_size):
            rank_metadata = None if use_cache else source_metadata_per_rank[rank]

            req = {
                "source_kv_indices": source_kv_indices,
                "token_ids": token_ids,
                "num_blocks": num_blocks,
                "source_metadata": rank_metadata,
                "source_worker_id": source_worker_id,
                "completion_file": completion_file,
                "timeout_ms": self.transfer_timeout_ms,
                "deadline": deadline,
            }
            self._kv_transfer_sockets[rank].send_pyobj(req)

        if source_worker_id is not None:
            self._scheduler_cached_sources.add(source_worker_id)

        t_sent = time.time()

        # Poll completion files from all TP ranks.
        # Each rank writes completion_file.rank{i} when its RDMA finishes.
        rank_files = [f"{completion_file}.rank{i}" for i in range(tp_size)]

        timeout_s = self.transfer_timeout_ms / 1000.0 + 5.0
        poll_start = time.monotonic()

        try:
            while True:
                all_done = True
                error = None
                for rank_file in rank_files:
                    if not os.path.exists(rank_file):
                        all_done = False
                        break
                    try:
                        with open(rank_file, "r") as f:
                            result = json.load(f)
                        if result.get("status") == "error":
                            error = result.get("message", "unknown error")
                            break
                    except (json.JSONDecodeError, IOError):
                        # File exists but not fully written yet
                        all_done = False
                        break

                if error:
                    raise RuntimeError(f"Async KV transfer failed: {error}")
                if all_done:
                    break
                if time.monotonic() - poll_start > timeout_s:
                    raise RuntimeError(
                        f"Async KV transfer completion poll timeout "
                        f"({timeout_s:.0f}s)"
                    )

                await asyncio.sleep(0.001)  # 1ms poll interval

            t_done = time.time()
            logging.info(
                f"receive_transfer_async timing: "
                f"send={(t_sent - t_start) * 1000:.0f}ms "
                f"rdma={(t_done - t_sent) * 1000:.0f}ms"
            )
        finally:
            # Clean up temp files
            for rank_file in rank_files:
                try:
                    os.unlink(rank_file)
                except OSError:
                    pass
            try:
                os.unlink(completion_file)
            except OSError:
                pass

        return True

    def get_cached_metadata(self, worker_id: int) -> Optional[dict]:
        """Get cached metadata for a remote worker."""
        if worker_id in self._metadata_cache:
            metadata, expiry = self._metadata_cache[worker_id]
            if time.time() < expiry:
                return metadata
            del self._metadata_cache[worker_id]
            # Also clear scheduler cache so next transfer sends fresh metadata
            self._scheduler_cached_sources.discard(worker_id)
        return None

    def cache_metadata(self, worker_id: int, metadata: dict):
        """Cache metadata from a remote worker."""
        self._metadata_cache[worker_id] = (
            metadata,
            time.time() + self._metadata_cache_ttl,
        )

    def invalidate_metadata(self, worker_id: int):
        """Invalidate cached metadata for a worker."""
        self._metadata_cache.pop(worker_id, None)
        self._scheduler_cached_sources.discard(worker_id)

    def publish_transferred_blocks(
        self,
        token_ids: list,
        num_blocks: int,
        kv_publisher,
    ):
        """Publish KvCacheEvent::Stored events for blocks received via transfer.

        After a successful RDMA transfer inserts blocks into the local radix
        tree, we publish events so the router's global radix tree learns about
        the new block locations on this worker.

        Hash compatibility
        ------------------
        SGLang's internal ``BlockStored`` events use SHA256 chain hashing
        (``get_hash_str`` + ``hash_str_to_int64`` from ``hicache_storage``).
        We must use the same functions so the router sees identical hashes
        for the same token sequence regardless of which worker published them.

        Args:
            token_ids: Full token sequence for the request.
            num_blocks: Number of blocks that were transferred.
            kv_publisher: KvEventPublisher instance (from publisher.kv_publisher).
        """
        from sglang.srt.mem_cache.hicache_storage import (
            get_hash_str,
            hash_str_to_int64,
        )

        page_size = self.page_size
        transferred_tokens = num_blocks * page_size
        prefix_token_ids = list(token_ids[:transferred_tokens])

        # Compute block hashes using the same SHA256 chain as SGLang's
        # radix cache BlockStored events.
        block_hashes = []
        prior_hash = None
        for block_idx in range(num_blocks):
            start = block_idx * page_size
            page_tokens = prefix_token_ids[start : start + page_size]
            hash_str = get_hash_str(page_tokens, prior_hash=prior_hash)
            block_hashes.append(hash_str_to_int64(hash_str))
            prior_hash = hash_str

        num_block_tokens = [page_size] * num_blocks

        kv_publisher.publish_stored(
            token_ids=prefix_token_ids,
            num_block_tokens=num_block_tokens,
            block_hashes=block_hashes,
            parent_hash=None,
        )
