# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scheduler mixin for KV transfer operations.

These methods run INSIDE the SGLang scheduler process and have direct
access to GPU buffers, the radix tree, and the page allocator.

Installation: At import time, install_kv_transfer_methods(Scheduler) is called
to add methods to the Scheduler class. These methods are then callable via
Engine.collective_rpc(method_name, **kwargs).

Data return: Since collective_rpc only returns success/failure, methods that
need to return data accept a `result_file` parameter. The scheduler writes
JSON results to this file, and the main process reads it after collective_rpc
returns. This works because collective_rpc is synchronous (blocking).

Dedicated ZMQ channel for transfers: Transfer initiation bypasses
collective_rpc entirely. The main process sends requests via per-rank
ZMQ PUSH sockets; the scheduler drains them from PULL sockets each
iteration in process_kv_transfers(). This avoids the ~530ms latency
of collective_rpc (which waits for the scheduler's event loop) and
removes lock serialization for concurrent transfers.

Non-blocking RDMA: process_kv_transfers polls pending RDMA transfers
each iteration. Completion is signalled via atomic completion files.
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)


def install_kv_transfer_methods(scheduler_class):
    """Install KV transfer methods onto the Scheduler class.

    Called once at import time, before Engine spawns the scheduler subprocess.
    Adds methods that the RPC dispatcher can invoke by name via
    getattr(self, method_name)(**params).
    """

    def init_kv_transfer(
        self,
        page_size=16,
        result_file=None,
        max_pending_kv_transfers=4,
        max_nixl_starts_per_poll=2,
    ):
        """Initialize NIXL transfer infrastructure.

        Registers KV cache GPU memory with NIXL agent for remote reads.
        Called via Engine.collective_rpc("init_kv_transfer").

        Also creates a dedicated ZMQ PULL socket for receiving KV transfer
        requests directly (bypassing collective_rpc). The IPC path is
        written to result_file so the main process can connect PUSH sockets.
        """
        try:
            from nixl._api import nixl_agent, nixl_agent_config
        except ImportError:
            logger.warning("NIXL not available, KV transfer disabled")
            self._kv_transfer_enabled = False
            return

        self._kv_transfer_enabled = True
        self._kv_transfer_page_size = page_size

        # Get KV cache buffer info from the memory pool
        kv_cache = self.tp_worker.model_runner.token_to_kv_pool

        self._kv_data_ptrs, self._kv_data_lens, self._kv_item_lens = (
            kv_cache.get_contiguous_buf_infos()
        )

        # Determine GPU ID from the KV cache buffers
        if hasattr(kv_cache, "k_buffer") and len(kv_cache.k_buffer) > 0:
            self._kv_gpu_id = kv_cache.k_buffer[0].device.index or 0
        elif hasattr(kv_cache, "kv_buffer") and len(kv_cache.kv_buffer) > 0:
            self._kv_gpu_id = kv_cache.kv_buffer[0].device.index or 0
        else:
            self._kv_gpu_id = 0

        # Create NIXL agent for this worker
        import uuid

        agent_name = f"kv_transfer_{uuid.uuid4()}"
        agent_config = nixl_agent_config(backends=["UCX"])
        self._nixl_agent = nixl_agent(agent_name, agent_config)

        # Register KV buffers with NIXL for remote access
        kv_addrs = []
        for ptr, length in zip(self._kv_data_ptrs, self._kv_data_lens):
            kv_addrs.append((ptr, length, self._kv_gpu_id, ""))
        self._nixl_kv_descs = self._nixl_agent.register_memory(kv_addrs, "VRAM")

        # Export agent metadata for remote workers to connect
        self._nixl_agent_metadata = self._nixl_agent.get_agent_metadata()

        # Number of layers: kv_data_ptrs has K + V per layer for MHA,
        # or just one buffer per layer for MLA
        if hasattr(kv_cache, "k_buffer"):
            self._kv_num_layers = len(self._kv_data_ptrs) // 2
        else:
            self._kv_num_layers = len(self._kv_data_ptrs)

        # Track remote peers for target-side RDMA reads
        self._nixl_remote_peers = {}

        logger.info(
            f"KV Transfer initialized: {self._kv_num_layers} layers, "
            f"GPU {self._kv_gpu_id}, {len(kv_addrs)} buffers registered, "
            f"page_size={page_size}"
        )

        # Track pending transfer locks for eviction protection (Phase 4)
        self._transfer_pending_locks = {}

        # Cache source worker metadata to avoid resending through ZMQ
        # on every collective_rpc call. Keyed by source_worker_id.
        self._nixl_source_metadata_cache = {}

        # Pending async KV transfers: transfer_id -> state dict.
        # Populated by _handle_kv_transfer_request, drained by
        # process_kv_transfers.
        self._pending_kv_transfers = {}

        # Transfer page budget: tracks tokens held by in-flight RDMA
        # transfers (allocated but not yet inserted into radix tree).
        self._transfer_pages_held = 0

        # Configurable concurrency limits (from CLI flags)
        self._max_pending_kv_transfers = max_pending_kv_transfers
        self._max_nixl_starts_per_poll = max_nixl_starts_per_poll

        # Create dedicated ZMQ PULL socket for receiving KV transfer
        # requests directly from the main process (bypasses collective_rpc).
        import tempfile

        import zmq

        tp_rank = getattr(self, "tp_rank", 0)
        ipc_path = tempfile.mktemp(
            prefix=f"kv_xfer_rank{tp_rank}_", dir="/tmp"
        )
        self._kv_transfer_ipc_path = ipc_path

        kv_ctx = zmq.Context(1)
        self._kv_transfer_recv = kv_ctx.socket(zmq.PULL)
        self._kv_transfer_recv.bind(f"ipc://{ipc_path}")
        self._kv_transfer_zmq_ctx = kv_ctx

        logger.info(
            f"KV Transfer socket bound: ipc://{ipc_path} (rank {tp_rank})"
        )

        # Compute bytes_per_block for transfer metrics.
        # sum(item_lens) gives total bytes per token across all KV buffers;
        # multiply by page_size to get bytes per block.
        self._bytes_per_block = sum(self._kv_item_lens) * page_size

        # Write IPC path to result_file for main process to read
        if result_file:
            rank_file = f"{result_file}.rank{tp_rank}"
            with open(rank_file, "w") as f:
                json.dump({
                    "status": "ok",
                    "ipc_path": ipc_path,
                    "bytes_per_block": self._bytes_per_block,
                }, f)

    def get_kv_transfer_metadata(self, result_file=None):
        """Return this worker's NIXL metadata + KV layout for remote workers.

        Called via Engine.collective_rpc("get_kv_transfer_metadata",
            result_file="/tmp/...").
        Writes result to result_file as JSON (since collective_rpc can't
        return data).
        """
        if not getattr(self, "_kv_transfer_enabled", False):
            data = {"status": "error", "message": "KV transfer not initialized"}
        else:
            import base64

            data = {
                "status": "ok",
                "agent_metadata": base64.b64encode(
                    self._nixl_agent_metadata
                ).decode("ascii"),
                "kv_data_ptrs": list(self._kv_data_ptrs),
                "kv_data_lens": list(self._kv_data_lens),
                "kv_item_lens": list(self._kv_item_lens),
                "gpu_id": self._kv_gpu_id,
                "num_layers": self._kv_num_layers,
                "page_size": self._kv_transfer_page_size,
            }

        if result_file:
            tp_rank = getattr(self, "tp_rank", 0)
            rank_file = f"{result_file}.rank{tp_rank}"
            with open(rank_file, "w") as f:
                json.dump(data, f)

    def query_kv_blocks_by_tokens(self, token_ids, num_blocks, result_file=None):
        """Look up KV page indices by matching token prefix in radix tree.

        The target worker sends its token sequence; we match against our
        local radix tree to find which physical KV pages hold cached blocks.

        Called via Engine.collective_rpc("query_kv_blocks_by_tokens", ...).
        """
        if not getattr(self, "_kv_transfer_enabled", False):
            data = {"status": "error", "message": "KV transfer not initialized"}
        else:
            try:
                from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
                from sglang.srt.mem_cache.radix_cache import RadixKey

                key = RadixKey(token_ids=list(token_ids))
                match_result = self.tree_cache.match_prefix(
                    MatchPrefixParams(key=key)
                )

                matched_indices = match_result.device_indices
                page_size = self._kv_transfer_page_size

                # Cap at requested num_blocks pages
                max_tokens = num_blocks * page_size
                if len(matched_indices) > max_tokens:
                    matched_indices = matched_indices[:max_tokens]

                # Extract page-level indices (one per page_size tokens)
                num_matched = len(matched_indices) // page_size
                kv_indices = []
                for i in range(num_matched):
                    idx = int(matched_indices[i * page_size])
                    kv_indices.append(idx)

                data = {
                    "status": "ok",
                    "kv_indices": kv_indices,
                    "num_matched_blocks": num_matched,
                }
            except Exception as e:
                logger.warning(f"Block query failed: {e}")
                data = {"status": "error", "message": str(e)}

        if result_file and getattr(self, "tp_rank", 0) == 0:
            with open(result_file, "w") as f:
                json.dump(data, f)

    def _resolve_source_metadata(
        self,
        source_worker_id,
        source_metadata_per_rank=None,
        source_metadata=None,
    ):
        """Resolve source metadata from parameters or cache, add NIXL peer.

        Returns (source_metadata, peer_name) for this TP rank.
        Shared by the synchronous (receive_kv_blocks), collective_rpc
        (initiate_kv_transfer), and socket-based (_handle_kv_transfer_request)
        paths.

        Args:
            source_metadata_per_rank: List of per-rank metadata dicts
                (from collective_rpc path, where all ranks get the same list).
            source_metadata: Single-rank metadata dict (from socket path,
                where each rank receives only its own metadata).
        """
        import base64

        tp_rank = getattr(self, "tp_rank", 0)

        # Socket path: single-rank metadata provided directly
        if source_metadata is not None:
            if source_worker_id is not None:
                self._nixl_source_metadata_cache[source_worker_id] = (
                    source_metadata
                )
        elif source_metadata_per_rank is not None:
            # Collective_rpc path: pick this rank's entry from the list
            source_metadata = source_metadata_per_rank[tp_rank]
            if source_worker_id is not None:
                self._nixl_source_metadata_cache[source_worker_id] = (
                    source_metadata
                )
        elif (
            source_worker_id is not None
            and source_worker_id in self._nixl_source_metadata_cache
        ):
            source_metadata = self._nixl_source_metadata_cache[source_worker_id]
        else:
            raise RuntimeError(
                "No source metadata provided and none cached "
                f"(source_worker_id={source_worker_id})"
            )

        source_agent_metadata_b64 = source_metadata["agent_metadata"]
        source_agent_metadata = base64.b64decode(source_agent_metadata_b64)
        peer_key = source_agent_metadata_b64

        if peer_key not in self._nixl_remote_peers:
            peer_name = self._nixl_agent.add_remote_agent(source_agent_metadata)
            self._nixl_remote_peers[peer_key] = peer_name
            try:
                self._nixl_agent.make_connection(peer_name)
            except Exception as e:
                logger.warning(
                    f"make_connection to {peer_name} failed (non-fatal): {e}"
                )
            logger.info(f"Added remote NIXL peer: {peer_name}")

        peer_name = self._nixl_remote_peers[peer_key]
        return source_metadata, peer_name

    def receive_kv_blocks(
        self,
        source_kv_indices,
        token_ids,
        num_blocks,
        timeout_ms=5000,
        source_worker_id=None,
        source_metadata_per_rank=None,
    ):
        """Receive KV blocks from a remote source worker via NIXL RDMA.

        Synchronous fallback path. Blocks the scheduler for the full RDMA
        duration. Prefer initiate_kv_transfer + process_kv_transfers for
        non-blocking transfers.

        Called via Engine.collective_rpc("receive_kv_blocks", ...).
        Raises on failure (collective_rpc will report the error).
        """
        if not getattr(self, "_kv_transfer_enabled", False):
            raise RuntimeError("KV transfer not initialized")

        source_metadata, peer_name = self._resolve_source_metadata(
            source_worker_id, source_metadata_per_rank
        )

        page_size = self._kv_transfer_page_size
        need_tokens = num_blocks * page_size

        # Budget check: prevent transfers from starving normal requests
        max_transfer_tokens = int(
            self.token_to_kv_pool_allocator.size * _TRANSFER_BUDGET_FRACTION
        )
        if self._transfer_pages_held + need_tokens > max_transfer_tokens:
            raise RuntimeError(
                f"Transfer page budget exceeded "
                f"({self._transfer_pages_held + need_tokens} > "
                f"{max_transfer_tokens})"
            )

        local_indices = self._alloc_with_eviction(need_tokens)
        if local_indices is None:
            raise RuntimeError(
                f"Could not allocate local KV pages for transfer "
                f"(need {need_tokens}, "
                f"avail {self.token_to_kv_pool_allocator.available_size()})"
            )

        try:
            self._execute_nixl_transfer(
                source_metadata, source_kv_indices, local_indices,
                num_blocks, page_size, peer_name, timeout_ms,
            )

            self._insert_transferred_blocks(
                local_indices, token_ids, num_blocks, page_size,
            )

        except Exception:
            try:
                self.token_to_kv_pool_allocator.free(local_indices)
            except Exception:
                pass
            raise

    def _insert_transferred_blocks(
        self, local_indices, token_ids, num_blocks, page_size,
    ):
        """Insert transferred blocks into radix tree with eviction protection.

        Shared by both the synchronous and asynchronous completion paths.
        """
        transferred_tokens = num_blocks * page_size
        prefix_token_ids = list(token_ids[:transferred_tokens])

        from sglang.srt.mem_cache.base_prefix_cache import (
            InsertParams,
            MatchPrefixParams,
        )
        from sglang.srt.mem_cache.radix_cache import RadixKey

        key = RadixKey(token_ids=prefix_token_ids)
        insert_result = self.tree_cache.insert(
            InsertParams(
                key=key,
                value=local_indices[:transferred_tokens],
            )
        )

        if insert_result.prefix_len > 0:
            duplicate_indices = local_indices[: insert_result.prefix_len]
            self.token_to_kv_pool_allocator.free(duplicate_indices)

        match_result = self.tree_cache.match_prefix(
            MatchPrefixParams(key=key)
        )
        transfer_last_node = match_result.last_device_node
        if transfer_last_node is not self.tree_cache.root_node:
            self.tree_cache.inc_lock_ref(transfer_last_node)
            lock_key = (
                tuple(prefix_token_ids),
                time.time_ns(),
            )
            self._transfer_pending_locks[lock_key] = (
                transfer_last_node,
                time.time(),
            )

        logger.info(
            f"KV transfer complete: {num_blocks} blocks "
            f"({transferred_tokens} tokens) received and inserted"
            f" (prefix_overlap={insert_result.prefix_len},"
            f" locked={'yes' if transfer_last_node is not self.tree_cache.root_node else 'no'})"
        )

    def _execute_nixl_transfer(
        self, source_metadata, source_kv_indices, local_indices,
        num_blocks, page_size, peer_name, timeout_ms,
    ):
        """Build descriptors and execute the NIXL RDMA transfer (synchronous).

        Blocks until RDMA completes or times out. Used by receive_kv_blocks
        as the synchronous fallback path.

        Uses prep_xfer_dlist + make_prepped_xfer for explicit resource
        lifecycle management. All NIXL handles (local_prep, remote_prep,
        xfer_handle) are released in a finally block.
        """
        import numpy as np

        src_kv_ptrs = source_metadata["kv_data_ptrs"]
        src_gpu_id = source_metadata["gpu_id"]

        src_addrs = []
        dst_addrs = []

        for layer_buf_idx in range(len(self._kv_data_ptrs)):
            src_ptr = src_kv_ptrs[layer_buf_idx]
            dst_ptr = self._kv_data_ptrs[layer_buf_idx]
            item_len = self._kv_item_lens[layer_buf_idx]
            per_token_bytes = item_len // page_size

            for block_idx in range(num_blocks):
                src_token_idx = source_kv_indices[block_idx]
                dst_token_idx = int(local_indices[block_idx * page_size])

                src_offset = src_token_idx * per_token_bytes
                src_addrs.append(
                    (src_ptr + src_offset, item_len, src_gpu_id)
                )

                dst_offset = dst_token_idx * per_token_bytes
                dst_addrs.append(
                    (dst_ptr + dst_offset, item_len, self._kv_gpu_id)
                )

        num_descs = len(src_addrs)
        src_arr = np.array(src_addrs, dtype=np.uint64)
        dst_arr = np.array(dst_addrs, dtype=np.uint64)

        src_descs = self._nixl_agent.get_xfer_descs(src_arr, "VRAM")
        dst_descs = self._nixl_agent.get_xfer_descs(dst_arr, "VRAM")

        local_prep = None
        remote_prep = None
        xfer_handle = None

        try:
            t0 = time.time_ns()
            local_prep = self._nixl_agent.prep_xfer_dlist(
                "NIXL_INIT_AGENT", dst_descs, "VRAM"
            )
            remote_prep = self._nixl_agent.prep_xfer_dlist(
                peer_name, src_descs, "VRAM"
            )

            indices = np.arange(num_descs, dtype=np.int32)
            notif_key = f"kv_xfer_{time.time_ns()}"

            xfer_handle = self._nixl_agent.make_prepped_xfer(
                "READ",
                local_prep,   # local destination (prepared)
                indices,       # use all local descriptors
                remote_prep,   # remote source (prepared)
                indices,       # use all remote descriptors
                notif_key.encode("ascii"),
            )
            t_prep = time.time_ns()

            if not xfer_handle:
                raise RuntimeError("Failed to create NIXL transfer handle")

            state = self._nixl_agent.transfer(xfer_handle)
            if state == "ERR":
                raise RuntimeError("NIXL transfer initiation failed")

            timeout_ns = timeout_ms * 1_000_000
            start = time.time_ns()
            while True:
                state = self._nixl_agent.check_xfer_state(xfer_handle)
                if state == "DONE":
                    break
                if state == "ERR":
                    raise RuntimeError("NIXL transfer error during execution")
                if time.time_ns() - start > timeout_ns:
                    raise RuntimeError(
                        f"NIXL transfer timeout ({timeout_ms}ms)"
                    )
                time.sleep(0.0001)  # 100us poll interval
            t_done = time.time_ns()
            rdma_seconds = (t_done - t_prep) / 1e9
            logger.info(
                f"NIXL timing: {num_descs} descs, "
                f"prep={(t_prep - t0) / 1e6:.1f}ms "
                f"rdma={rdma_seconds * 1000:.1f}ms"
            )
            try:
                from dynamo.sglang.kv_transfer.metrics import (
                    record_nixl_read_op,
                )

                rdma_byte_count = (
                    num_blocks * getattr(self, "_bytes_per_block", 0)
                )
                if rdma_byte_count > 0:
                    record_nixl_read_op(rdma_byte_count, rdma_seconds)
            except Exception:
                pass

        finally:
            if xfer_handle is not None:
                try:
                    xfer_handle.release()
                except Exception:
                    pass
            if remote_prep is not None:
                try:
                    remote_prep.release()
                except Exception:
                    pass
            if local_prep is not None:
                try:
                    local_prep.release()
                except Exception:
                    pass

    def _start_nixl_transfer(
        self, source_metadata, source_kv_indices, local_indices,
        num_blocks, page_size, peer_name,
    ):
        """Build descriptors and START the NIXL RDMA transfer (non-blocking).

        Returns (xfer_handle, local_prep, remote_prep, prep_ms) on success.
        The caller is responsible for polling check_xfer_state() and releasing
        all handles when done.

        Raises on failure (descriptors/handles are cleaned up internally).
        """
        import numpy as np

        src_kv_ptrs = source_metadata["kv_data_ptrs"]
        src_gpu_id = source_metadata["gpu_id"]

        src_addrs = []
        dst_addrs = []

        for layer_buf_idx in range(len(self._kv_data_ptrs)):
            src_ptr = src_kv_ptrs[layer_buf_idx]
            dst_ptr = self._kv_data_ptrs[layer_buf_idx]
            item_len = self._kv_item_lens[layer_buf_idx]
            per_token_bytes = item_len // page_size

            for block_idx in range(num_blocks):
                src_token_idx = source_kv_indices[block_idx]
                dst_token_idx = int(local_indices[block_idx * page_size])

                src_offset = src_token_idx * per_token_bytes
                src_addrs.append(
                    (src_ptr + src_offset, item_len, src_gpu_id)
                )

                dst_offset = dst_token_idx * per_token_bytes
                dst_addrs.append(
                    (dst_ptr + dst_offset, item_len, self._kv_gpu_id)
                )

        num_descs = len(src_addrs)
        src_arr = np.array(src_addrs, dtype=np.uint64)
        dst_arr = np.array(dst_addrs, dtype=np.uint64)

        src_descs = self._nixl_agent.get_xfer_descs(src_arr, "VRAM")
        dst_descs = self._nixl_agent.get_xfer_descs(dst_arr, "VRAM")

        local_prep = None
        remote_prep = None
        xfer_handle = None

        try:
            t0 = time.time_ns()
            local_prep = self._nixl_agent.prep_xfer_dlist(
                "NIXL_INIT_AGENT", dst_descs, "VRAM"
            )
            remote_prep = self._nixl_agent.prep_xfer_dlist(
                peer_name, src_descs, "VRAM"
            )

            indices = np.arange(num_descs, dtype=np.int32)
            notif_key = f"kv_xfer_{time.time_ns()}"

            xfer_handle = self._nixl_agent.make_prepped_xfer(
                "READ",
                local_prep,
                indices,
                remote_prep,
                indices,
                notif_key.encode("ascii"),
            )

            if not xfer_handle:
                raise RuntimeError("Failed to create NIXL transfer handle")

            state = self._nixl_agent.transfer(xfer_handle)
            if state == "ERR":
                raise RuntimeError("NIXL transfer initiation failed")

            t_prep = time.time_ns()
            prep_ms = (t_prep - t0) / 1e6

            return xfer_handle, local_prep, remote_prep, prep_ms

        except Exception:
            if xfer_handle is not None:
                try:
                    xfer_handle.release()
                except Exception:
                    pass
            if remote_prep is not None:
                try:
                    remote_prep.release()
                except Exception:
                    pass
            if local_prep is not None:
                try:
                    local_prep.release()
                except Exception:
                    pass
            raise

    def process_kv_transfers(self):
        """Poll pending async KV transfers and receive new requests.

        Called every scheduler iteration from the wrapped process_input_requests.

        Phase A: Drain new transfer requests from the dedicated ZMQ PULL
        socket (non-blocking). Each request is processed immediately:
        allocate pages, start RDMA, add to pending dict.

        Phase B: Poll pending RDMA transfers:
        - DONE: release handles, insert into radix tree, write completion file
        - ERR: release handles, free pages, write error to completion file
        - PROC: skip (check again next iteration)
        - Timeout: treat as error
        """
        # Phase A: Receive new transfer requests from dedicated socket.
        # Rate-limit RDMA starts to avoid overwhelming the NIXL/RDMA layer
        # with burst traffic. Excess messages are drained from ZMQ but
        # rejected immediately so senders fall back to full prefill fast.
        if hasattr(self, "_kv_transfer_recv"):
            import zmq

            nixl_starts = 0
            while True:
                try:
                    req = self._kv_transfer_recv.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                if nixl_starts >= self._max_nixl_starts_per_poll:
                    # Drain but reject — too many RDMA starts this iteration
                    self._write_completion_error(
                        req.get("completion_file"),
                        "Transfer rate limited (burst)",
                    )
                    continue
                pending_before = len(self._pending_kv_transfers)
                self._handle_kv_transfer_request(req)
                if len(self._pending_kv_transfers) > pending_before:
                    nixl_starts += 1

        if not hasattr(self, "_pending_kv_transfers"):
            return
        if not self._pending_kv_transfers:
            return

        completed = []
        for transfer_id, state in self._pending_kv_transfers.items():
            xfer_handle = state["xfer_handle"]

            try:
                xfer_state = self._nixl_agent.check_xfer_state(xfer_handle)
            except Exception as e:
                logger.error(
                    f"process_kv_transfers: check_xfer_state failed "
                    f"for {transfer_id}: {e}"
                )
                xfer_state = "ERR"

            if xfer_state == "DONE":
                self._complete_kv_transfer(transfer_id, state)
                completed.append(transfer_id)
            elif xfer_state == "ERR":
                self._fail_kv_transfer(
                    transfer_id, state, "NIXL transfer error"
                )
                completed.append(transfer_id)
            else:
                # Still in progress — check timeout
                elapsed_ms = (time.time() - state["start_time"]) * 1000
                if elapsed_ms > state["timeout_ms"]:
                    self._fail_kv_transfer(
                        transfer_id, state,
                        f"NIXL transfer timeout ({state['timeout_ms']}ms)",
                    )
                    completed.append(transfer_id)

        for transfer_id in completed:
            del self._pending_kv_transfers[transfer_id]

        # Phase C: Periodic cleanup of stale transfer locks.
        # Releases locks from completed transfers whose requests have
        # arrived (or never will), keeping evictable pool pages available.
        self._cleanup_stale_transfer_locks()

    def _complete_kv_transfer(self, transfer_id, state):
        """Handle successful RDMA completion for an async transfer.

        Releases NIXL handles, inserts blocks into radix tree, and writes
        completion status to the completion file.
        """
        rdma_ms = (time.time() - state["start_time"]) * 1000

        # Release NIXL handles
        for handle in (state["xfer_handle"], state["remote_prep"], state["local_prep"]):
            if handle is not None:
                try:
                    handle.release()
                except Exception:
                    pass

        # Release transfer page budget before insertion (pages transfer
        # ownership from "in-flight budget" to "radix tree")
        released_tokens = state["num_blocks"] * state["page_size"]
        self._transfer_pages_held = max(
            0, self._transfer_pages_held - released_tokens
        )

        # Insert into radix tree with eviction protection
        self._insert_transferred_blocks(
            state["local_indices"],
            state["token_ids"],
            state["num_blocks"],
            state["page_size"],
        )

        logger.info(
            f"process_kv_transfers: completed {transfer_id}, "
            f"{state['num_blocks']} blocks, prep={state['prep_ms']:.1f}ms "
            f"rdma={rdma_ms:.1f}ms"
        )
        try:
            from dynamo.sglang.kv_transfer.metrics import record_nixl_read_op

            rdma_byte_count = (
                state["num_blocks"] * getattr(self, "_bytes_per_block", 0)
            )
            if rdma_byte_count > 0:
                record_nixl_read_op(rdma_byte_count, rdma_ms / 1000.0)
        except Exception:
            pass

        # Write completion file (atomic rename to prevent partial reads)
        completion_file = state.get("completion_file")
        if completion_file:
            tp_rank = getattr(self, "tp_rank", 0)
            rank_file = f"{completion_file}.rank{tp_rank}"
            tmp_file = rank_file + ".tmp"
            with open(tmp_file, "w") as f:
                json.dump({"status": "ok", "transfer_id": transfer_id}, f)
            os.rename(tmp_file, rank_file)

    def _fail_kv_transfer(self, transfer_id, state, error_msg):
        """Handle failed or timed-out async transfer.

        Releases NIXL handles, frees allocated pages, and writes error
        status to the completion file.
        """
        # Release NIXL handles
        for handle in (state["xfer_handle"], state["remote_prep"], state["local_prep"]):
            if handle is not None:
                try:
                    handle.release()
                except Exception:
                    pass

        # Free allocated pages and release transfer page budget
        try:
            self.token_to_kv_pool_allocator.free(state["local_indices"])
        except Exception:
            pass
        released_tokens = state["num_blocks"] * state["page_size"]
        self._transfer_pages_held = max(
            0, self._transfer_pages_held - released_tokens
        )

        logger.error(
            f"process_kv_transfers: failed {transfer_id}: {error_msg}"
        )

        # Write error to completion file (atomic rename)
        completion_file = state.get("completion_file")
        if completion_file:
            tp_rank = getattr(self, "tp_rank", 0)
            rank_file = f"{completion_file}.rank{tp_rank}"
            tmp_file = rank_file + ".tmp"
            with open(tmp_file, "w") as f:
                json.dump({
                    "status": "error",
                    "message": error_msg,
                    "transfer_id": transfer_id,
                }, f)
            os.rename(tmp_file, rank_file)

    def _handle_kv_transfer_request(self, req):
        """Process a KV transfer request received from the dedicated socket.

        Same logic as the old initiate_kv_transfer, but reads parameters
        from a dict sent by the main process via ZMQ PUSH.
        """
        completion_file = req.get("completion_file")

        # Guard 1: Reject if too many concurrent transfers
        if len(self._pending_kv_transfers) >= self._max_pending_kv_transfers:
            self._write_completion_error(
                completion_file, "Too many pending transfers"
            )
            return

        # Guard 2: Skip stale requests (main process already timed out)
        deadline = req.get("deadline", float("inf"))
        if time.time() > deadline:
            logger.warning("Dropping stale KV transfer request")
            return

        if not getattr(self, "_kv_transfer_enabled", False):
            self._write_completion_error(
                completion_file, "KV transfer not initialized"
            )
            return

        source_worker_id = req.get("source_worker_id")
        source_metadata = req.get("source_metadata")  # single-rank dict
        source_kv_indices = req["source_kv_indices"]
        token_ids = req["token_ids"]
        num_blocks = req["num_blocks"]
        timeout_ms = req.get("timeout_ms", 5000)

        try:
            source_meta, peer_name = self._resolve_source_metadata(
                source_worker_id,
                source_metadata=source_metadata,
            )
        except Exception as e:
            self._write_completion_error(
                completion_file, f"Metadata resolution failed: {e}"
            )
            return

        page_size = self._kv_transfer_page_size
        need_tokens = num_blocks * page_size

        # Budget check: prevent transfers from starving normal requests
        max_transfer_tokens = int(
            self.token_to_kv_pool_allocator.size * _TRANSFER_BUDGET_FRACTION
        )
        if self._transfer_pages_held + need_tokens > max_transfer_tokens:
            self._write_completion_error(
                completion_file,
                f"Transfer page budget exceeded "
                f"({self._transfer_pages_held + need_tokens} > "
                f"{max_transfer_tokens})",
            )
            return

        local_indices = self._alloc_with_eviction(need_tokens)
        if local_indices is None:
            self._write_completion_error(
                completion_file,
                f"Could not allocate local KV pages for transfer "
                f"(need {need_tokens}, "
                f"avail {self.token_to_kv_pool_allocator.available_size()})",
            )
            return

        self._transfer_pages_held += need_tokens

        try:
            xfer_handle, local_prep, remote_prep, prep_ms = (
                self._start_nixl_transfer(
                    source_meta,
                    source_kv_indices,
                    local_indices,
                    num_blocks,
                    page_size,
                    peer_name,
                )
            )
        except Exception as e:
            try:
                self.token_to_kv_pool_allocator.free(local_indices)
            except Exception:
                pass
            self._transfer_pages_held = max(
                0, self._transfer_pages_held - need_tokens
            )
            self._write_completion_error(
                completion_file, f"NIXL transfer start failed: {e}"
            )
            return

        tp_rank = getattr(self, "tp_rank", 0)
        transfer_id = f"kv_xfer_{time.time_ns()}_{tp_rank}"

        self._pending_kv_transfers[transfer_id] = {
            "xfer_handle": xfer_handle,
            "local_prep": local_prep,
            "remote_prep": remote_prep,
            "local_indices": local_indices,
            "token_ids": list(token_ids[:need_tokens]),
            "num_blocks": num_blocks,
            "page_size": page_size,
            "completion_file": completion_file,
            "start_time": time.time(),
            "timeout_ms": timeout_ms,
            "prep_ms": prep_ms,
        }

        logger.info(
            f"_handle_kv_transfer_request: {transfer_id} started, "
            f"{num_blocks} blocks, prep={prep_ms:.1f}ms"
        )

    def _write_completion_error(self, completion_file, error_msg):
        """Write an error status to the completion file for the main process."""
        if not completion_file:
            logger.error(
                f"KV transfer error (no completion file): {error_msg}"
            )
            return

        tp_rank = getattr(self, "tp_rank", 0)
        rank_file = f"{completion_file}.rank{tp_rank}"
        tmp_file = rank_file + ".tmp"
        try:
            with open(tmp_file, "w") as f:
                json.dump({"status": "error", "message": error_msg}, f)
            os.rename(tmp_file, rank_file)
        except Exception as e:
            logger.error(
                f"Failed to write completion error file: {e}"
            )

    # Maximum fraction of KV pool that transfers may consume.
    # Prevents transfer bursts from starving normal request processing.
    _TRANSFER_BUDGET_FRACTION = 0.5

    def _alloc_with_eviction(self, need_tokens):
        """Allocate KV pages for transfer, evicting cache entries if needed.

        Mirrors SGLang's evict_from_tree_cache() pattern:
        1. Check available_size()
        2. If insufficient, release stale transfer locks (aggressive, 3s)
        3. If still insufficient, evict LRU leaves from radix tree
        4. Attempt allocation
        5. If still fails, nuclear fallback: release ALL locks, evict again

        Returns tensor of page indices, or None if truly out of memory.
        """
        import torch

        allocator = self.token_to_kv_pool_allocator

        # Fast path: enough free pages already
        result = allocator.alloc(need_tokens)
        if result is not None and not (
            isinstance(result, torch.Tensor) and result.numel() == 0
        ):
            return result

        # Step 1: Release stale transfer locks aggressively (3s not 30s).
        # This unlocks radix tree nodes from completed transfers whose
        # requests have already arrived, making them evictable.
        self._cleanup_stale_transfer_locks(max_age_s=3)

        # Step 2: Evict from radix tree if still insufficient
        available = allocator.available_size()
        if available < need_tokens:
            shortfall = need_tokens - available
            # Request 2x shortfall to leave headroom for the next transfer
            evict_target = shortfall * 2
            try:
                from sglang.srt.mem_cache.base_prefix_cache import EvictParams

                evict_result = self.tree_cache.evict(
                    EvictParams(num_tokens=evict_target)
                )
                if evict_result.num_tokens_evicted > 0:
                    logger.info(
                        f"Transfer alloc: evicted "
                        f"{evict_result.num_tokens_evicted} tokens from "
                        f"radix tree (needed {need_tokens}, had {available})"
                    )
            except Exception as e:
                logger.warning(f"Transfer alloc: eviction failed: {e}")

        # Step 3: Attempt allocation
        result = allocator.alloc(need_tokens)
        if result is not None and not (
            isinstance(result, torch.Tensor) and result.numel() == 0
        ):
            return result

        # Step 4: Nuclear fallback — release ALL transfer locks and
        # evict again. This sacrifices recently-transferred cache entries
        # to make room for the current transfer.
        self._cleanup_stale_transfer_locks(max_age_s=0)
        available = allocator.available_size()
        if available < need_tokens:
            try:
                from sglang.srt.mem_cache.base_prefix_cache import EvictParams

                self.tree_cache.evict(
                    EvictParams(num_tokens=need_tokens - available)
                )
            except Exception:
                pass
        result = allocator.alloc(need_tokens)
        if result is not None and not (
            isinstance(result, torch.Tensor) and result.numel() == 0
        ):
            return result

        return None

    def _cleanup_stale_transfer_locks(self, max_age_s=5):
        """Release transfer locks older than max_age_s.

        Transferred blocks are locked (inc_lock_ref) to prevent eviction
        between transfer and the request that will use them. This cleanup
        ensures locks are eventually released even if the request never
        arrives (e.g., client disconnect, routing change).
        """
        if not hasattr(self, "_transfer_pending_locks"):
            return
        now = time.time()
        expired = [
            k
            for k, (node, ts) in self._transfer_pending_locks.items()
            if now - ts > max_age_s
        ]
        for k in expired:
            node, _ts = self._transfer_pending_locks.pop(k)
            try:
                self.tree_cache.dec_lock_ref(node)
            except Exception as e:
                logger.warning(f"Failed to release stale transfer lock: {e}")
        if expired:
            logger.info(f"Released {len(expired)} stale transfer lock(s)")

    # Install methods on the scheduler class
    scheduler_class.init_kv_transfer = init_kv_transfer
    scheduler_class.get_kv_transfer_metadata = get_kv_transfer_metadata
    scheduler_class.query_kv_blocks_by_tokens = query_kv_blocks_by_tokens
    scheduler_class.receive_kv_blocks = receive_kv_blocks
    scheduler_class._resolve_source_metadata = _resolve_source_metadata
    scheduler_class._insert_transferred_blocks = _insert_transferred_blocks
    scheduler_class._execute_nixl_transfer = _execute_nixl_transfer
    scheduler_class._start_nixl_transfer = _start_nixl_transfer
    scheduler_class.process_kv_transfers = process_kv_transfers
    scheduler_class._complete_kv_transfer = _complete_kv_transfer
    scheduler_class._fail_kv_transfer = _fail_kv_transfer
    scheduler_class._handle_kv_transfer_request = _handle_kv_transfer_request
    scheduler_class._write_completion_error = _write_completion_error
    scheduler_class._alloc_with_eviction = _alloc_with_eviction
    scheduler_class._cleanup_stale_transfer_locks = _cleanup_stale_transfer_locks

    # Wrap process_input_requests to poll async transfers every iteration.
    # This mirrors SGLang's disaggregated decode pattern where
    # process_decode_queue() is called each iteration after
    # process_input_requests() in event_loop_normal_disagg_decode.
    _original_process_input = scheduler_class.process_input_requests

    def _process_input_requests_with_kv_poll(self, recv_reqs):
        _original_process_input(self, recv_reqs)
        # Always call process_kv_transfers — it handles both draining
        # new requests from the ZMQ socket and polling pending RDMAs.
        if hasattr(self, "_kv_transfer_enabled"):
            self.process_kv_transfers()

    scheduler_class.process_input_requests = _process_input_requests_with_kv_poll

    # Wrap self_check_during_idle to skip check_memory when there are
    # in-flight RDMA transfers. Pending transfers have pages allocated
    # from the KV pool but not yet inserted into the radix tree, so
    # check_memory sees them as "leaked" and crashes the scheduler.
    _original_self_check = scheduler_class.self_check_during_idle

    def _self_check_with_transfer_guard(self):
        if hasattr(self, "_pending_kv_transfers") and self._pending_kv_transfers:
            # Poll transfers instead of idle-checking — pages are in flight
            self.process_kv_transfers()
            return
        if hasattr(self, "_transfer_pending_locks") and self._transfer_pending_locks:
            # Release stale locks before check_memory sees them as leaked
            self._cleanup_stale_transfer_locks(max_age_s=0)
        _original_self_check(self)

    scheduler_class.self_check_during_idle = _self_check_with_transfer_guard
