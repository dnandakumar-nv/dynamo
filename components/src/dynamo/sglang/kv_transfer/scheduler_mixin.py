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

Non-blocking transfers: The initiate_kv_transfer + process_kv_transfers
pattern splits RDMA into a fast initiation phase (~10ms via collective_rpc)
and a non-blocking completion phase (polled each scheduler iteration via
process_kv_transfers). This avoids blocking the scheduler for the 500-800ms
duration of each RDMA transfer.
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

    def init_kv_transfer(self, page_size=16):
        """Initialize NIXL transfer infrastructure.

        Registers KV cache GPU memory with NIXL agent for remote reads.
        Called via Engine.collective_rpc("init_kv_transfer").
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
        # Populated by initiate_kv_transfer, drained by process_kv_transfers.
        self._pending_kv_transfers = {}

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

    def _resolve_source_metadata(self, source_worker_id, source_metadata_per_rank):
        """Resolve source metadata from parameters or cache, add NIXL peer.

        Returns (source_metadata, peer_name) for this TP rank.
        Shared by both the synchronous (receive_kv_blocks) and asynchronous
        (initiate_kv_transfer) paths.
        """
        import base64

        tp_rank = getattr(self, "tp_rank", 0)

        if source_metadata_per_rank is not None:
            source_metadata = source_metadata_per_rank[tp_rank]
            if source_worker_id is not None:
                self._nixl_source_metadata_cache[source_worker_id] = source_metadata
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

        self._cleanup_stale_transfer_locks()

        page_size = self._kv_transfer_page_size
        need_tokens = num_blocks * page_size

        import torch

        local_indices = self.token_to_kv_pool_allocator.alloc(need_tokens)
        if local_indices is None or (
            isinstance(local_indices, torch.Tensor) and local_indices.numel() == 0
        ):
            raise RuntimeError("Could not allocate local KV pages for transfer")

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
            logger.info(
                f"NIXL timing: {num_descs} descs, "
                f"prep={(t_prep - t0) / 1e6:.1f}ms "
                f"rdma={(t_done - t_prep) / 1e6:.1f}ms"
            )

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

    def initiate_kv_transfer(
        self,
        source_kv_indices,
        token_ids,
        num_blocks,
        timeout_ms=5000,
        source_worker_id=None,
        source_metadata_per_rank=None,
        completion_file=None,
    ):
        """Initiate a non-blocking KV transfer (Phase 1: fast start).

        Allocates local pages, builds NIXL descriptors, starts RDMA, and
        returns immediately. The transfer completes asynchronously — polled
        by process_kv_transfers() in the scheduler event loop.

        Writes completion status to completion_file.rank{tp_rank} when RDMA
        finishes (via process_kv_transfers).

        Called via Engine.collective_rpc("initiate_kv_transfer", ...).
        """
        if not getattr(self, "_kv_transfer_enabled", False):
            raise RuntimeError("KV transfer not initialized")

        source_metadata, peer_name = self._resolve_source_metadata(
            source_worker_id, source_metadata_per_rank
        )

        self._cleanup_stale_transfer_locks()

        page_size = self._kv_transfer_page_size
        need_tokens = num_blocks * page_size

        import torch

        local_indices = self.token_to_kv_pool_allocator.alloc(need_tokens)
        if local_indices is None or (
            isinstance(local_indices, torch.Tensor) and local_indices.numel() == 0
        ):
            raise RuntimeError("Could not allocate local KV pages for transfer")

        try:
            xfer_handle, local_prep, remote_prep, prep_ms = (
                self._start_nixl_transfer(
                    source_metadata, source_kv_indices, local_indices,
                    num_blocks, page_size, peer_name,
                )
            )
        except Exception:
            try:
                self.token_to_kv_pool_allocator.free(local_indices)
            except Exception:
                pass
            raise

        tp_rank = getattr(self, "tp_rank", 0)
        transfer_id = f"kv_xfer_{time.time_ns()}_{tp_rank}"

        if not hasattr(self, "_pending_kv_transfers"):
            self._pending_kv_transfers = {}

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
            f"initiate_kv_transfer: {transfer_id} started, "
            f"{num_blocks} blocks, prep={prep_ms:.1f}ms"
        )

    def process_kv_transfers(self):
        """Poll pending async KV transfers (Phase 2: non-blocking completion).

        Called every scheduler iteration from the wrapped process_input_requests.
        For each pending transfer:
        - DONE: release handles, insert into radix tree, write completion file
        - ERR: release handles, free pages, write error to completion file
        - PROC: skip (check again next iteration)
        - Timeout: treat as error
        """
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

        # Free allocated pages
        try:
            self.token_to_kv_pool_allocator.free(state["local_indices"])
        except Exception:
            pass

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

    def _cleanup_stale_transfer_locks(self, max_age_s=30):
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
    scheduler_class.initiate_kv_transfer = initiate_kv_transfer
    scheduler_class.process_kv_transfers = process_kv_transfers
    scheduler_class._complete_kv_transfer = _complete_kv_transfer
    scheduler_class._fail_kv_transfer = _fail_kv_transfer
    scheduler_class._cleanup_stale_transfer_locks = _cleanup_stale_transfer_locks

    # Wrap process_input_requests to poll async transfers every iteration.
    # This mirrors SGLang's disaggregated decode pattern where
    # process_decode_queue() is called each iteration after
    # process_input_requests() in event_loop_normal_disagg_decode.
    _original_process_input = scheduler_class.process_input_requests

    def _process_input_requests_with_kv_poll(self, recv_reqs):
        _original_process_input(self, recv_reqs)
        if hasattr(self, "_pending_kv_transfers"):
            self.process_kv_transfers()

    scheduler_class.process_input_requests = _process_input_requests_with_kv_poll
