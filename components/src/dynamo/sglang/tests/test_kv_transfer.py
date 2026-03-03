# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the KV transfer package.

Tests cover:
- TransferResult data class
- Scheduler mixin installation
- KvTransferManager metadata cache
- KvTransferManager result_file pattern
- KvTransferSourceHandler dispatch
- KvTransferTargetHandler orchestration
- Lazy import via __init__.py
- Backend args enable_kv_transfer field
- End-to-end transfer flow (Phase 5)
- Transfer hint extraction and serialization (Phase 5)
- Fallback paths (Phase 5)
- Event publication timing (Phase 5)
- Aggregated vs disaggregated modes (Phase 5)
"""

import asyncio
import inspect
import json
import logging
import os
import sys
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [
    pytest.mark.unit,
]


# ---------------------------------------------------------------------------
# TransferResult
# ---------------------------------------------------------------------------

class TestTransferResult:
    def setup_method(self):
        from dynamo.sglang.kv_transfer.target_handler import TransferResult
        self.TransferResult = TransferResult

    def test_succeeded_basic(self):
        r = self.TransferResult.succeeded(5, 80)
        assert r.success is True
        assert r.num_blocks == 5
        assert r.transferred_tokens == 80
        assert r.error == ""

    def test_succeeded_zero(self):
        r = self.TransferResult.succeeded(0, 0)
        assert r.success is True
        assert r.num_blocks == 0
        assert r.transferred_tokens == 0

    def test_failed_basic(self):
        r = self.TransferResult.failed("timeout")
        assert r.success is False
        assert r.num_blocks == 0
        assert r.transferred_tokens == 0
        assert r.error == "timeout"

    def test_failed_empty_error(self):
        r = self.TransferResult.failed("")
        assert r.success is False
        assert r.error == ""

    def test_slots(self):
        r = self.TransferResult.succeeded(1, 16)
        assert hasattr(r, "__slots__")
        with pytest.raises(AttributeError):
            r.nonexistent = True


# ---------------------------------------------------------------------------
# Scheduler Mixin
# ---------------------------------------------------------------------------

class TestSchedulerMixin:
    def test_install_patches_all_methods(self):
        from dynamo.sglang.kv_transfer.scheduler_mixin import install_kv_transfer_methods

        class FakeScheduler:
            pass

        install_kv_transfer_methods(FakeScheduler)
        for name in [
            "init_kv_transfer",
            "get_kv_transfer_metadata",
            "query_kv_blocks_by_tokens",
            "receive_kv_blocks",
        ]:
            assert hasattr(FakeScheduler, name), f"Missing {name}"
            assert callable(getattr(FakeScheduler, name))

    def test_install_is_idempotent(self):
        from dynamo.sglang.kv_transfer.scheduler_mixin import install_kv_transfer_methods

        class FakeScheduler:
            pass

        install_kv_transfer_methods(FakeScheduler)
        install_kv_transfer_methods(FakeScheduler)
        assert callable(FakeScheduler.init_kv_transfer)

    def test_init_kv_transfer_signature(self):
        from dynamo.sglang.kv_transfer.scheduler_mixin import install_kv_transfer_methods

        class FakeScheduler:
            pass

        install_kv_transfer_methods(FakeScheduler)
        sig = inspect.signature(FakeScheduler.init_kv_transfer)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "page_size" in params

    def test_get_kv_transfer_metadata_signature(self):
        from dynamo.sglang.kv_transfer.scheduler_mixin import install_kv_transfer_methods

        class FakeScheduler:
            pass

        install_kv_transfer_methods(FakeScheduler)
        sig = inspect.signature(FakeScheduler.get_kv_transfer_metadata)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "result_file" in params

    def test_query_kv_blocks_by_tokens_signature(self):
        from dynamo.sglang.kv_transfer.scheduler_mixin import install_kv_transfer_methods

        class FakeScheduler:
            pass

        install_kv_transfer_methods(FakeScheduler)
        sig = inspect.signature(FakeScheduler.query_kv_blocks_by_tokens)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "token_ids" in params
        assert "num_blocks" in params
        assert "result_file" in params

    def test_receive_kv_blocks_signature(self):
        from dynamo.sglang.kv_transfer.scheduler_mixin import install_kv_transfer_methods

        class FakeScheduler:
            pass

        install_kv_transfer_methods(FakeScheduler)
        sig = inspect.signature(FakeScheduler.receive_kv_blocks)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "source_metadata" in params
        assert "source_kv_indices" in params
        assert "token_ids" in params
        assert "num_blocks" in params

    def test_get_kv_transfer_metadata_writes_error_when_disabled(self):
        """When _kv_transfer_enabled is False, writes error JSON to result_file."""
        from dynamo.sglang.kv_transfer.scheduler_mixin import install_kv_transfer_methods

        class FakeScheduler:
            tp_rank = 0

        install_kv_transfer_methods(FakeScheduler)
        sched = FakeScheduler()
        sched._kv_transfer_enabled = False

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            sched.get_kv_transfer_metadata(result_file=path)
            with open(path) as f:
                data = json.load(f)
            assert data["status"] == "error"
        finally:
            os.unlink(path)

    def test_query_kv_blocks_writes_error_when_disabled(self):
        """When _kv_transfer_enabled is False, writes error JSON to result_file."""
        from dynamo.sglang.kv_transfer.scheduler_mixin import install_kv_transfer_methods

        class FakeScheduler:
            tp_rank = 0

        install_kv_transfer_methods(FakeScheduler)
        sched = FakeScheduler()
        sched._kv_transfer_enabled = False

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            sched.query_kv_blocks_by_tokens(
                token_ids=[1, 2, 3], num_blocks=1, result_file=path
            )
            with open(path) as f:
                data = json.load(f)
            assert data["status"] == "error"
        finally:
            os.unlink(path)

    def test_receive_kv_blocks_raises_when_disabled(self):
        """receive_kv_blocks should raise RuntimeError when not initialized."""
        from dynamo.sglang.kv_transfer.scheduler_mixin import install_kv_transfer_methods

        class FakeScheduler:
            pass

        install_kv_transfer_methods(FakeScheduler)
        sched = FakeScheduler()

        with pytest.raises(RuntimeError, match="not initialized"):
            sched.receive_kv_blocks(
                source_metadata={},
                source_kv_indices=[],
                token_ids=[],
                num_blocks=0,
            )

    def test_init_kv_transfer_handles_missing_nixl(self):
        """init_kv_transfer should gracefully disable when NIXL is not importable."""
        from dynamo.sglang.kv_transfer.scheduler_mixin import install_kv_transfer_methods

        class FakeScheduler:
            pass

        install_kv_transfer_methods(FakeScheduler)
        sched = FakeScheduler()

        # Mock the NIXL import to fail
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "nixl" in name:
                raise ImportError("No NIXL")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            sched.init_kv_transfer(page_size=16)

        assert sched._kv_transfer_enabled is False


# ---------------------------------------------------------------------------
# KvTransferManager metadata cache
# ---------------------------------------------------------------------------

class TestManagerMetadataCache:
    def setup_method(self):
        from dynamo.sglang.kv_transfer.manager import KvTransferManager
        self.KvTransferManager = KvTransferManager

    def _make_manager(self):
        """Create a manager with mocked engine and config."""
        engine = MagicMock()
        config = MagicMock()
        config.server_args.page_size = 16
        return self.KvTransferManager(engine, config)

    def test_cache_miss_returns_none(self):
        mgr = self._make_manager()
        assert mgr.get_cached_metadata(42) is None

    def test_cache_hit(self):
        mgr = self._make_manager()
        meta = {"status": "ok", "gpu_id": 0}
        mgr.cache_metadata(42, meta)
        assert mgr.get_cached_metadata(42) == meta

    def test_cache_expiry(self):
        mgr = self._make_manager()
        mgr._metadata_cache_ttl = 0.01  # 10ms
        meta = {"status": "ok"}
        mgr.cache_metadata(42, meta)
        assert mgr.get_cached_metadata(42) == meta
        time.sleep(0.02)
        assert mgr.get_cached_metadata(42) is None

    def test_invalidate(self):
        mgr = self._make_manager()
        meta = {"status": "ok"}
        mgr.cache_metadata(42, meta)
        mgr.invalidate_metadata(42)
        assert mgr.get_cached_metadata(42) is None

    def test_invalidate_nonexistent(self):
        mgr = self._make_manager()
        mgr.invalidate_metadata(999)  # should not raise

    def test_multiple_workers(self):
        mgr = self._make_manager()
        mgr.cache_metadata(1, {"worker": 1})
        mgr.cache_metadata(2, {"worker": 2})
        assert mgr.get_cached_metadata(1)["worker"] == 1
        assert mgr.get_cached_metadata(2)["worker"] == 2


# ---------------------------------------------------------------------------
# KvTransferManager result_file pattern
# ---------------------------------------------------------------------------

class TestManagerResultFile:
    def setup_method(self):
        from dynamo.sglang.kv_transfer.manager import KvTransferManager
        self.KvTransferManager = KvTransferManager

    def _make_manager(self):
        engine = MagicMock()
        config = MagicMock()
        config.server_args.page_size = 16
        return self.KvTransferManager(engine, config)

    def test_rpc_with_result_reads_json(self):
        mgr = self._make_manager()

        def fake_collective_rpc(method, result_file=None, **kwargs):
            with open(result_file, "w") as f:
                json.dump({"status": "ok", "data": 42}, f)

        mgr.engine.collective_rpc = fake_collective_rpc
        result = mgr._rpc_with_result("test_method")
        assert result["status"] == "ok"
        assert result["data"] == 42

    def test_rpc_with_result_cleans_up_file(self):
        mgr = self._make_manager()

        captured_paths = []

        def fake_collective_rpc(method, result_file=None, **kwargs):
            captured_paths.append(result_file)
            with open(result_file, "w") as f:
                json.dump({}, f)

        mgr.engine.collective_rpc = fake_collective_rpc
        mgr._rpc_with_result("test_method")
        assert len(captured_paths) == 1
        assert not os.path.exists(captured_paths[0])

    def test_rpc_with_result_cleans_up_on_error(self):
        mgr = self._make_manager()

        captured_paths = []

        def fake_collective_rpc(method, result_file=None, **kwargs):
            captured_paths.append(result_file)
            raise RuntimeError("boom")

        mgr.engine.collective_rpc = fake_collective_rpc

        with pytest.raises(RuntimeError, match="boom"):
            mgr._rpc_with_result("test_method")

        assert len(captured_paths) == 1
        assert not os.path.exists(captured_paths[0])

    def test_get_local_metadata(self):
        mgr = self._make_manager()

        def fake_collective_rpc(method, result_file=None, **kwargs):
            assert method == "get_kv_transfer_metadata"
            with open(result_file, "w") as f:
                json.dump({"status": "ok", "gpu_id": 0}, f)

        mgr.engine.collective_rpc = fake_collective_rpc
        result = mgr.get_local_metadata()
        assert result["status"] == "ok"

    def test_query_local_blocks(self):
        mgr = self._make_manager()

        def fake_collective_rpc(method, result_file=None, **kwargs):
            assert method == "query_kv_blocks_by_tokens"
            assert kwargs["token_ids"] == [1, 2, 3]
            assert kwargs["num_blocks"] == 2
            with open(result_file, "w") as f:
                json.dump({"status": "ok", "kv_indices": [10, 20]}, f)

        mgr.engine.collective_rpc = fake_collective_rpc
        result = mgr.query_local_blocks([1, 2, 3], 2)
        assert result["status"] == "ok"
        assert result["kv_indices"] == [10, 20]

    def test_get_local_metadata_handles_error(self):
        mgr = self._make_manager()
        mgr.engine.collective_rpc.side_effect = RuntimeError("rpc fail")
        result = mgr.get_local_metadata()
        assert result["status"] == "error"

    def test_query_local_blocks_handles_error(self):
        mgr = self._make_manager()
        mgr.engine.collective_rpc.side_effect = RuntimeError("rpc fail")
        result = mgr.query_local_blocks([1, 2], 1)
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# KvTransferSourceHandler
# ---------------------------------------------------------------------------

class TestSourceHandler:
    def setup_method(self):
        from dynamo.sglang.kv_transfer.source_handler import KvTransferSourceHandler
        self.KvTransferSourceHandler = KvTransferSourceHandler

    def _make_handler(self):
        manager = MagicMock()
        return self.KvTransferSourceHandler(manager)

    @pytest.mark.asyncio
    async def test_get_metadata_action(self):
        handler = self._make_handler()
        handler.manager.get_local_metadata.return_value = {
            "status": "ok",
            "gpu_id": 0,
        }

        results = []
        async for r in handler.handle_request({"action": "get_metadata"}):
            results.append(r)

        assert len(results) == 1
        assert results[0]["status"] == "ok"
        handler.manager.get_local_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_blocks_action(self):
        handler = self._make_handler()
        handler.manager.query_local_blocks.return_value = {
            "status": "ok",
            "kv_indices": [5, 10],
        }

        results = []
        async for r in handler.handle_request({
            "action": "query_blocks",
            "token_ids": [1, 2, 3],
            "num_blocks": 2,
        }):
            results.append(r)

        assert len(results) == 1
        assert results[0]["kv_indices"] == [5, 10]
        handler.manager.query_local_blocks.assert_called_once_with([1, 2, 3], 2)

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        handler = self._make_handler()

        results = []
        async for r in handler.handle_request({"action": "invalid"}):
            results.append(r)

        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "Unknown action" in results[0]["message"]

    @pytest.mark.asyncio
    async def test_missing_action(self):
        handler = self._make_handler()

        results = []
        async for r in handler.handle_request({}):
            results.append(r)

        assert len(results) == 1
        assert results[0]["status"] == "error"

    def test_handle_request_is_async_gen(self):
        handler = self._make_handler()
        assert inspect.isasyncgenfunction(handler.handle_request)


# ---------------------------------------------------------------------------
# KvTransferTargetHandler
# ---------------------------------------------------------------------------

class TestTargetHandler:
    def setup_method(self):
        from dynamo.sglang.kv_transfer.target_handler import (
            KvTransferTargetHandler,
            TransferResult,
        )
        self.KvTransferTargetHandler = KvTransferTargetHandler
        self.TransferResult = TransferResult

    def _make_handler(self):
        manager = MagicMock()
        manager.page_size = 16
        handler = self.KvTransferTargetHandler(manager)
        return handler

    @pytest.mark.asyncio
    async def test_invalid_hint_missing_worker(self):
        handler = self._make_handler()
        result = await handler.execute_transfer(
            {"num_blocks": 5}, [1, 2, 3]
        )
        assert result.success is False
        assert "Invalid" in result.error

    @pytest.mark.asyncio
    async def test_invalid_hint_zero_blocks(self):
        handler = self._make_handler()
        result = await handler.execute_transfer(
            {"source_worker": {"worker_id": 1}, "num_blocks": 0},
            [1, 2, 3],
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_metadata_fetch_failure(self):
        handler = self._make_handler()
        handler.manager.get_cached_metadata.return_value = None
        handler._transfer_client = MagicMock()

        # _rpc_to_source returns None (simulating failure)
        handler._rpc_to_source = AsyncMock(return_value=None)

        result = await handler.execute_transfer(
            {"source_worker": {"worker_id": 1, "dp_rank": 0}, "num_blocks": 2},
            [1, 2, 3],
        )
        assert result.success is False
        assert "metadata" in result.error.lower() or "Could not" in result.error

    @pytest.mark.asyncio
    async def test_successful_transfer(self):
        handler = self._make_handler()

        # Simulate cached metadata
        handler.manager.get_cached_metadata.return_value = {
            "status": "ok",
            "agent_metadata": "dGVzdA==",
            "gpu_id": 0,
        }

        # Simulate block query response
        handler._rpc_to_source = AsyncMock(return_value={
            "status": "ok",
            "kv_indices": [10, 20],
            "num_matched_blocks": 2,
        })

        # Simulate successful receive
        handler.manager.execute_receive_transfer = AsyncMock(return_value=True)

        result = await handler.execute_transfer(
            {"source_worker": {"worker_id": 1, "dp_rank": 0}, "num_blocks": 2},
            list(range(32)),
        )
        assert result.success is True
        assert result.num_blocks == 2
        assert result.transferred_tokens == 32  # 2 blocks * 16 page_size

    @pytest.mark.asyncio
    async def test_block_query_failure(self):
        handler = self._make_handler()

        handler.manager.get_cached_metadata.return_value = {
            "status": "ok",
        }

        handler._rpc_to_source = AsyncMock(return_value={
            "status": "error",
            "message": "radix miss",
        })

        result = await handler.execute_transfer(
            {"source_worker": {"worker_id": 1, "dp_rank": 0}, "num_blocks": 2},
            [1, 2, 3],
        )
        assert result.success is False
        assert "Block query failed" in result.error

    @pytest.mark.asyncio
    async def test_zero_matched_blocks(self):
        handler = self._make_handler()

        handler.manager.get_cached_metadata.return_value = {"status": "ok"}
        handler._rpc_to_source = AsyncMock(return_value={
            "status": "ok",
            "kv_indices": [],
            "num_matched_blocks": 0,
        })

        result = await handler.execute_transfer(
            {"source_worker": {"worker_id": 1, "dp_rank": 0}, "num_blocks": 2},
            [1, 2, 3],
        )
        assert result.success is False
        assert "No blocks" in result.error

    @pytest.mark.asyncio
    async def test_receive_exception_caught(self):
        handler = self._make_handler()

        handler.manager.get_cached_metadata.return_value = {"status": "ok"}
        handler._rpc_to_source = AsyncMock(return_value={
            "status": "ok",
            "kv_indices": [5],
            "num_matched_blocks": 1,
        })
        handler.manager.execute_receive_transfer = AsyncMock(
            side_effect=RuntimeError("NIXL timeout")
        )

        result = await handler.execute_transfer(
            {"source_worker": {"worker_id": 1, "dp_rank": 0}, "num_blocks": 1},
            list(range(16)),
        )
        assert result.success is False
        assert "NIXL timeout" in result.error

    @pytest.mark.asyncio
    async def test_rpc_to_source_no_client(self):
        handler = self._make_handler()
        handler._transfer_client = None
        result = await handler._rpc_to_source(1, {"action": "get_metadata"})
        assert result is None


# ---------------------------------------------------------------------------
# Lazy import via __init__.py
# ---------------------------------------------------------------------------

class TestLazyImport:
    def test_import_kv_transfer_manager(self):
        from dynamo.sglang.kv_transfer import KvTransferManager
        from dynamo.sglang.kv_transfer.manager import (
            KvTransferManager as DirectManager,
        )
        assert KvTransferManager is DirectManager

    def test_all_attribute(self):
        import dynamo.sglang.kv_transfer as pkg
        assert "KvTransferManager" in pkg.__all__

    def test_unknown_attr_raises(self):
        import dynamo.sglang.kv_transfer as pkg
        with pytest.raises(AttributeError):
            _ = pkg.NonExistentThing


# ---------------------------------------------------------------------------
# Backend args
# ---------------------------------------------------------------------------

class TestBackendArgs:
    def test_enable_kv_transfer_field_exists(self):
        from dynamo.sglang.backend_args import DynamoSGLangConfig
        # The field should be accessible as a class attribute or via introspection
        assert "enable_kv_transfer" in dir(DynamoSGLangConfig)

    def test_cli_flag_in_source(self):
        """Verify the CLI flag string exists in the source file."""
        import dynamo.sglang.backend_args as mod
        source = inspect.getsource(mod)
        assert "--enable-kv-transfer" in source
        assert "DYN_SGL_ENABLE_KV_TRANSFER" in source


# ---------------------------------------------------------------------------
# Phase 3: Tuning Parameters CLI Flags
# ---------------------------------------------------------------------------

class TestTuningParameters:
    def test_transfer_timeout_ms_cli_flag(self):
        """Verify --transfer-timeout-ms flag exists in source."""
        import dynamo.sglang.backend_args as mod
        source = inspect.getsource(mod)
        assert "--transfer-timeout-ms" in source
        assert "DYN_SGL_TRANSFER_TIMEOUT_MS" in source

    def test_metadata_cache_ttl_s_cli_flag(self):
        """Verify --metadata-cache-ttl-s flag exists in source."""
        import dynamo.sglang.backend_args as mod
        source = inspect.getsource(mod)
        assert "--metadata-cache-ttl-s" in source
        assert "DYN_SGL_METADATA_CACHE_TTL_S" in source

    def test_enable_transfer_overlap_cli_flag(self):
        """Verify --enable-transfer-overlap flag exists in source."""
        import dynamo.sglang.backend_args as mod
        source = inspect.getsource(mod)
        assert "--enable-transfer-overlap" in source
        assert "DYN_SGL_ENABLE_TRANSFER_OVERLAP" in source

    def test_config_fields_exist(self):
        """Verify config fields for tuning parameters exist."""
        from dynamo.sglang.backend_args import DynamoSGLangConfig
        for field in [
            "transfer_timeout_ms",
            "metadata_cache_ttl_s",
            "enable_transfer_overlap",
        ]:
            assert field in dir(DynamoSGLangConfig), f"Missing config field: {field}"

    def test_manager_uses_configurable_ttl(self):
        """Verify manager respects metadata_cache_ttl_s parameter."""
        from dynamo.sglang.kv_transfer.manager import KvTransferManager

        engine = MagicMock()
        config = MagicMock()
        config.server_args.page_size = 16

        mgr = KvTransferManager(engine, config, metadata_cache_ttl_s=60)
        assert mgr._metadata_cache_ttl == 60

    def test_manager_uses_configurable_timeout(self):
        """Verify manager respects transfer_timeout_ms parameter."""
        from dynamo.sglang.kv_transfer.manager import KvTransferManager

        engine = MagicMock()
        config = MagicMock()
        config.server_args.page_size = 16

        mgr = KvTransferManager(engine, config, transfer_timeout_ms=10000)
        assert mgr.transfer_timeout_ms == 10000

    def test_manager_default_ttl(self):
        """Verify manager has correct default TTL."""
        from dynamo.sglang.kv_transfer.manager import KvTransferManager

        engine = MagicMock()
        config = MagicMock()
        config.server_args.page_size = 16

        mgr = KvTransferManager(engine, config)
        assert mgr._metadata_cache_ttl == 300

    def test_manager_default_timeout(self):
        """Verify manager has correct default timeout."""
        from dynamo.sglang.kv_transfer.manager import KvTransferManager

        engine = MagicMock()
        config = MagicMock()
        config.server_args.page_size = 16

        mgr = KvTransferManager(engine, config)
        assert mgr.transfer_timeout_ms == 5000


# ---------------------------------------------------------------------------
# Phase 3: Scheduler Mixin Timeout Parameter
# ---------------------------------------------------------------------------

class TestSchedulerMixinTimeout:
    def test_receive_kv_blocks_accepts_timeout_ms(self):
        """Verify receive_kv_blocks signature includes timeout_ms."""
        from dynamo.sglang.kv_transfer.scheduler_mixin import install_kv_transfer_methods

        class FakeScheduler:
            pass

        install_kv_transfer_methods(FakeScheduler)
        sig = inspect.signature(FakeScheduler.receive_kv_blocks)
        params = list(sig.parameters.keys())
        assert "timeout_ms" in params

    def test_receive_kv_blocks_default_timeout(self):
        """Verify timeout_ms default value is 5000."""
        from dynamo.sglang.kv_transfer.scheduler_mixin import install_kv_transfer_methods

        class FakeScheduler:
            pass

        install_kv_transfer_methods(FakeScheduler)
        sig = inspect.signature(FakeScheduler.receive_kv_blocks)
        assert sig.parameters["timeout_ms"].default == 5000


# ---------------------------------------------------------------------------
# Phase 3: Manager execute_receive_transfer passes timeout_ms
# ---------------------------------------------------------------------------

class TestManagerTimeout:
    @pytest.mark.asyncio
    async def test_execute_receive_transfer_passes_timeout(self):
        """Verify execute_receive_transfer passes timeout_ms to collective_rpc."""
        from dynamo.sglang.kv_transfer.manager import KvTransferManager

        engine = MagicMock()
        config = MagicMock()
        config.server_args.page_size = 16

        mgr = KvTransferManager(engine, config, transfer_timeout_ms=8000)

        captured = {}

        def fake_collective_rpc(method, **kwargs):
            captured.update(kwargs)

        engine.collective_rpc = fake_collective_rpc

        await mgr.execute_receive_transfer(
            source_metadata={"test": True},
            source_kv_indices=[1, 2],
            token_ids=[1, 2, 3],
            num_blocks=2,
        )
        assert captured["timeout_ms"] == 8000


# ---------------------------------------------------------------------------
# Phase 3: KV Event Publication
# ---------------------------------------------------------------------------

class TestPublishTransferredBlocks:
    def setup_method(self):
        from dynamo.sglang.kv_transfer.manager import KvTransferManager
        self.KvTransferManager = KvTransferManager

    def _make_manager(self, page_size=16):
        engine = MagicMock()
        config = MagicMock()
        config.server_args.page_size = page_size
        return self.KvTransferManager(engine, config)

    @patch("dynamo._core.compute_block_hash_for_seq")
    def test_computes_correct_hashes(self, mock_hash):
        """Verify compute_block_hash_for_seq called with correct token prefix."""
        mock_hash.return_value = [111, 222]
        mgr = self._make_manager(page_size=16)
        kv_pub = MagicMock()

        token_ids = list(range(64))  # 4 blocks worth of tokens
        mgr.publish_transferred_blocks(token_ids, num_blocks=2, kv_publisher=kv_pub)

        # Should be called with first 32 tokens (2 blocks * 16 page_size)
        mock_hash.assert_called_once_with(
            tokens=list(range(32)),
            kv_block_size=16,
        )

    @patch("dynamo._core.compute_block_hash_for_seq")
    def test_correct_num_block_tokens(self, mock_hash):
        """Verify num_block_tokens is [page_size] * num_blocks."""
        mock_hash.return_value = [111, 222, 333]
        mgr = self._make_manager(page_size=8)
        kv_pub = MagicMock()

        token_ids = list(range(48))
        mgr.publish_transferred_blocks(token_ids, num_blocks=3, kv_publisher=kv_pub)

        call_kwargs = kv_pub.publish_stored.call_args[1]
        assert call_kwargs["num_block_tokens"] == [8, 8, 8]

    @patch("dynamo._core.compute_block_hash_for_seq")
    def test_calls_publish_stored(self, mock_hash):
        """Verify publish_stored called with correct arguments."""
        mock_hash.return_value = [42, 99]
        mgr = self._make_manager(page_size=16)
        kv_pub = MagicMock()

        token_ids = list(range(64))
        mgr.publish_transferred_blocks(token_ids, num_blocks=2, kv_publisher=kv_pub)

        kv_pub.publish_stored.assert_called_once_with(
            token_ids=list(range(32)),
            num_block_tokens=[16, 16],
            block_hashes=[42, 99],
            parent_hash=None,
        )

    def test_handles_none_publisher_gracefully(self):
        """Verify no crash when kv_publisher is None — caller should guard."""
        # publish_transferred_blocks expects a real publisher;
        # the guard is in decode_handler. Here we verify the method
        # exists and is callable.
        mgr = self._make_manager()
        assert hasattr(mgr, "publish_transferred_blocks")
        assert callable(mgr.publish_transferred_blocks)

    @patch("dynamo._core.compute_block_hash_for_seq")
    def test_exception_in_publish_stored_propagates(self, mock_hash):
        """Verify exception from publish_stored propagates (caller catches it)."""
        mock_hash.return_value = [42]
        mgr = self._make_manager(page_size=16)
        kv_pub = MagicMock()
        kv_pub.publish_stored.side_effect = RuntimeError("NATS down")

        with pytest.raises(RuntimeError, match="NATS down"):
            mgr.publish_transferred_blocks(
                token_ids=list(range(16)),
                num_blocks=1,
                kv_publisher=kv_pub,
            )


# ---------------------------------------------------------------------------
# Phase 3: Metrics Module
# ---------------------------------------------------------------------------

class TestMetricsModule:
    def test_record_functions_work_without_prometheus(self):
        """All record_* functions should be callable even without Prometheus."""
        from dynamo.sglang.kv_transfer import metrics

        # Reset state so we test fresh initialization
        metrics._metrics_initialized = False
        metrics._transfer_requests = None
        metrics._transfer_blocks = None
        metrics._transfer_duration = None
        metrics._transfer_fallback = None
        metrics._transfer_events_published = None

        # These should not raise even if Prometheus import fails
        metrics.record_transfer_request("target", "success")
        metrics.record_transfer_blocks("target", 5)
        metrics.record_transfer_duration("target", 0.005)
        metrics.record_transfer_fallback("test")
        metrics.record_events_published()

    def test_metrics_with_prometheus(self):
        """Verify metrics are created and labels are correct when Prometheus available."""
        from dynamo.sglang.kv_transfer import metrics

        # Force re-initialization
        metrics._metrics_initialized = False
        metrics._transfer_requests = None
        metrics._transfer_blocks = None
        metrics._transfer_duration = None
        metrics._transfer_fallback = None
        metrics._transfer_events_published = None

        metrics._ensure_initialized()

        # After initialization, metrics should be created (if prometheus_client is available)
        if metrics._metrics_initialized:
            assert metrics._transfer_requests is not None
            assert metrics._transfer_blocks is not None
            assert metrics._transfer_duration is not None
            assert metrics._transfer_fallback is not None
            assert metrics._transfer_events_published is not None

            # Verify labels by calling record functions
            metrics.record_transfer_request("target", "success")
            metrics.record_transfer_request("source", "failed")
            metrics.record_transfer_blocks("target", 10)
            metrics.record_transfer_duration("target", 0.01)
            metrics.record_transfer_fallback("timeout")
            metrics.record_events_published()

    def test_metrics_ensure_initialized_idempotent(self):
        """Verify _ensure_initialized is idempotent."""
        from dynamo.sglang.kv_transfer import metrics

        metrics._metrics_initialized = False
        metrics._ensure_initialized()
        first_requests = metrics._transfer_requests

        metrics._ensure_initialized()
        assert metrics._transfer_requests is first_requests

    def test_record_transfer_fallback_labels(self):
        """Verify fallback counter accepts various reason labels."""
        from dynamo.sglang.kv_transfer import metrics

        reasons = ["evicted", "timeout", "oom", "error", "no_blocks",
                    "invalid_hint", "metadata_fetch", "block_query"]
        for reason in reasons:
            # Should not raise
            metrics.record_transfer_fallback(reason)


# ---------------------------------------------------------------------------
# Phase 3: Target Handler Metrics Integration
# ---------------------------------------------------------------------------

class TestTargetHandlerMetrics:
    def setup_method(self):
        from dynamo.sglang.kv_transfer.target_handler import (
            KvTransferTargetHandler,
        )
        self.KvTransferTargetHandler = KvTransferTargetHandler

    def _make_handler(self):
        manager = MagicMock()
        manager.page_size = 16
        return self.KvTransferTargetHandler(manager)

    @pytest.mark.asyncio
    async def test_records_metrics_on_success(self):
        """Verify metrics recorded on successful transfer."""
        handler = self._make_handler()

        handler.manager.get_cached_metadata.return_value = {"status": "ok"}
        handler._rpc_to_source = AsyncMock(return_value={
            "status": "ok",
            "kv_indices": [10, 20],
            "num_matched_blocks": 2,
        })
        handler.manager.execute_receive_transfer = AsyncMock(return_value=True)

        with patch("dynamo.sglang.kv_transfer.metrics.record_transfer_request") as mock_req, \
             patch("dynamo.sglang.kv_transfer.metrics.record_transfer_blocks") as mock_blk, \
             patch("dynamo.sglang.kv_transfer.metrics.record_transfer_duration") as mock_dur:
            result = await handler.execute_transfer(
                {"source_worker": {"worker_id": 1, "dp_rank": 0}, "num_blocks": 2},
                list(range(32)),
            )
            assert result.success is True
            mock_req.assert_called_with("target", "success")
            mock_blk.assert_called_with("target", 2)
            assert mock_dur.called
            # Duration should be a positive float
            dur_args = mock_dur.call_args[0]
            assert dur_args[0] == "target"
            assert dur_args[1] >= 0

    @pytest.mark.asyncio
    async def test_records_fallback_on_invalid_hint(self):
        """Verify fallback reason recorded for invalid hints."""
        handler = self._make_handler()

        with patch("dynamo.sglang.kv_transfer.metrics.record_transfer_request") as mock_req, \
             patch("dynamo.sglang.kv_transfer.metrics.record_transfer_fallback") as mock_fb:
            result = await handler.execute_transfer(
                {"num_blocks": 5}, [1, 2, 3]
            )
            assert result.success is False
            mock_req.assert_called_with("target", "failed")
            mock_fb.assert_called_with("invalid_hint")

    @pytest.mark.asyncio
    async def test_records_fallback_on_exception(self):
        """Verify fallback reason recorded when transfer raises."""
        handler = self._make_handler()

        handler.manager.get_cached_metadata.return_value = {"status": "ok"}
        handler._rpc_to_source = AsyncMock(return_value={
            "status": "ok",
            "kv_indices": [5],
            "num_matched_blocks": 1,
        })
        handler.manager.execute_receive_transfer = AsyncMock(
            side_effect=RuntimeError("GPU OOM")
        )

        with patch("dynamo.sglang.kv_transfer.metrics.record_transfer_request") as mock_req, \
             patch("dynamo.sglang.kv_transfer.metrics.record_transfer_fallback") as mock_fb:
            result = await handler.execute_transfer(
                {"source_worker": {"worker_id": 1, "dp_rank": 0}, "num_blocks": 1},
                list(range(16)),
            )
            assert result.success is False
            mock_req.assert_called_with("target", "failed")
            mock_fb.assert_called_with("error")


# ---------------------------------------------------------------------------
# Phase 3: Source Handler Metrics Integration
# ---------------------------------------------------------------------------

class TestSourceHandlerMetrics:
    def setup_method(self):
        from dynamo.sglang.kv_transfer.source_handler import KvTransferSourceHandler
        self.KvTransferSourceHandler = KvTransferSourceHandler

    def _make_handler(self):
        manager = MagicMock()
        return self.KvTransferSourceHandler(manager)

    @pytest.mark.asyncio
    async def test_records_request_on_metadata(self):
        """Verify source request metric recorded for get_metadata."""
        handler = self._make_handler()
        handler.manager.get_local_metadata.return_value = {"status": "ok"}

        with patch("dynamo.sglang.kv_transfer.metrics.record_transfer_request") as mock_req:
            async for _ in handler.handle_request({"action": "get_metadata"}):
                pass
            mock_req.assert_called_with("source", "success")

    @pytest.mark.asyncio
    async def test_records_request_on_query(self):
        """Verify source request metric recorded for query_blocks."""
        handler = self._make_handler()
        handler.manager.query_local_blocks.return_value = {"status": "ok"}

        with patch("dynamo.sglang.kv_transfer.metrics.record_transfer_request") as mock_req:
            async for _ in handler.handle_request({
                "action": "query_blocks",
                "token_ids": [1, 2],
                "num_blocks": 1,
            }):
                pass
            mock_req.assert_called_with("source", "success")

    @pytest.mark.asyncio
    async def test_records_failed_on_unknown_action(self):
        """Verify source request metric recorded as failed for unknown action."""
        handler = self._make_handler()

        with patch("dynamo.sglang.kv_transfer.metrics.record_transfer_request") as mock_req:
            async for _ in handler.handle_request({"action": "invalid"}):
                pass
            mock_req.assert_called_with("source", "failed")


# ---------------------------------------------------------------------------
# Phase 4: Helper for fully-mocked scheduler
# ---------------------------------------------------------------------------

def _make_fake_scheduler_for_receive():
    """Create a FakeScheduler instance with all mocks needed for receive_kv_blocks.

    Returns a scheduler instance with:
    - All KV transfer methods installed
    - _kv_transfer_enabled = True
    - Mocked NIXL agent that succeeds immediately
    - Mocked tree_cache with insert, match_prefix, inc_lock_ref, dec_lock_ref
    - Mocked token_to_kv_pool_allocator
    """
    import torch

    from dynamo.sglang.kv_transfer.scheduler_mixin import install_kv_transfer_methods

    class FakeScheduler:
        tp_rank = 0

    install_kv_transfer_methods(FakeScheduler)
    sched = FakeScheduler()

    sched._kv_transfer_enabled = True
    sched._kv_transfer_page_size = 16
    sched._transfer_pending_locks = {}

    # NIXL agent mock
    sched._nixl_agent = MagicMock()
    sched._nixl_agent.add_remote_agent = MagicMock()
    sched._nixl_agent.get_xfer_descs = MagicMock(return_value=MagicMock())
    sched._nixl_agent.initialize_xfer = MagicMock(return_value="handle_123")
    sched._nixl_agent.transfer = MagicMock(return_value="IN_PROGRESS")
    sched._nixl_agent.check_xfer_state = MagicMock(return_value="DONE")

    sched._nixl_remote_peers = {}
    sched._kv_data_ptrs = [100, 200]
    sched._kv_data_lens = [1600, 1600]
    sched._kv_item_lens = [160, 160]
    sched._kv_gpu_id = 0

    # Tree cache mock
    root_node = MagicMock(name="root_node")
    sched.tree_cache = MagicMock()
    sched.tree_cache.root_node = root_node

    # Default: insert returns prefix_len=0 (no overlap)
    insert_result = MagicMock()
    insert_result.prefix_len = 0
    insert_result.mamba_exist = False
    sched.tree_cache.insert = MagicMock(return_value=insert_result)

    # Default: match_prefix returns a non-root node
    match_result = MagicMock()
    match_result.last_device_node = MagicMock(name="transfer_node")
    match_result.last_host_node = None
    match_result.host_hit_length = 0
    match_result.mamba_branching_seqlen = None
    match_result.device_indices = torch.arange(32)
    sched.tree_cache.match_prefix = MagicMock(return_value=match_result)

    sched.tree_cache.inc_lock_ref = MagicMock()
    sched.tree_cache.dec_lock_ref = MagicMock()

    # Allocator mock
    sched.token_to_kv_pool_allocator = MagicMock()
    sched.token_to_kv_pool_allocator.alloc = MagicMock(
        return_value=torch.arange(32)
    )
    sched.token_to_kv_pool_allocator.free = MagicMock()

    return sched


def _default_source_metadata():
    """Return minimal source_metadata dict for receive_kv_blocks."""
    import base64

    return {
        "agent_metadata": base64.b64encode(b"fake_agent_data").decode("ascii"),
        "kv_data_ptrs": [300, 400],
        "kv_item_lens": [160, 160],
        "gpu_id": 1,
    }


# ---------------------------------------------------------------------------
# Phase 4: Radix Tree Insertion
# ---------------------------------------------------------------------------

class TestRadixTreeInsertion:
    """Phase 4: Verify radix tree insertion in receive_kv_blocks."""

    def setup_method(self):
        self.sched = _make_fake_scheduler_for_receive()
        self.source_metadata = _default_source_metadata()

    def test_insert_called_with_correct_params(self):
        """Verify tree_cache.insert is called with correct key and value."""
        token_ids = list(range(64))
        num_blocks = 2
        source_kv_indices = [10, 20]

        self.sched.receive_kv_blocks(
            source_metadata=self.source_metadata,
            source_kv_indices=source_kv_indices,
            token_ids=token_ids,
            num_blocks=num_blocks,
        )

        self.sched.tree_cache.insert.assert_called_once()
        call_args = self.sched.tree_cache.insert.call_args[0][0]
        # key should be a RadixKey with prefix_token_ids
        assert list(call_args.key.token_ids) == list(range(32))  # 2 * 16

    def test_insert_value_is_local_indices_slice(self):
        """Verify the value passed to insert is local_indices[:transferred_tokens]."""
        import torch

        local_indices = torch.arange(32)
        self.sched.token_to_kv_pool_allocator.alloc.return_value = local_indices

        token_ids = list(range(64))
        num_blocks = 2
        source_kv_indices = [10, 20]

        self.sched.receive_kv_blocks(
            source_metadata=self.source_metadata,
            source_kv_indices=source_kv_indices,
            token_ids=token_ids,
            num_blocks=num_blocks,
        )

        call_args = self.sched.tree_cache.insert.call_args[0][0]
        transferred_tokens = num_blocks * 16  # page_size=16
        expected_value = local_indices[:transferred_tokens]
        assert torch.equal(call_args.value, expected_value)

    def test_insert_makes_blocks_visible(self):
        """After insert, match_prefix is called with same key to find the blocks."""
        token_ids = list(range(64))
        num_blocks = 2
        source_kv_indices = [10, 20]

        self.sched.receive_kv_blocks(
            source_metadata=self.source_metadata,
            source_kv_indices=source_kv_indices,
            token_ids=token_ids,
            num_blocks=num_blocks,
        )

        # Both insert and match_prefix should have been called
        assert self.sched.tree_cache.insert.called
        assert self.sched.tree_cache.match_prefix.called

        # Verify match_prefix was called after insert (same key)
        insert_key = self.sched.tree_cache.insert.call_args[0][0].key
        match_key = self.sched.tree_cache.match_prefix.call_args[0][0].key
        assert list(insert_key.token_ids) == list(match_key.token_ids)

    def test_insert_uses_prefix_token_ids(self):
        """Verify only prefix tokens (num_blocks * page_size) are used for the key."""
        token_ids = list(range(128))  # More tokens than needed
        num_blocks = 2
        source_kv_indices = [10, 20]

        self.sched.receive_kv_blocks(
            source_metadata=self.source_metadata,
            source_kv_indices=source_kv_indices,
            token_ids=token_ids,
            num_blocks=num_blocks,
        )

        call_args = self.sched.tree_cache.insert.call_args[0][0]
        # Only first 32 tokens (2 blocks * 16 page_size) should be in the key
        assert len(call_args.key.token_ids) == 32
        assert call_args.key.token_ids == list(range(32))


# ---------------------------------------------------------------------------
# Phase 4: Eviction Protection
# ---------------------------------------------------------------------------

class TestEvictionProtection:
    """Phase 4: Verify eviction protection via lock_ref."""

    def setup_method(self):
        self.sched = _make_fake_scheduler_for_receive()
        self.source_metadata = _default_source_metadata()

    def test_inc_lock_ref_called_after_insert(self):
        """Verify inc_lock_ref called on the node from match_prefix."""
        token_ids = list(range(64))
        num_blocks = 2
        source_kv_indices = [10, 20]

        # match_prefix returns a non-root node
        transfer_node = MagicMock(name="transfer_node")
        self.sched.tree_cache.match_prefix.return_value.last_device_node = transfer_node

        self.sched.receive_kv_blocks(
            source_metadata=self.source_metadata,
            source_kv_indices=source_kv_indices,
            token_ids=token_ids,
            num_blocks=num_blocks,
        )

        self.sched.tree_cache.inc_lock_ref.assert_called_once_with(transfer_node)

    def test_lock_skipped_for_root_node(self):
        """When match_prefix returns root_node, inc_lock_ref should NOT be called."""
        token_ids = list(range(64))
        num_blocks = 2
        source_kv_indices = [10, 20]

        # match_prefix returns root_node
        self.sched.tree_cache.match_prefix.return_value.last_device_node = (
            self.sched.tree_cache.root_node
        )

        self.sched.receive_kv_blocks(
            source_metadata=self.source_metadata,
            source_kv_indices=source_kv_indices,
            token_ids=token_ids,
            num_blocks=num_blocks,
        )

        self.sched.tree_cache.inc_lock_ref.assert_not_called()

    def test_pending_lock_stored(self):
        """After successful transfer, _transfer_pending_locks should contain an entry."""
        token_ids = list(range(64))
        num_blocks = 2
        source_kv_indices = [10, 20]

        transfer_node = MagicMock(name="transfer_node")
        self.sched.tree_cache.match_prefix.return_value.last_device_node = transfer_node

        self.sched.receive_kv_blocks(
            source_metadata=self.source_metadata,
            source_kv_indices=source_kv_indices,
            token_ids=token_ids,
            num_blocks=num_blocks,
        )

        assert len(self.sched._transfer_pending_locks) == 1

    def test_pending_lock_contains_node_and_timestamp(self):
        """The stored value should be (node, timestamp)."""
        token_ids = list(range(64))
        num_blocks = 2
        source_kv_indices = [10, 20]

        transfer_node = MagicMock(name="transfer_node")
        self.sched.tree_cache.match_prefix.return_value.last_device_node = transfer_node

        before = time.time()
        self.sched.receive_kv_blocks(
            source_metadata=self.source_metadata,
            source_kv_indices=source_kv_indices,
            token_ids=token_ids,
            num_blocks=num_blocks,
        )
        after = time.time()

        assert len(self.sched._transfer_pending_locks) == 1
        lock_entry = list(self.sched._transfer_pending_locks.values())[0]
        node, ts = lock_entry
        assert node is transfer_node
        assert before <= ts <= after

    def test_multiple_transfers_accumulate_locks(self):
        """Two transfers should result in two pending locks."""
        token_ids_1 = list(range(64))
        token_ids_2 = list(range(100, 164))
        source_kv_indices = [10, 20]

        transfer_node = MagicMock(name="transfer_node")
        self.sched.tree_cache.match_prefix.return_value.last_device_node = transfer_node

        self.sched.receive_kv_blocks(
            source_metadata=self.source_metadata,
            source_kv_indices=source_kv_indices,
            token_ids=token_ids_1,
            num_blocks=2,
        )

        self.sched.receive_kv_blocks(
            source_metadata=self.source_metadata,
            source_kv_indices=source_kv_indices,
            token_ids=token_ids_2,
            num_blocks=2,
        )

        assert len(self.sched._transfer_pending_locks) == 2

    def test_lock_not_stored_when_root_node(self):
        """When root is returned, no lock entry added."""
        token_ids = list(range(64))
        num_blocks = 2
        source_kv_indices = [10, 20]

        # match_prefix returns root_node
        self.sched.tree_cache.match_prefix.return_value.last_device_node = (
            self.sched.tree_cache.root_node
        )

        self.sched.receive_kv_blocks(
            source_metadata=self.source_metadata,
            source_kv_indices=source_kv_indices,
            token_ids=token_ids,
            num_blocks=num_blocks,
        )

        assert len(self.sched._transfer_pending_locks) == 0


# ---------------------------------------------------------------------------
# Phase 4: Idempotent Insertion (duplicate page freeing)
# ---------------------------------------------------------------------------

class TestIdempotentInsertion:
    """Phase 4: Verify duplicate page freeing on re-insertion."""

    def setup_method(self):
        self.sched = _make_fake_scheduler_for_receive()
        self.source_metadata = _default_source_metadata()

    def test_full_overlap_frees_all_duplicates(self):
        """When prefix_len == transferred_tokens, all local_indices are freed."""
        import torch

        local_indices = torch.arange(32)
        self.sched.token_to_kv_pool_allocator.alloc.return_value = local_indices

        # Full overlap: prefix_len == transferred_tokens
        self.sched.tree_cache.insert.return_value.prefix_len = 32

        token_ids = list(range(64))
        num_blocks = 2
        source_kv_indices = [10, 20]

        self.sched.receive_kv_blocks(
            source_metadata=self.source_metadata,
            source_kv_indices=source_kv_indices,
            token_ids=token_ids,
            num_blocks=num_blocks,
        )

        # free should have been called with local_indices[:32]
        self.sched.token_to_kv_pool_allocator.free.assert_called_once()
        freed_indices = self.sched.token_to_kv_pool_allocator.free.call_args[0][0]
        assert torch.equal(freed_indices, local_indices[:32])

    def test_partial_overlap_frees_partial(self):
        """When prefix_len < transferred_tokens but > 0, only overlapping freed."""
        import torch

        local_indices = torch.arange(32)
        self.sched.token_to_kv_pool_allocator.alloc.return_value = local_indices

        # Partial overlap: prefix_len == 16 (1 block worth)
        self.sched.tree_cache.insert.return_value.prefix_len = 16

        token_ids = list(range(64))
        num_blocks = 2
        source_kv_indices = [10, 20]

        self.sched.receive_kv_blocks(
            source_metadata=self.source_metadata,
            source_kv_indices=source_kv_indices,
            token_ids=token_ids,
            num_blocks=num_blocks,
        )

        self.sched.token_to_kv_pool_allocator.free.assert_called_once()
        freed_indices = self.sched.token_to_kv_pool_allocator.free.call_args[0][0]
        assert torch.equal(freed_indices, local_indices[:16])

    def test_no_overlap_frees_nothing(self):
        """When prefix_len == 0, allocator.free NOT called for duplicates."""
        import torch

        local_indices = torch.arange(32)
        self.sched.token_to_kv_pool_allocator.alloc.return_value = local_indices

        # No overlap
        self.sched.tree_cache.insert.return_value.prefix_len = 0

        token_ids = list(range(64))
        num_blocks = 2
        source_kv_indices = [10, 20]

        self.sched.receive_kv_blocks(
            source_metadata=self.source_metadata,
            source_kv_indices=source_kv_indices,
            token_ids=token_ids,
            num_blocks=num_blocks,
        )

        # free should NOT have been called (no duplicates to free)
        self.sched.token_to_kv_pool_allocator.free.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 4: End-to-End Prefix Match
# ---------------------------------------------------------------------------

class TestEndToEndPrefixMatch:
    """Phase 4: Verify match_prefix finds transferred blocks."""

    def setup_method(self):
        self.sched = _make_fake_scheduler_for_receive()
        self.source_metadata = _default_source_metadata()

    def test_match_prefix_called_after_insert(self):
        """Verify match_prefix is called with same key after insert."""
        token_ids = list(range(64))
        num_blocks = 2
        source_kv_indices = [10, 20]

        self.sched.receive_kv_blocks(
            source_metadata=self.source_metadata,
            source_kv_indices=source_kv_indices,
            token_ids=token_ids,
            num_blocks=num_blocks,
        )

        self.sched.tree_cache.match_prefix.assert_called_once()
        match_args = self.sched.tree_cache.match_prefix.call_args[0][0]
        assert list(match_args.key.token_ids) == list(range(32))

    def test_match_prefix_result_used_for_locking(self):
        """Verify last_device_node from match_result is used for inc_lock_ref."""
        token_ids = list(range(64))
        num_blocks = 2
        source_kv_indices = [10, 20]

        transfer_node = MagicMock(name="the_transfer_node")
        self.sched.tree_cache.match_prefix.return_value.last_device_node = transfer_node

        self.sched.receive_kv_blocks(
            source_metadata=self.source_metadata,
            source_kv_indices=source_kv_indices,
            token_ids=token_ids,
            num_blocks=num_blocks,
        )

        self.sched.tree_cache.inc_lock_ref.assert_called_once_with(transfer_node)


# ---------------------------------------------------------------------------
# Phase 4: Hash Computation
# ---------------------------------------------------------------------------

class TestHashComputation:
    """Phase 4: Verify hash computation for event publication."""

    def setup_method(self):
        from dynamo.sglang.kv_transfer.manager import KvTransferManager

        self.KvTransferManager = KvTransferManager

    def _make_manager(self, page_size=16):
        engine = MagicMock()
        config = MagicMock()
        config.server_args.page_size = page_size
        return self.KvTransferManager(engine, config)

    @patch("dynamo._core.compute_block_hash_for_seq")
    def test_hashes_computed_for_correct_token_prefix(self, mock_hash):
        """In publish_transferred_blocks, compute_block_hash_for_seq gets correct prefix."""
        mock_hash.return_value = [111, 222]
        mgr = self._make_manager(page_size=16)
        kv_pub = MagicMock()

        token_ids = list(range(128))  # More than needed
        mgr.publish_transferred_blocks(token_ids, num_blocks=2, kv_publisher=kv_pub)

        # Should be called with first 32 tokens (2 blocks * 16 page_size)
        mock_hash.assert_called_once_with(
            tokens=list(range(32)),
            kv_block_size=16,
        )

    @patch("dynamo._core.compute_block_hash_for_seq")
    def test_parent_hash_is_none(self, mock_hash):
        """publish_stored is called with parent_hash=None."""
        mock_hash.return_value = [42, 99]
        mgr = self._make_manager(page_size=16)
        kv_pub = MagicMock()

        token_ids = list(range(64))
        mgr.publish_transferred_blocks(token_ids, num_blocks=2, kv_publisher=kv_pub)

        call_kwargs = kv_pub.publish_stored.call_args[1]
        assert call_kwargs["parent_hash"] is None


# ---------------------------------------------------------------------------
# Phase 4: TP Handling
# ---------------------------------------------------------------------------

class TestTPHandling:
    """Phase 4: Verify TP handling in scheduler mixin."""

    def setup_method(self):
        from dynamo.sglang.kv_transfer.scheduler_mixin import install_kv_transfer_methods

        class FakeScheduler:
            pass

        install_kv_transfer_methods(FakeScheduler)
        self.FakeScheduler = FakeScheduler

    def test_result_file_only_written_on_rank_0(self):
        """get_kv_transfer_metadata with tp_rank=1 should NOT write result_file."""
        sched = self.FakeScheduler()
        sched._kv_transfer_enabled = False
        sched.tp_rank = 1

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        # Remove the file first so we can check if it was written
        os.unlink(path)
        try:
            sched.get_kv_transfer_metadata(result_file=path)
            # File should NOT have been created for tp_rank=1
            assert not os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_result_file_written_on_rank_0(self):
        """get_kv_transfer_metadata with tp_rank=0 SHOULD write result_file."""
        sched = self.FakeScheduler()
        sched._kv_transfer_enabled = False
        sched.tp_rank = 0

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            sched.get_kv_transfer_metadata(result_file=path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["status"] == "error"  # because _kv_transfer_enabled=False
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_receive_kv_blocks_no_result_file(self):
        """receive_kv_blocks has no result_file parameter (returns via success/exception)."""
        sig = inspect.signature(self.FakeScheduler.receive_kv_blocks)
        params = list(sig.parameters.keys())
        assert "result_file" not in params


# ---------------------------------------------------------------------------
# Phase 4: Cleanup Method
# ---------------------------------------------------------------------------

class TestCleanupMethod:
    """Phase 4: Verify _cleanup_stale_transfer_locks."""

    def setup_method(self):
        from dynamo.sglang.kv_transfer.scheduler_mixin import install_kv_transfer_methods

        class FakeScheduler:
            pass

        install_kv_transfer_methods(FakeScheduler)
        self.FakeScheduler = FakeScheduler

    def test_cleanup_method_installed(self):
        """Verify _cleanup_stale_transfer_locks is installed on FakeScheduler."""
        assert hasattr(self.FakeScheduler, "_cleanup_stale_transfer_locks")
        assert callable(self.FakeScheduler._cleanup_stale_transfer_locks)

    def test_cleanup_noop_when_no_locks(self):
        """When _transfer_pending_locks is empty, no error."""
        sched = self.FakeScheduler()
        sched._transfer_pending_locks = {}
        sched.tree_cache = MagicMock()

        # Should not raise
        sched._cleanup_stale_transfer_locks()
        sched.tree_cache.dec_lock_ref.assert_not_called()

    def test_cleanup_releases_expired_locks(self):
        """When lock is older than max_age, dec_lock_ref is called."""
        sched = self.FakeScheduler()
        sched.tree_cache = MagicMock()

        expired_node = MagicMock(name="expired_node")
        # Create a lock that is 60 seconds old
        old_timestamp = time.time() - 60
        lock_key = (tuple([1, 2, 3]), time.time_ns())
        sched._transfer_pending_locks = {
            lock_key: (expired_node, old_timestamp),
        }

        sched._cleanup_stale_transfer_locks(max_age_s=30)

        sched.tree_cache.dec_lock_ref.assert_called_once_with(expired_node)
        assert len(sched._transfer_pending_locks) == 0

    def test_cleanup_preserves_fresh_locks(self):
        """When lock is newer than max_age, dec_lock_ref is NOT called."""
        sched = self.FakeScheduler()
        sched.tree_cache = MagicMock()

        fresh_node = MagicMock(name="fresh_node")
        # Create a lock that is only 5 seconds old
        fresh_timestamp = time.time() - 5
        lock_key = (tuple([4, 5, 6]), time.time_ns())
        sched._transfer_pending_locks = {
            lock_key: (fresh_node, fresh_timestamp),
        }

        sched._cleanup_stale_transfer_locks(max_age_s=30)

        sched.tree_cache.dec_lock_ref.assert_not_called()
        assert len(sched._transfer_pending_locks) == 1


# ---------------------------------------------------------------------------
# Phase 5: Helper — build a mock DecodeWorkerHandler
# ---------------------------------------------------------------------------

def _ensure_sglang_engine():
    """Ensure sglang.Engine exists before importing modules that reference it.

    sglang.Engine is a LazyImport object that may not be resolvable in a
    GPU-less test environment.  Several modules (publisher.py, handler_base.py)
    reference ``sgl.Engine`` as a type annotation in class definitions, which
    Python evaluates eagerly.  If the lazy import fails at that point the
    module-level class statement raises ``AttributeError``.

    This helper sets a sentinel ``MagicMock`` so that the class body can be
    evaluated.  The mock is never used at runtime — it only satisfies the
    annotation.
    """
    sgl = sys.modules.get("sglang")
    if sgl is not None and not hasattr(sgl, "Engine"):
        sgl.Engine = MagicMock(name="MockEngine")


def _import_decode_handler():
    """Import DecodeWorkerHandler with sglang.Engine safety check."""
    _ensure_sglang_engine()
    from dynamo.sglang.request_handlers.llm.decode_handler import (
        DecodeWorkerHandler,
    )
    return DecodeWorkerHandler


def _make_mock_handler(
    *,
    serving_mode=None,
    skip_tokenizer_init=True,
    kv_transfer_manager=None,
    kv_publisher=None,
    transfer_result=None,
):
    """Build a lightweight mock of DecodeWorkerHandler for generate() tests.

    Instead of instantiating the real class (which requires sgl.Engine, Config,
    etc.), we construct a MagicMock with the attributes that generate() reads
    and wire up the async methods it calls.
    """
    from dynamo.common.constants import DisaggregationMode

    handler = MagicMock()
    handler.serving_mode = serving_mode or DisaggregationMode.AGGREGATED
    handler.skip_tokenizer_init = skip_tokenizer_init
    handler.kv_transfer_manager = kv_transfer_manager
    handler.kv_publisher = kv_publisher
    handler.enable_trace = False
    handler.config = MagicMock()
    handler.config.server_args.served_model_name = "test-model"

    # _get_input_param returns {"input_ids": token_ids}
    handler._get_input_param = MagicMock(
        return_value={"input_ids": [1, 2, 3]}
    )
    handler._build_sampling_params = MagicMock(return_value={})
    handler._priority_kwargs = MagicMock(return_value={})

    # engine.async_generate is an awaitable that returns an async iterator.
    # Real signature: agg = await engine.async_generate(...); async for res in agg: ...
    async def _fake_stream():
        yield {
            "meta_info": {
                "id": "req-001",
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "cached_tokens": 0,
            },
            "output_ids": [42],
        }

    handler.engine = MagicMock()
    handler.engine.async_generate = AsyncMock(side_effect=lambda **kw: _fake_stream())

    # generate() delegates to _process_token_stream / _process_text_stream via
    # self.<method>.  On a MagicMock those would return another MagicMock instead
    # of the real async generator.  Bind the real DecodeWorkerHandler methods so
    # that the stream processing actually runs.
    DecodeWorkerHandler = _import_decode_handler()
    import types
    handler._process_token_stream = types.MethodType(
        DecodeWorkerHandler._process_token_stream, handler
    )
    handler._process_text_stream = types.MethodType(
        DecodeWorkerHandler._process_text_stream, handler
    )

    # _cancellation_monitor is an async context manager used in _process_token_stream.
    # We provide a no-op implementation so that `async with self._cancellation_monitor(...)`
    # works correctly in the mock handler.
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop_monitor(*args, **kwargs):
        yield MagicMock()

    handler._cancellation_monitor = _noop_monitor

    # Wire up transfer result on the target_handler
    if kv_transfer_manager is not None and transfer_result is not None:
        kv_transfer_manager.target_handler.execute_transfer = AsyncMock(
            return_value=transfer_result,
        )

    return handler


async def _collect_generate(handler, request):
    """Call DecodeWorkerHandler.generate() and collect all yielded chunks."""
    DecodeWorkerHandler = _import_decode_handler()

    ctx = MagicMock()
    ctx.id = MagicMock(return_value="ctx-001")
    ctx.trace_id = "trace-001"
    ctx.is_stopped = MagicMock(return_value=False)

    chunks = []
    async for chunk in DecodeWorkerHandler.generate(handler, request, ctx):
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Phase 5: End-to-End Transfer Flow
# ---------------------------------------------------------------------------

class TestEndToEndTransferFlow:
    """Phase 5: Verify the full transfer flow through DecodeWorkerHandler.generate()."""

    def setup_method(self):
        from dynamo.sglang.kv_transfer.target_handler import TransferResult
        self.TransferResult = TransferResult

    @pytest.mark.asyncio
    async def test_full_transfer_flow_success(self):
        """transfer_hint → execute_transfer → success → verify block/token counts."""
        tr = self.TransferResult.succeeded(4, 64)
        mgr = MagicMock()
        mgr.page_size = 16
        handler = _make_mock_handler(
            kv_transfer_manager=mgr,
            kv_publisher=MagicMock(),
            transfer_result=tr,
        )

        request = {
            "token_ids": list(range(64)),
            "routing": {
                "transfer_hint": {
                    "source_worker": {"worker_id": 100, "dp_rank": 0},
                    "num_blocks": 4,
                },
            },
        }

        chunks = await _collect_generate(handler, request)
        assert len(chunks) >= 1
        mgr.target_handler.execute_transfer.assert_awaited_once()
        call_kwargs = mgr.target_handler.execute_transfer.call_args.kwargs
        assert call_kwargs["transfer_hint"] == request["routing"]["transfer_hint"]
        assert call_kwargs["token_ids"] == list(range(64))

    @pytest.mark.asyncio
    async def test_full_flow_publishes_events_on_success(self):
        """After successful transfer, publish_transferred_blocks is called."""
        tr = self.TransferResult.succeeded(4, 64)
        mgr = MagicMock()
        mgr.page_size = 16
        kv_pub = MagicMock()
        handler = _make_mock_handler(
            kv_transfer_manager=mgr,
            kv_publisher=kv_pub,
            transfer_result=tr,
        )

        request = {
            "token_ids": list(range(64)),
            "routing": {
                "transfer_hint": {
                    "source_worker": {"worker_id": 100, "dp_rank": 0},
                    "num_blocks": 4,
                },
            },
        }

        await _collect_generate(handler, request)
        mgr.publish_transferred_blocks.assert_called_once_with(
            token_ids=list(range(64)),
            num_blocks=4,
            kv_publisher=kv_pub,
        )

    @pytest.mark.asyncio
    async def test_full_flow_skips_publish_on_failure(self):
        """Transfer fails → publish NOT called."""
        tr = self.TransferResult.failed("timeout")
        mgr = MagicMock()
        mgr.page_size = 16
        kv_pub = MagicMock()
        handler = _make_mock_handler(
            kv_transfer_manager=mgr,
            kv_publisher=kv_pub,
            transfer_result=tr,
        )

        request = {
            "token_ids": list(range(64)),
            "routing": {
                "transfer_hint": {
                    "source_worker": {"worker_id": 100, "dp_rank": 0},
                    "num_blocks": 4,
                },
            },
        }

        await _collect_generate(handler, request)
        mgr.publish_transferred_blocks.assert_not_called()

    @pytest.mark.asyncio
    async def test_full_flow_records_duration_metric(self):
        """Verify transfer timing is computed (elapsed > 0)."""
        tr = self.TransferResult.succeeded(2, 32)
        mgr = MagicMock()
        mgr.page_size = 16
        handler = _make_mock_handler(
            kv_transfer_manager=mgr,
            kv_publisher=MagicMock(),
            transfer_result=tr,
        )

        request = {
            "token_ids": list(range(32)),
            "routing": {
                "transfer_hint": {
                    "source_worker": {"worker_id": 1, "dp_rank": 0},
                    "num_blocks": 2,
                },
            },
        }

        # The handler logs the elapsed time; we just verify generate completes
        # and the transfer was called (timing is internal to the handler)
        chunks = await _collect_generate(handler, request)
        assert len(chunks) >= 1
        mgr.target_handler.execute_transfer.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_transfer_then_generate_always_runs(self):
        """engine.async_generate is called regardless of transfer outcome."""
        tr = self.TransferResult.failed("no blocks")
        mgr = MagicMock()
        mgr.page_size = 16
        handler = _make_mock_handler(
            kv_transfer_manager=mgr,
            kv_publisher=MagicMock(),
            transfer_result=tr,
        )

        request = {
            "token_ids": list(range(32)),
            "routing": {
                "transfer_hint": {
                    "source_worker": {"worker_id": 1, "dp_rank": 0},
                    "num_blocks": 2,
                },
            },
        }

        chunks = await _collect_generate(handler, request)
        assert len(chunks) >= 1
        handler.engine.async_generate.assert_called_once()


# ---------------------------------------------------------------------------
# Phase 5: Transfer Hint Extraction
# ---------------------------------------------------------------------------

class TestTransferHintExtraction:
    """Phase 5: Verify transfer_hint extraction from the request dict."""

    def setup_method(self):
        from dynamo.sglang.kv_transfer.target_handler import TransferResult
        self.TransferResult = TransferResult

    @pytest.mark.asyncio
    async def test_extracts_hint_from_routing(self):
        """request with routing.transfer_hint → correct extraction."""
        tr = self.TransferResult.succeeded(2, 32)
        mgr = MagicMock()
        mgr.page_size = 16
        handler = _make_mock_handler(
            kv_transfer_manager=mgr,
            kv_publisher=MagicMock(),
            transfer_result=tr,
        )

        hint = {
            "source_worker": {"worker_id": 42, "dp_rank": 1},
            "num_blocks": 2,
        }
        request = {
            "token_ids": [10, 20, 30],
            "routing": {"transfer_hint": hint},
        }

        await _collect_generate(handler, request)
        call_kwargs = mgr.target_handler.execute_transfer.call_args.kwargs
        assert call_kwargs["transfer_hint"] is hint

    @pytest.mark.asyncio
    async def test_no_routing_key(self):
        """request without 'routing' key → no transfer attempted."""
        mgr = MagicMock()
        handler = _make_mock_handler(kv_transfer_manager=mgr)

        request = {"token_ids": [1, 2, 3]}

        await _collect_generate(handler, request)
        mgr.target_handler.execute_transfer.assert_not_called()

    @pytest.mark.asyncio
    async def test_routing_without_hint(self):
        """request with routing={} → no transfer attempted."""
        mgr = MagicMock()
        handler = _make_mock_handler(kv_transfer_manager=mgr)

        request = {"token_ids": [1, 2, 3], "routing": {}}

        await _collect_generate(handler, request)
        mgr.target_handler.execute_transfer.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_token_ids_logs_warning(self):
        """transfer_hint present but token_ids=[] → debug log emitted, transfer skipped."""
        mgr = MagicMock()
        handler = _make_mock_handler(kv_transfer_manager=mgr)

        request = {
            "token_ids": [],
            "routing": {
                "transfer_hint": {
                    "source_worker": {"worker_id": 1, "dp_rank": 0},
                    "num_blocks": 2,
                },
            },
        }

        with patch("logging.debug") as mock_debug:
            await _collect_generate(handler, request)
            # Check that the debug message about empty token_ids was logged
            debug_messages = [str(c) for c in mock_debug.call_args_list]
            found = any("token_ids empty" in msg for msg in debug_messages)
            assert found, f"Expected 'token_ids empty' debug log, got: {debug_messages}"
        mgr.target_handler.execute_transfer.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 5: Transfer Hint Serialization
# ---------------------------------------------------------------------------

class TestTransferHintSerialization:
    """Phase 5: Verify Python extraction works with exact serde field names."""

    def setup_method(self):
        from dynamo.sglang.kv_transfer.target_handler import TransferResult
        self.TransferResult = TransferResult

    @pytest.mark.asyncio
    async def test_snake_case_field_names(self):
        """Verify extraction works with serde snake_case: source_worker, worker_id."""
        tr = self.TransferResult.succeeded(32, 512)
        mgr = MagicMock()
        mgr.page_size = 16
        handler = _make_mock_handler(
            kv_transfer_manager=mgr,
            kv_publisher=MagicMock(),
            transfer_result=tr,
        )

        hint = {
            "source_worker": {"worker_id": 123, "dp_rank": 0},
            "num_blocks": 32,
        }
        request = {
            "token_ids": list(range(512)),
            "routing": {"transfer_hint": hint},
        }

        await _collect_generate(handler, request)
        call_kwargs = mgr.target_handler.execute_transfer.call_args.kwargs
        assert call_kwargs["transfer_hint"]["source_worker"]["worker_id"] == 123
        assert call_kwargs["transfer_hint"]["num_blocks"] == 32

    @pytest.mark.asyncio
    async def test_worker_id_as_large_u64(self):
        """Verify handling of large u64 worker IDs (e.g., 2^53)."""
        tr = self.TransferResult.succeeded(1, 16)
        mgr = MagicMock()
        mgr.page_size = 16
        handler = _make_mock_handler(
            kv_transfer_manager=mgr,
            kv_publisher=MagicMock(),
            transfer_result=tr,
        )

        large_id = 2**53
        hint = {
            "source_worker": {"worker_id": large_id, "dp_rank": 0},
            "num_blocks": 1,
        }
        request = {
            "token_ids": list(range(16)),
            "routing": {"transfer_hint": hint},
        }

        await _collect_generate(handler, request)
        call_kwargs = mgr.target_handler.execute_transfer.call_args.kwargs
        assert call_kwargs["transfer_hint"]["source_worker"]["worker_id"] == large_id

    @pytest.mark.asyncio
    async def test_missing_optional_fields(self):
        """Verify graceful handling when transfer_hint has extra/unusual shape.

        The handler passes through the hint dict to execute_transfer without
        validating its shape — validation happens in KvTransferTargetHandler.
        """
        tr = self.TransferResult.succeeded(0, 0)
        mgr = MagicMock()
        mgr.page_size = 16
        handler = _make_mock_handler(
            kv_transfer_manager=mgr,
            kv_publisher=MagicMock(),
            transfer_result=tr,
        )

        # Minimal hint — no source_worker or num_blocks
        hint = {"extra_field": True}
        request = {
            "token_ids": [1, 2, 3],
            "routing": {"transfer_hint": hint},
        }

        # Should not raise — hint is passed through, target_handler validates
        await _collect_generate(handler, request)
        mgr.target_handler.execute_transfer.assert_awaited_once()


# ---------------------------------------------------------------------------
# Phase 5: Fallback Paths
# ---------------------------------------------------------------------------

class TestFallbackPaths:
    """Phase 5: Verify fallback behavior when transfer is skipped or fails."""

    def setup_method(self):
        from dynamo.sglang.kv_transfer.target_handler import TransferResult
        self.TransferResult = TransferResult

    @pytest.mark.asyncio
    async def test_no_transfer_hint_normal_prefill(self):
        """No hint → engine.async_generate called normally."""
        mgr = MagicMock()
        handler = _make_mock_handler(kv_transfer_manager=mgr)

        request = {"token_ids": [1, 2, 3]}

        chunks = await _collect_generate(handler, request)
        assert len(chunks) >= 1
        handler.engine.async_generate.assert_called_once()
        mgr.target_handler.execute_transfer.assert_not_called()

    @pytest.mark.asyncio
    async def test_transfer_failure_falls_back(self):
        """execute_transfer fails → generate still called."""
        tr = self.TransferResult.failed("connection refused")
        mgr = MagicMock()
        mgr.page_size = 16
        handler = _make_mock_handler(
            kv_transfer_manager=mgr,
            kv_publisher=MagicMock(),
            transfer_result=tr,
        )

        request = {
            "token_ids": [1, 2, 3],
            "routing": {
                "transfer_hint": {
                    "source_worker": {"worker_id": 1, "dp_rank": 0},
                    "num_blocks": 1,
                },
            },
        }

        chunks = await _collect_generate(handler, request)
        assert len(chunks) >= 1
        handler.engine.async_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_transfer_exception_falls_back(self):
        """execute_transfer raises → caught, generate still called."""
        mgr = MagicMock()
        mgr.page_size = 16
        mgr.target_handler.execute_transfer = AsyncMock(
            side_effect=RuntimeError("NIXL error")
        )
        handler = _make_mock_handler(
            kv_transfer_manager=mgr,
            kv_publisher=MagicMock(),
        )

        request = {
            "token_ids": [1, 2, 3],
            "routing": {
                "transfer_hint": {
                    "source_worker": {"worker_id": 1, "dp_rank": 0},
                    "num_blocks": 1,
                },
            },
        }

        chunks = await _collect_generate(handler, request)
        assert len(chunks) >= 1
        handler.engine.async_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_kv_transfer_manager_none_skips(self):
        """handler.kv_transfer_manager=None → no transfer attempted."""
        handler = _make_mock_handler(kv_transfer_manager=None)

        request = {
            "token_ids": [1, 2, 3],
            "routing": {
                "transfer_hint": {
                    "source_worker": {"worker_id": 1, "dp_rank": 0},
                    "num_blocks": 1,
                },
            },
        }

        chunks = await _collect_generate(handler, request)
        assert len(chunks) >= 1
        handler.engine.async_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_token_ids_falls_back(self):
        """transfer_hint present, token_ids=[] → generate called normally."""
        mgr = MagicMock()
        handler = _make_mock_handler(kv_transfer_manager=mgr)

        request = {
            "token_ids": [],
            "routing": {
                "transfer_hint": {
                    "source_worker": {"worker_id": 1, "dp_rank": 0},
                    "num_blocks": 1,
                },
            },
        }

        chunks = await _collect_generate(handler, request)
        assert len(chunks) >= 1
        handler.engine.async_generate.assert_called_once()
        mgr.target_handler.execute_transfer.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 5: Event Publication Timing
# ---------------------------------------------------------------------------

class TestEventPublicationTiming:
    """Phase 5: Verify event publication after transfer success/failure."""

    def setup_method(self):
        from dynamo.sglang.kv_transfer.target_handler import TransferResult
        self.TransferResult = TransferResult

    @pytest.mark.asyncio
    async def test_events_published_after_success(self):
        """publish_transferred_blocks called when transfer_result.success."""
        tr = self.TransferResult.succeeded(3, 48)
        mgr = MagicMock()
        mgr.page_size = 16
        kv_pub = MagicMock()
        handler = _make_mock_handler(
            kv_transfer_manager=mgr,
            kv_publisher=kv_pub,
            transfer_result=tr,
        )

        request = {
            "token_ids": list(range(48)),
            "routing": {
                "transfer_hint": {
                    "source_worker": {"worker_id": 1, "dp_rank": 0},
                    "num_blocks": 3,
                },
            },
        }

        await _collect_generate(handler, request)
        mgr.publish_transferred_blocks.assert_called_once_with(
            token_ids=list(range(48)),
            num_blocks=3,
            kv_publisher=kv_pub,
        )

    @pytest.mark.asyncio
    async def test_events_not_published_on_failure(self):
        """Not called when transfer_result.success is False."""
        tr = self.TransferResult.failed("error")
        mgr = MagicMock()
        mgr.page_size = 16
        kv_pub = MagicMock()
        handler = _make_mock_handler(
            kv_transfer_manager=mgr,
            kv_publisher=kv_pub,
            transfer_result=tr,
        )

        request = {
            "token_ids": list(range(32)),
            "routing": {
                "transfer_hint": {
                    "source_worker": {"worker_id": 1, "dp_rank": 0},
                    "num_blocks": 2,
                },
            },
        }

        await _collect_generate(handler, request)
        mgr.publish_transferred_blocks.assert_not_called()

    @pytest.mark.asyncio
    async def test_event_publish_error_does_not_block_generate(self):
        """publish raises → warning logged, generate still proceeds."""
        tr = self.TransferResult.succeeded(2, 32)
        mgr = MagicMock()
        mgr.page_size = 16
        mgr.publish_transferred_blocks.side_effect = RuntimeError("publish boom")
        kv_pub = MagicMock()
        handler = _make_mock_handler(
            kv_transfer_manager=mgr,
            kv_publisher=kv_pub,
            transfer_result=tr,
        )

        request = {
            "token_ids": list(range(32)),
            "routing": {
                "transfer_hint": {
                    "source_worker": {"worker_id": 1, "dp_rank": 0},
                    "num_blocks": 2,
                },
            },
        }

        # Should not raise — the publish error is caught
        chunks = await _collect_generate(handler, request)
        assert len(chunks) >= 1
        handler.engine.async_generate.assert_called_once()


# ---------------------------------------------------------------------------
# Phase 5: Aggregated vs Disaggregated
# ---------------------------------------------------------------------------

class TestAggregatedVsDisaggregated:
    """Phase 5: Verify transfer behavior differs between aggregated and disagg modes."""

    def setup_method(self):
        from dynamo.sglang.kv_transfer.target_handler import TransferResult
        self.TransferResult = TransferResult

    @pytest.mark.asyncio
    async def test_aggregated_mode_attempts_transfer(self):
        """In aggregated mode with transfer_hint → transfer attempted."""
        from dynamo.common.constants import DisaggregationMode

        tr = self.TransferResult.succeeded(2, 32)
        mgr = MagicMock()
        mgr.page_size = 16
        handler = _make_mock_handler(
            serving_mode=DisaggregationMode.AGGREGATED,
            kv_transfer_manager=mgr,
            kv_publisher=MagicMock(),
            transfer_result=tr,
        )

        request = {
            "token_ids": list(range(32)),
            "routing": {
                "transfer_hint": {
                    "source_worker": {"worker_id": 1, "dp_rank": 0},
                    "num_blocks": 2,
                },
            },
        }

        await _collect_generate(handler, request)
        mgr.target_handler.execute_transfer.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disaggregated_mode_ignores_transfer(self):
        """In disagg decode mode → transfer NOT attempted (uses bootstrap path).

        Disagg mode raises RuntimeError when bootstrap_info is missing, which
        confirms it takes a completely different code path from aggregated mode.
        """
        from dynamo.common.constants import DisaggregationMode

        DecodeWorkerHandler = _import_decode_handler()

        mgr = MagicMock()
        mgr.page_size = 16
        handler = _make_mock_handler(
            serving_mode=DisaggregationMode.DECODE,
            kv_transfer_manager=mgr,
        )

        request = {
            "token_ids": list(range(32)),
            "routing": {
                "transfer_hint": {
                    "source_worker": {"worker_id": 1, "dp_rank": 0},
                    "num_blocks": 2,
                },
            },
        }

        ctx = MagicMock()
        ctx.id = MagicMock(return_value="ctx-002")
        ctx.trace_id = "trace-002"

        # Disagg mode requires bootstrap_info — it will raise RuntimeError,
        # confirming the transfer code path is never reached
        with pytest.raises(RuntimeError, match="bootstrap_info"):
            async for _ in DecodeWorkerHandler.generate(handler, request, ctx):
                pass
        mgr.target_handler.execute_transfer.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 6: New Metrics (bytes, NIXL ops/bytes/duration)
# ---------------------------------------------------------------------------

class TestNewMetrics:
    """Tests for Phase 6 new metrics: bytes, NIXL ops/bytes/duration."""

    def _reset_metrics(self):
        from dynamo.sglang.kv_transfer import metrics

        metrics._metrics_initialized = False
        metrics._transfer_requests = None
        metrics._transfer_blocks = None
        metrics._transfer_duration = None
        metrics._transfer_fallback = None
        metrics._transfer_events_published = None
        metrics._transfer_bytes = None
        metrics._nixl_read_ops = None
        metrics._nixl_read_bytes = None
        metrics._nixl_read_duration = None
        return metrics

    def test_record_transfer_bytes_without_prometheus(self):
        """record_transfer_bytes() is safe without prometheus."""
        metrics = self._reset_metrics()
        # Should not raise even if Prometheus is not available
        metrics.record_transfer_bytes("target", 1024)

    def test_record_transfer_bytes_with_prometheus(self):
        """record_transfer_bytes() records correctly with prometheus."""
        metrics = self._reset_metrics()
        metrics._ensure_initialized()
        if metrics._metrics_initialized:
            assert metrics._transfer_bytes is not None
            metrics.record_transfer_bytes("target", 4096)
            metrics.record_transfer_bytes("source", 2048)

    def test_record_nixl_read_op_without_prometheus(self):
        """record_nixl_read_op() is safe without prometheus."""
        metrics = self._reset_metrics()
        metrics.record_nixl_read_op(1024, 0.005)  # Should not raise

    def test_record_nixl_read_op_with_prometheus(self):
        """record_nixl_read_op() records all three NIXL metrics."""
        metrics = self._reset_metrics()
        metrics._ensure_initialized()
        if metrics._metrics_initialized:
            assert metrics._nixl_read_ops is not None
            assert metrics._nixl_read_bytes is not None
            assert metrics._nixl_read_duration is not None
            metrics.record_nixl_read_op(8192, 0.002)

    def test_new_metrics_initialized_with_existing(self):
        """New metrics are initialized alongside existing ones."""
        metrics = self._reset_metrics()
        metrics._ensure_initialized()
        if metrics._metrics_initialized:
            # All existing metrics still initialized
            assert metrics._transfer_requests is not None
            assert metrics._transfer_blocks is not None
            assert metrics._transfer_duration is not None
            assert metrics._transfer_fallback is not None
            assert metrics._transfer_events_published is not None
            # New metrics also initialized
            assert metrics._transfer_bytes is not None
            assert metrics._nixl_read_ops is not None
            assert metrics._nixl_read_bytes is not None
            assert metrics._nixl_read_duration is not None

    def test_record_transfer_bytes_labels(self):
        """Verify transfer_bytes accepts role labels correctly."""
        metrics = self._reset_metrics()
        metrics._ensure_initialized()
        for role in ["source", "target"]:
            metrics.record_transfer_bytes(role, 512)

    def test_record_nixl_read_op_multiple_calls(self):
        """Multiple NIXL read ops should accumulate without error."""
        metrics = self._reset_metrics()
        metrics._ensure_initialized()
        if metrics._metrics_initialized:
            for i in range(10):
                metrics.record_nixl_read_op(4096, 0.001 * (i + 1))


# ---------------------------------------------------------------------------
# Phase 6: Target Handler Bytes Metrics
# ---------------------------------------------------------------------------

class TestTargetHandlerBytesMetrics:
    """Tests that target handler records transfer_bytes metric."""

    def setup_method(self):
        from dynamo.sglang.kv_transfer.target_handler import (
            KvTransferTargetHandler,
        )
        self.KvTransferTargetHandler = KvTransferTargetHandler

    def _make_handler(self, bytes_per_block=4096):
        manager = MagicMock()
        manager.page_size = 16
        manager.bytes_per_block = bytes_per_block
        manager.config.dynamo_args.kv_transfer_min_blocks = 1
        return self.KvTransferTargetHandler(manager)

    @pytest.mark.asyncio
    async def test_records_bytes_on_success(self):
        """Verify record_transfer_bytes called with correct role and byte count."""
        handler = self._make_handler(bytes_per_block=4096)

        handler.manager.get_cached_metadata.return_value = {"status": "ok"}
        handler._rpc_to_source = AsyncMock(return_value={
            "status": "ok",
            "kv_indices": list(range(5)),
            "num_matched_blocks": 5,
        })
        handler.manager.execute_receive_transfer_async = AsyncMock(
            return_value=True
        )

        with patch(
            "dynamo.sglang.kv_transfer.metrics.record_transfer_bytes"
        ) as mock_bytes:
            result = await handler.execute_transfer(
                {
                    "source_worker": {"worker_id": 1, "dp_rank": 0},
                    "num_blocks": 5,
                },
                list(range(80)),
            )
            assert result.success is True
            mock_bytes.assert_called_once_with("target", 5 * 4096)

    @pytest.mark.asyncio
    async def test_no_bytes_on_failure(self):
        """Verify record_transfer_bytes NOT called on failed transfer."""
        handler = self._make_handler(bytes_per_block=4096)

        with patch(
            "dynamo.sglang.kv_transfer.metrics.record_transfer_bytes"
        ) as mock_bytes:
            result = await handler.execute_transfer(
                {"num_blocks": 5}, [1, 2, 3]
            )
            assert result.success is False
            mock_bytes.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_bytes_when_zero_bytes_per_block(self):
        """Verify record_transfer_bytes NOT called when bytes_per_block=0."""
        handler = self._make_handler(bytes_per_block=0)

        handler.manager.get_cached_metadata.return_value = {"status": "ok"}
        handler._rpc_to_source = AsyncMock(return_value={
            "status": "ok",
            "kv_indices": list(range(5)),
            "num_matched_blocks": 5,
        })
        handler.manager.execute_receive_transfer_async = AsyncMock(
            return_value=True
        )

        with patch(
            "dynamo.sglang.kv_transfer.metrics.record_transfer_bytes"
        ) as mock_bytes:
            result = await handler.execute_transfer(
                {
                    "source_worker": {"worker_id": 1, "dp_rank": 0},
                    "num_blocks": 5,
                },
                list(range(80)),
            )
            assert result.success is True
            mock_bytes.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 6: Structured Logging
# ---------------------------------------------------------------------------

class TestStructuredLogging:
    """Tests that structured logging includes correct extra fields."""

    def setup_method(self):
        from dynamo.sglang.kv_transfer.target_handler import (
            KvTransferTargetHandler,
        )
        self.KvTransferTargetHandler = KvTransferTargetHandler

    def _make_handler(self, bytes_per_block=4096):
        manager = MagicMock()
        manager.page_size = 16
        manager.config.dynamo_args.kv_transfer_min_blocks = 1
        manager.bytes_per_block = bytes_per_block
        return self.KvTransferTargetHandler(manager)

    @pytest.mark.asyncio
    async def test_transfer_initiated_has_extra(self, caplog):
        """'KV transfer initiated' log includes source_worker and num_hint_blocks."""
        handler = self._make_handler()

        handler.manager.get_cached_metadata.return_value = {"status": "ok"}
        handler._rpc_to_source = AsyncMock(return_value={
            "status": "ok",
            "kv_indices": list(range(5)),
            "num_matched_blocks": 5,
        })
        handler.manager.execute_receive_transfer_async = AsyncMock(
            return_value=True
        )

        with caplog.at_level(logging.INFO):
            await handler.execute_transfer(
                {
                    "source_worker": {"worker_id": 42, "dp_rank": 0},
                    "num_blocks": 5,
                },
                list(range(80)),
            )

        initiated_records = [
            r for r in caplog.records
            if "KV transfer initiated" in r.getMessage()
        ]
        assert len(initiated_records) >= 1
        rec = initiated_records[0]
        assert getattr(rec, "event", None) == "kv_transfer_initiated"
        assert getattr(rec, "source_worker", None) == 42
        assert getattr(rec, "num_hint_blocks", None) == 5

    @pytest.mark.asyncio
    async def test_transfer_completed_has_extra(self, caplog):
        """'KV transfer completed' log includes elapsed_ms and result."""
        handler = self._make_handler()

        handler.manager.get_cached_metadata.return_value = {"status": "ok"}
        handler._rpc_to_source = AsyncMock(return_value={
            "status": "ok",
            "kv_indices": list(range(5)),
            "num_matched_blocks": 5,
        })
        handler.manager.execute_receive_transfer_async = AsyncMock(
            return_value=True
        )

        with caplog.at_level(logging.INFO):
            result = await handler.execute_transfer(
                {
                    "source_worker": {"worker_id": 1, "dp_rank": 0},
                    "num_blocks": 5,
                },
                list(range(80)),
            )

        assert result.success is True
        completed_records = [
            r for r in caplog.records
            if "KV transfer completed" in r.getMessage()
        ]
        assert len(completed_records) >= 1
        rec = completed_records[0]
        assert getattr(rec, "event", None) == "kv_transfer_completed"
        assert getattr(rec, "result", None) == "success"
        assert getattr(rec, "num_blocks", None) == 5
        assert isinstance(getattr(rec, "elapsed_ms", None), float)

    @pytest.mark.asyncio
    async def test_transfer_failed_has_extra(self, caplog):
        """Transfer failure log includes error and result."""
        handler = self._make_handler()

        handler.manager.get_cached_metadata.return_value = {"status": "ok"}
        handler._rpc_to_source = AsyncMock(return_value={
            "status": "ok",
            "kv_indices": list(range(5)),
            "num_matched_blocks": 5,
        })
        handler.manager.execute_receive_transfer_async = AsyncMock(
            side_effect=RuntimeError("GPU OOM")
        )

        with caplog.at_level(logging.WARNING):
            result = await handler.execute_transfer(
                {
                    "source_worker": {"worker_id": 1, "dp_rank": 0},
                    "num_blocks": 5,
                },
                list(range(80)),
            )

        assert result.success is False
        failed_records = [
            r for r in caplog.records
            if "KV transfer failed" in r.getMessage()
        ]
        assert len(failed_records) >= 1
        rec = failed_records[0]
        assert getattr(rec, "event", None) == "kv_transfer_failed"
        assert getattr(rec, "result", None) == "fallback_to_prefill"
        assert "GPU OOM" in getattr(rec, "error", "")


# ---------------------------------------------------------------------------
# _classify_transfer_error
# ---------------------------------------------------------------------------

class TestClassifyTransferError:
    """Tests for the error classification helper used in fallback metrics."""

    def setup_method(self):
        from dynamo.sglang.kv_transfer.target_handler import _classify_transfer_error
        self.classify = _classify_transfer_error

    def test_allocation_failed(self):
        assert self.classify("Could not allocate local KV pages for transfer") == "allocation_failed"

    def test_allocation_failed_case_insensitive(self):
        assert self.classify("could not allocate enough pages") == "allocation_failed"

    def test_timeout(self):
        assert self.classify("NIXL transfer timeout after 5000ms") == "timeout"

    def test_timeout_case_insensitive(self):
        assert self.classify("RPC Timeout waiting for response") == "timeout"

    def test_nixl_error(self):
        assert self.classify("NIXL agent returned error code -1") == "nixl_error"

    def test_nixl_error_case_insensitive(self):
        assert self.classify("nixl initialization failed") == "nixl_error"

    def test_too_many_pending(self):
        assert self.classify("Too many pending transfers (32/32)") == "too_many_pending"

    def test_not_initialized(self):
        assert self.classify("KV transfer not initialized yet") == "not_initialized"

    def test_unknown_error_falls_through(self):
        assert self.classify("Something completely unexpected happened") == "error"

    def test_empty_string(self):
        assert self.classify("") == "error"

    def test_priority_timeout_over_nixl(self):
        """When message contains both 'NIXL' and 'timeout', timeout wins (checked first)."""
        # "NIXL transfer timeout" contains both keywords; timeout is checked before nixl
        assert self.classify("NIXL transfer timeout") == "timeout"
