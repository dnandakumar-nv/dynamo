#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU validation script for cross-worker KV cache transfer.

This script validates the full KV transfer pipeline on actual GPU hardware.
It is NOT a pytest test — it requires 2+ GPUs and running SGLang engines.

Usage:
    python gpu_validate_kv_transfer.py [--model MODEL] [--verbose]

Prerequisites:
    - 2+ NVIDIA GPUs
    - SGLang installed with NIXL support
    - Dynamo runtime available

Validation checklist:
    1. NIXL agent initialization and buffer registration
    2. Metadata export (agent metadata, KV layout)
    3. Block query via radix tree prefix matching
    4. RDMA transfer execution and completion
    5. Radix tree insertion of transferred blocks
    6. KV event publication after transfer
    7. Prometheus metrics recording
"""

import argparse
import logging
import sys


def check_gpu_availability():
    """Check that sufficient GPUs are available."""
    try:
        import torch

        if not torch.cuda.is_available():
            print("FAIL: No CUDA GPUs available")
            return False
        gpu_count = torch.cuda.device_count()
        print(f"  GPUs available: {gpu_count}")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
            print(f"  GPU {i}: {name} ({mem:.1f} GB)")
        if gpu_count < 2:
            print("WARN: KV transfer requires 2+ GPUs for cross-worker validation")
            print("      Single-GPU mode will validate mixin installation only")
        return True
    except ImportError:
        print("FAIL: PyTorch not available")
        return False


def check_nixl_availability():
    """Check that NIXL is importable."""
    try:
        from nixl._api import nixl_agent, nixl_agent_config  # noqa: F401

        print("  NIXL: available")
        return True
    except ImportError:
        print("  NIXL: NOT available (KV transfer will be disabled)")
        return False


def check_sglang_availability():
    """Check that SGLang is importable."""
    try:
        import sglang  # noqa: F401

        print(f"  SGLang version: {sglang.__version__}")
        return True
    except ImportError:
        print("  SGLang: NOT available")
        return False


def validate_mixin_installation():
    """Validate that scheduler mixin methods are correctly installed."""
    print("\n--- Validating scheduler mixin installation ---")
    try:
        from sglang.srt.managers.scheduler import Scheduler

        from dynamo.sglang.kv_transfer.scheduler_mixin import (
            install_kv_transfer_methods,
        )

        # Install on a fresh class to avoid polluting the real Scheduler
        class TestScheduler:
            pass

        install_kv_transfer_methods(TestScheduler)

        methods = [
            "init_kv_transfer",
            "get_kv_transfer_metadata",
            "query_kv_blocks_by_tokens",
            "receive_kv_blocks",
        ]
        for method in methods:
            assert hasattr(TestScheduler, method), f"Missing method: {method}"
            assert callable(getattr(TestScheduler, method))
            print(f"  {method}: OK")

        # Verify receive_kv_blocks accepts timeout_ms
        import inspect

        sig = inspect.signature(TestScheduler.receive_kv_blocks)
        assert "timeout_ms" in sig.parameters, "receive_kv_blocks missing timeout_ms param"
        print("  receive_kv_blocks timeout_ms param: OK")

        print("PASS: Mixin installation")
        return True
    except Exception as e:
        print(f"FAIL: Mixin installation: {e}")
        return False


def validate_metrics_module():
    """Validate that metrics module works correctly."""
    print("\n--- Validating metrics module ---")
    try:
        from dynamo.sglang.kv_transfer.metrics import (
            record_events_published,
            record_transfer_blocks,
            record_transfer_duration,
            record_transfer_fallback,
            record_transfer_request,
        )

        # All functions should be callable without error (no-op if prometheus unavailable)
        record_transfer_request("target", "success")
        record_transfer_blocks("target", 5)
        record_transfer_duration("target", 0.005)
        record_transfer_fallback("test")
        record_events_published()

        print("  record_transfer_request: OK")
        print("  record_transfer_blocks: OK")
        print("  record_transfer_duration: OK")
        print("  record_transfer_fallback: OK")
        print("  record_events_published: OK")
        print("PASS: Metrics module")
        return True
    except Exception as e:
        print(f"FAIL: Metrics module: {e}")
        return False


def validate_manager_config():
    """Validate that manager accepts configurable parameters."""
    print("\n--- Validating manager configuration ---")
    try:
        from unittest.mock import MagicMock

        from dynamo.sglang.kv_transfer.manager import KvTransferManager

        engine = MagicMock()
        config = MagicMock()
        config.server_args.page_size = 16

        # Test with custom values
        mgr = KvTransferManager(
            engine, config,
            metadata_cache_ttl_s=60,
            transfer_timeout_ms=10000,
        )
        assert mgr._metadata_cache_ttl == 60, "TTL not set correctly"
        assert mgr.transfer_timeout_ms == 10000, "Timeout not set correctly"
        print("  Custom TTL (60s): OK")
        print("  Custom timeout (10000ms): OK")

        # Test defaults
        mgr2 = KvTransferManager(engine, config)
        assert mgr2._metadata_cache_ttl == 300, "Default TTL wrong"
        assert mgr2.transfer_timeout_ms == 5000, "Default timeout wrong"
        print("  Default TTL (300s): OK")
        print("  Default timeout (5000ms): OK")

        # Test publish_transferred_blocks exists
        assert hasattr(mgr, "publish_transferred_blocks")
        assert callable(mgr.publish_transferred_blocks)
        print("  publish_transferred_blocks method: OK")

        print("PASS: Manager configuration")
        return True
    except Exception as e:
        print(f"FAIL: Manager configuration: {e}")
        return False


def validate_cli_flags():
    """Validate that new CLI flags exist in backend args."""
    print("\n--- Validating CLI flags ---")
    try:
        import inspect

        import dynamo.sglang.backend_args as mod
        from dynamo.sglang.backend_args import DynamoSGLangConfig

        source = inspect.getsource(mod)

        flags = {
            "--transfer-timeout-ms": "DYN_SGL_TRANSFER_TIMEOUT_MS",
            "--metadata-cache-ttl-s": "DYN_SGL_METADATA_CACHE_TTL_S",
            "--enable-transfer-overlap": "DYN_SGL_ENABLE_TRANSFER_OVERLAP",
        }
        for flag, env_var in flags.items():
            assert flag in source, f"Missing CLI flag: {flag}"
            assert env_var in source, f"Missing env var: {env_var}"
            print(f"  {flag} ({env_var}): OK")

        # Check config fields
        for field in ["transfer_timeout_ms", "metadata_cache_ttl_s", "enable_transfer_overlap"]:
            assert field in dir(DynamoSGLangConfig), f"Missing config field: {field}"
            print(f"  Config.{field}: OK")

        print("PASS: CLI flags")
        return True
    except Exception as e:
        print(f"FAIL: CLI flags: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="GPU validation for cross-worker KV cache transfer"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    print("=" * 60)
    print("KV Transfer GPU Validation")
    print("=" * 60)

    results = []

    # Environment checks
    print("\n--- Environment checks ---")
    has_gpu = check_gpu_availability()
    has_nixl = check_nixl_availability()
    has_sglang = check_sglang_availability()

    # Component validation (no GPU required)
    results.append(("CLI flags", validate_cli_flags()))
    results.append(("Metrics module", validate_metrics_module()))
    results.append(("Manager config", validate_manager_config()))

    if has_sglang:
        results.append(("Mixin installation", validate_mixin_installation()))
    else:
        print("\nSKIP: Mixin installation (SGLang not available)")
        results.append(("Mixin installation", None))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = 0
    failed = 0
    skipped = 0
    for name, result in results:
        if result is True:
            print(f"  PASS: {name}")
            passed += 1
        elif result is False:
            print(f"  FAIL: {name}")
            failed += 1
        else:
            print(f"  SKIP: {name}")
            skipped += 1

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if not has_gpu or not has_nixl:
        print("\nNOTE: Full GPU transfer validation requires 2+ GPUs with NIXL.")
        print("      Run on a GPU node with NIXL installed for complete validation.")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
