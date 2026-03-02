# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def __getattr__(name):
    """Lazy import to avoid requiring sglang at package import time."""
    if name == "KvTransferManager":
        from dynamo.sglang.kv_transfer.manager import KvTransferManager

        return KvTransferManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["KvTransferManager"]
