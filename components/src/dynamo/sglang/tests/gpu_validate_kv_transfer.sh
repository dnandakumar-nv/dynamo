#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# GPU validation script for KV cache transfer.
#
# Requirements:
#   - 2+ NVIDIA GPUs
#   - SGLang with NIXL support
#   - Dynamo runtime
#
# Usage:
#   bash gpu_validate_kv_transfer.sh [--verbose]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running KV Transfer GPU validation..."
python3 "${SCRIPT_DIR}/gpu_validate_kv_transfer.py" "$@"
