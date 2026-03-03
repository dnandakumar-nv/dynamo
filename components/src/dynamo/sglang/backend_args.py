# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo SGLang wrapper configuration ArgGroup."""

from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument

from . import __version__


class DynamoSGLangArgGroup(ArgGroup):
    """SGLang-specific Dynamo wrapper configuration (not native SGLang engine args)."""

    name = "dynamo-sglang"

    def add_arguments(self, parser) -> None:
        """Add Dynamo SGLang arguments to parser."""

        parser.add_argument(
            "--version",
            action="version",
            version=f"Dynamo Backend SGLang {__version__}",
        )

        g = parser.add_argument_group("Dynamo SGLang Options")

        add_negatable_bool_argument(
            g,
            flag_name="--use-sglang-tokenizer",
            env_var="DYN_SGL_USE_TOKENIZER",
            default=False,
            help="Use SGLang's tokenizer for pre and post processing. This bypasses Dynamo's preprocessor and only v1/chat/completions will be available through the Dynamo frontend. Cannot be used with --custom-jinja-template.",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--multimodal-processor",
            env_var="DYN_SGL_MULTIMODAL_PROCESSOR",
            default=False,
            help="Run as multimodal processor component for handling multimodal requests.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--multimodal-encode-worker",
            env_var="DYN_SGL_MULTIMODAL_ENCODE_WORKER",
            default=False,
            help="Run as multimodal encode worker component for processing images/videos.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--multimodal-worker",
            env_var="DYN_SGL_MULTIMODAL_WORKER",
            default=False,
            help="Run as multimodal worker component for LLM inference with multimodal data.",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--embedding-worker",
            env_var="DYN_SGL_EMBEDDING_WORKER",
            default=False,
            help="Run as embedding worker component (Dynamo flag, also sets SGLang's --is-embedding).",
        )

        add_negatable_bool_argument(
            g,
            flag_name="--image-diffusion-worker",
            env_var="DYN_SGL_IMAGE_DIFFUSION_WORKER",
            default=False,
            help="Run as image diffusion worker for image generation.",
        )
        add_argument(
            g,
            flag_name="--disagg-config",
            env_var="DYN_SGL_DISAGG_CONFIG",
            default=None,
            help="Disaggregation configuration file in YAML format.",
        )
        add_argument(
            g,
            flag_name="--disagg-config-key",
            env_var="DYN_SGL_DISAGG_CONFIG_KEY",
            default=None,
            help="Key to select from nested disaggregation configuration file (e.g., 'prefill', 'decode').",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--video-generation-worker",
            env_var="DYN_SGL_VIDEO_GENERATION_WORKER",
            default=False,
            help="Run as video generation worker for video generation (T2V/I2V).",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-kv-transfer",
            env_var="DYN_SGL_ENABLE_KV_TRANSFER",
            default=False,
            help="Enable cross-worker KV cache transfer service. "
            "When enabled, this worker can receive KV blocks from "
            "remote workers and serve its own blocks to others via NIXL RDMA.",
        )
        add_argument(
            g,
            flag_name="--transfer-timeout-ms",
            env_var="DYN_SGL_TRANSFER_TIMEOUT_MS",
            default=5000,
            arg_type=int,
            help="NIXL transfer timeout in milliseconds (default: 5000).",
        )
        add_argument(
            g,
            flag_name="--metadata-cache-ttl-s",
            env_var="DYN_SGL_METADATA_CACHE_TTL_S",
            default=300,
            arg_type=int,
            help="Metadata cache TTL in seconds for remote worker NIXL info (default: 300).",
        )
        add_argument(
            g,
            flag_name="--max-pending-kv-transfers",
            env_var="DYN_SGL_MAX_PENDING_KV_TRANSFERS",
            default=4,
            arg_type=int,
            help="Max concurrent RDMA transfers per scheduler (default: 4).",
        )
        add_argument(
            g,
            flag_name="--max-nixl-starts-per-poll",
            env_var="DYN_SGL_MAX_NIXL_STARTS_PER_POLL",
            default=2,
            arg_type=int,
            help="Max new RDMA transfers started per scheduler iteration (default: 2). "
            "Prevents burst traffic from overwhelming the NIXL/RDMA layer.",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-transfer-overlap",
            env_var="DYN_SGL_ENABLE_TRANSFER_OVERLAP",
            default=False,
            help="Enable transfer/prefill overlap optimization (experimental, default: disabled).",
        )


class DynamoSGLangConfig(ConfigBase):
    """Configuration for Dynamo SGLang wrapper (SGLang-specific only)."""

    use_sglang_tokenizer: bool
    multimodal_processor: bool
    multimodal_encode_worker: bool
    multimodal_worker: bool
    embedding_worker: bool
    image_diffusion_worker: bool

    disagg_config: Optional[str] = None
    disagg_config_key: Optional[str] = None

    video_generation_worker: bool
    enable_kv_transfer: bool = False
    transfer_timeout_ms: int = 5000
    metadata_cache_ttl_s: int = 300
    max_pending_kv_transfers: int = 4
    max_nixl_starts_per_poll: int = 2
    enable_transfer_overlap: bool = False

    def validate(self) -> None:
        if (self.disagg_config is not None) ^ (self.disagg_config_key is not None):
            raise ValueError(
                "Both 'disagg_config' and 'disagg_config_key' must be provided together."
            )
