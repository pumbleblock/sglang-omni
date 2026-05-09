# SPDX-License-Identifier: Apache-2.0
"""Encoder-only ServerArgs helper.

Distinct from :mod:`server_args_builder` because encoder stages do not
have a meaningful ``context_length`` / ``mem_fraction_static`` /
running-request queue. See sglang-project/sglang-omni#375 design
("``build_sglang_encoder_server_args``" section) for the full rationale.
"""

from __future__ import annotations

from typing import Any

from sglang.srt.server_args import ServerArgs

# Worker invariants that MUST NOT be reachable from
# ``server_args_overrides``. Mutating them would either invalidate GPU
# placement, change the parallelism axis the runner already committed
# to, swap the encoder-vs-language-only fork, or reintroduce SGLang AR
# memory semantics into a path that doesn't have a KV pool. The set is
# checked once at the helper boundary so the failure surfaces with a
# clear "go set this through StageConfig" message instead of a deeper
# ``ServerArgs`` error.
_ENCODER_PROTECTED_KEYS: frozenset[str] = frozenset(
    {
        # Parallelism / placement
        "tp_size",
        "pp_size",
        "dp_size",
        "moe_dp_size",
        "ep_size",
        "attn_cp_size",
        "moe_dense_tp_size",
        "nnodes",
        "node_rank",
        "rank",
        "world_size",
        "tp_rank",
        "gpu_id",
        "base_gpu_id",
        "nccl_port",
        "dist_init_addr",
        # Encoder-only fork
        "encoder_only",
        "language_only",
        "mm_enable_dp_encoder",
        "enable_dp_attention",
        "enable_dp_attention_local_control_broadcast",
        "enable_dp_lm_head",
        "disable_cuda_graph",
        "device",
        # AR-only knobs that have no meaning for an encoder-only worker
        "mem_fraction_static",
        "max_running_requests",
        "max_prefill_tokens",
        "chunked_prefill_size",
        "context_length",
    }
)


def build_sglang_encoder_server_args(
    model_path: str,
    *,
    tp_size: int,
    base_gpu_id: int,
    dist_init_addr: str,
    dtype: str | None = None,
    load_format: str | None = None,
    **overrides: Any,
) -> ServerArgs:
    """Construct ``ServerArgs`` for an encoder-only worker.

    Args:
        model_path: HF model path / local snapshot.
        tp_size: TP world size for this encoder process group.
        base_gpu_id: Local CUDA device index inside the worker process.
            With per-process ``CUDA_VISIBLE_DEVICES`` remapping in place
            (see ``stage_process._prepare_cuda_environment``) this is
            always 0.
        dist_init_addr: ``host:port`` form for SGLang's
            ``ServerArgs.dist_init_addr``. The ``tcp://`` URL form torch
            ``init_distributed_environment`` expects is composed by the
            worker.
        dtype: Optional dtype hint forwarded to ``ServerArgs.dtype``.
        load_format: Optional weight-loader override.
        **overrides: Additional ``ServerArgs`` kwargs. Loader / processor
            knobs (``model_loader_extra_config``,
            ``remote_instance_weight_loader_*``,
            ``disable_fast_image_processor``, …) flow through
            unchanged. Topology / encoder-fork / AR-only keys are
            rejected — pass them through ``StageConfig`` instead.

    Raises:
        ValueError: If ``overrides`` tries to mutate a protected key.
    """
    bad = sorted(_ENCODER_PROTECTED_KEYS & overrides.keys())
    if bad:
        raise ValueError(
            f"server_args_overrides cannot override protected keys: {bad}. "
            f"These are decided by the worker / pipeline runner; pass them "
            f"through StageConfig (tp_size, gpu) or the worker's explicit "
            f"keyword arguments (dtype, load_format) instead."
        )

    kwargs: dict[str, Any] = {
        "model_path": model_path,
        "trust_remote_code": True,
        "tp_size": tp_size,
        "pp_size": 1,
        "base_gpu_id": base_gpu_id,
        "dist_init_addr": dist_init_addr,
        "encoder_only": True,
        "language_only": False,
        # MVP: TP only. DP encoder coupling with text-side TP is a Phase
        # 2+ workstream — see #375 Open Questions.
        "mm_enable_dp_encoder": False,
        # Encoder activations have variable shapes (per-image grids,
        # per-audio mel lengths). Piecewise CUDA graph for the
        # encoder ViT lands later.
        "disable_cuda_graph": True,
        "random_seed": 123,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    if load_format is not None:
        kwargs["load_format"] = load_format
    kwargs.update(overrides)
    return ServerArgs(**kwargs)


__all__ = [
    "_ENCODER_PROTECTED_KEYS",
    "build_sglang_encoder_server_args",
]
