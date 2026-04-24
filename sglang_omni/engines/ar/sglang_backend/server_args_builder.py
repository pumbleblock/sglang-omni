# SPDX-License-Identifier: Apache-2.0
"""Shared ServerArgs construction for SGLang AR engines."""
from __future__ import annotations

from typing import Any

from sglang.srt.server_args import ServerArgs


def build_sglang_server_args(
    model_path: str,
    context_length: int,
    *,
    chunked_prefill_size: int = 128,
    max_prefill_tokens: int = 4096,
    max_running_requests: int = 16,
    mem_fraction_static: float | None = None,
    auto_mem_fraction_static_reserve: float | None = None,
    **overrides: Any,
) -> ServerArgs:
    """Build ServerArgs with shared defaults for all SGLang AR engines.

    When ``mem_fraction_static`` is ``None`` and
    ``auto_mem_fraction_static_reserve`` is positive, the reserve is
    subtracted from SGLang's auto-selected ``mem_fraction_static``. If
    that would leave less than 0.01, we raise instead of silently
    underfunding SGLang's KV pool.
    """
    kwargs: dict[str, Any] = {
        "model_path": model_path,
        "trust_remote_code": True,
        "tp_size": 1,
        "pp_size": 1,
        "disable_cuda_graph": True,
        "chunked_prefill_size": chunked_prefill_size,
        "max_prefill_tokens": max_prefill_tokens,
        "max_running_requests": max_running_requests,
        "random_seed": 123,
        "context_length": context_length,
    }
    if mem_fraction_static is not None:
        kwargs["mem_fraction_static"] = mem_fraction_static
    kwargs.update(overrides)
    server_args = ServerArgs(**kwargs)
    _apply_auto_mem_fraction_static_reserve(
        server_args,
        enabled=auto_mem_fraction_static_reserve is not None,
        user_mem_fraction_static=mem_fraction_static,
        reserve=auto_mem_fraction_static_reserve or 0.0,
    )
    return server_args


# Minimum ``mem_fraction_static`` SGLang can still boot with on a typical
# Qwen3-Omni-30B thinker: below this the KV allocator fails deep in the
# scheduler with a confusing stack trace, so raise early at build time
# instead. 0.05 was too permissive (SGLang crashes ~0.08 on H200 for 30B).
_MIN_MEM_FRACTION_STATIC_AFTER_RESERVE = 0.1


def _apply_auto_mem_fraction_static_reserve(
    server_args: ServerArgs,
    *,
    enabled: bool,
    user_mem_fraction_static: float | None,
    reserve: float,
) -> None:
    """Subtract a caller-requested reserve from SGLang's auto-selected value.

    Raises ``ValueError`` when the resulting ``mem_fraction_static`` would
    fall below ``_MIN_MEM_FRACTION_STATIC_AFTER_RESERVE`` — otherwise SGLang
    fails deep inside KV allocation with a confusing traceback.
    """
    if not enabled or user_mem_fraction_static is not None:
        return
    if reserve <= 0:
        return

    current = server_args.mem_fraction_static
    if current is None:
        return
    new_value = current - reserve
    if new_value < _MIN_MEM_FRACTION_STATIC_AFTER_RESERVE:
        raise ValueError(
            f"auto mem_fraction_static {current:.3f} minus encoder_mem_reserve "
            f"{reserve:.3f} = {new_value:.3f} is below the safe floor "
            f"{_MIN_MEM_FRACTION_STATIC_AFTER_RESERVE}; lower encoder_mem_reserve "
            f"or pin --mem-fraction-static explicitly."
        )
    server_args.mem_fraction_static = round(new_value, 3)
