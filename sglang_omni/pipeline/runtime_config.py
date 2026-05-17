# SPDX-License-Identifier: Apache-2.0
"""Pipeline runtime preparation and relay config helpers."""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.config.placement import StagePlacementPlan, build_stage_placement_plan
from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.pipeline.endpoints import (
    IpcRuntimeDir,
    allocate_endpoints,
    create_ipc_runtime_dir,
)


@dataclass(frozen=True)
class PipelineRuntimePrep:
    """Prepared stage, endpoint, and placement state for one runner."""

    stages_cfg: list[StageConfig]
    name_map: dict[str, str]
    entry_stage: str
    endpoints: dict[str, str]
    placement_plan: StagePlacementPlan
    runtime_dir: IpcRuntimeDir | None
    runtime_dir_created_here: bool


def prepare_pipeline_runtime(
    config: PipelineConfig,
    *,
    ipc_runtime_dir: IpcRuntimeDir | None = None,
) -> PipelineRuntimePrep:
    """Prepare fused stages, placement, endpoints, and optional IPC runtime dir."""

    runtime_dir = ipc_runtime_dir
    created_runtime_dir = None
    if runtime_dir is None:
        runtime_dir = create_ipc_runtime_dir(config)
        created_runtime_dir = runtime_dir

    try:
        stages_cfg, name_map, entry_stage = config.apply_fusion()
        placement_plan = build_stage_placement_plan(config, stages_cfg=stages_cfg)
        endpoints = allocate_endpoints(
            config,
            stages=stages_cfg,
            ipc_base_dir=runtime_dir.path if runtime_dir else None,
        )
    except Exception:
        if created_runtime_dir is not None:
            created_runtime_dir.close()
        raise

    return PipelineRuntimePrep(
        stages_cfg=stages_cfg,
        name_map=name_map,
        entry_stage=entry_stage,
        endpoints=endpoints,
        placement_plan=placement_plan,
        runtime_dir=runtime_dir,
        runtime_dir_created_here=created_runtime_dir is not None,
    )


def build_relay_config(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
) -> dict[str, object]:
    relay_cfg = stage_cfg.relay
    if relay_cfg is not None:
        return {
            "relay_type": global_cfg.relay_backend,
            "slot_size_mb": relay_cfg.slot_size_mb,
            "credits": relay_cfg.credits,
            "rank": relay_cfg.rank,
            "world_size": relay_cfg.world_size,
            "gpu_id": parse_gpu_id(relay_cfg.device),
        }

    if global_cfg.relay_backend == "shm":
        gpu_id = None
    else:
        gpu = stage_cfg.gpu
        if gpu is None:
            gpu_id = None
        elif isinstance(gpu, list):
            gpu_id = gpu[0]
        else:
            gpu_id = gpu

    return {
        "relay_type": global_cfg.relay_backend,
        "slot_size_mb": 512,
        "credits": 2,
        "rank": None,
        "world_size": None,
        "gpu_id": gpu_id,
    }


def parse_gpu_id(device: str) -> int | None:
    if device == "cpu":
        return None
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"Unsupported device string: {device}")
