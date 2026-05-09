# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni encoder adapters for the SGLang-native worker.

Translates between v1 ``PipelineState.encoder_inputs`` /
``encoder_outs`` and upstream SGLang
``MultimodalDataItem`` / ``thinker.get_*_feature(items)`` signatures.

Two adapters live here — both share the same
``build_batch / run_feature / slice_results`` contract that
:class:`EncoderScheduler` calls into. The image and audio encoders are
distinct stages with distinct cost models, but use the same
``EncoderRequestData`` round-trip helpers
(:func:`build_encoder_request` / :func:`apply_encoder_result`) the local
v1 path already exercises.

Key invariants:

- **Skip semantics from preprocessing.** The preprocessor stamps each
  encoder stage's ``encoder_inputs`` slot with one of three shapes:
  active inputs, ``{"_skip": True, "_result": {}}`` (modality absent),
  or ``cache_key`` metadata. The adapter must respect ``_skip``
  bit-for-bit — the very common image-only request would otherwise
  ``KeyError`` on ``inputs["input_features"]`` inside the audio
  encoder stage. ``run_feature`` is a no-op when the entire batch is
  skip-only.
- **No cache in Phase 0.** ``cache_key`` is preserved in the
  ``RequestSpan`` for Phase 2 wiring but is not consulted today: a
  TP-aware cache requires the entry rank to broadcast hit/miss
  before the broadcast, otherwise different ranks would compose
  divergent broadcast structures. See sglang-project/sglang-omni#375
  ("Skip / cache semantics", "TP-aware encoder cache" open question).
- **Cost-fn metadata from HF config, not the SGLang wrapper.** The
  upstream ``Qwen3VLMoeVisionModel.__init__`` writes
  ``self.out_hidden_size = vision_config.out_hidden_size * (1 + len(deepstack_visual_indexes))``
  — deepstack is **already folded** into the wrapper's
  ``out_hidden_size``. Reading the wrapper would double-count
  deepstack and starve long-video batches. The adapter takes its
  cost metadata from the HF ``vision_config`` directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from sglang_omni_v1.models.qwen3_omni.payload_types import PipelineState
from sglang_omni_v1.models.qwen3_omni.request_builders import (
    AUDIO_STAGE,
    IMAGE_STAGE,
    EncoderRequestData,
    apply_encoder_result,
    build_encoder_request,
)
from sglang_omni_v1.proto import StagePayload
from sglang_omni_v1.scheduling.messages import IncomingMessage

logger = logging.getLogger(__name__)


# Inherited from the local v1 image-encoder path. Keep both the budget
# and the activation multiplier identical so admission control matches
# the cost model the local path was tuned against.
QWEN3_IMAGE_ENCODER_BATCH_BUDGET_BYTES = 10 * 1024**3
QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER = 5


# ---------------------------------------------------------------------------
# BatchPlan + RequestSpan
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RequestSpan:
    """Slot for one request inside a :class:`BatchPlan`.

    Exactly one of ``skip_result`` or the active-payload fields is
    populated. ``skip_result`` carries the preprocessor's pre-computed
    output dict so :meth:`EncoderAdapter.slice_results` can write it
    back into pipeline state without ever calling the encoder.
    """

    request_id: str
    skip_result: dict[str, Any] | None = None
    cache_key: str | None = None

    # Visual
    image_rows: int = 0
    video_rows: int = 0
    image_token_count: int = 0
    video_token_count: int = 0

    # Audio — keep both unpadded per-request lengths (for merge_for_thinker)
    # and the downsampled output lengths (for slicing the encoder output
    # along the token axis).
    audio_feature_lengths: torch.Tensor | None = None
    audio_output_lengths: torch.Tensor | None = None
    audio_rows: int = 0


@dataclass(slots=True)
class BatchPlan:
    """Deterministic, rank-equal description of one encoder forward.

    ``image_items`` / ``video_items`` / ``audio_items`` are flat across
    active requests only — skip-only requests contribute nothing to
    them but still appear in ``spans`` so :meth:`slice_results` can
    write their ``skip_result`` back.
    """

    adapter: Any
    image_items: list[Any] = field(default_factory=list)
    video_items: list[Any] = field(default_factory=list)
    audio_items: list[Any] = field(default_factory=list)
    spans: list[RequestSpan] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.image_items or self.video_items or self.audio_items)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _payload_with_state(payload: StagePayload, state: PipelineState) -> StagePayload:
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def _request_from_message(
    msg: IncomingMessage, *, stage_name: str
) -> tuple[EncoderRequestData, StagePayload, PipelineState]:
    payload: StagePayload = msg.data
    state = PipelineState.from_dict(payload.data)
    request = build_encoder_request(state, stage_name=stage_name)
    return request, payload, state


def _tensor_bytes(value: Any) -> int:
    if not isinstance(value, torch.Tensor):
        return 0
    return int(value.numel() * value.element_size())


def _grid_visual_tokens(grid: Any, merge: int) -> int:
    if not isinstance(grid, torch.Tensor) or grid.numel() == 0:
        return 0
    # Cost is computed before any GPU staging, so the grid must live on
    # CPU. The local v1 path enforces the same invariant — see
    # sglang_omni/models/qwen3_omni/pipeline/visual_budget.py.
    if grid.device.type != "cpu":
        grid = grid.cpu()
    return int((grid.to(dtype=torch.long).prod(dim=-1) // merge).sum().item())


# ---------------------------------------------------------------------------
# Image / video adapter
# ---------------------------------------------------------------------------


def _ensure_multimodal_imports():
    """Import upstream multimodal types lazily.

    Keeps this module importable in unit tests that stub sglang.
    """
    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

    return Modality, MultimodalDataItem


@dataclass
class Qwen3OmniImageEncoderAdapter:
    """Adapter for the image (visual) encoder stage.

    Args:
        thinker_config: The HF ``thinker_config`` (typically loaded from
            ``AutoConfig`` of the parent omni model). Used both at
            ``build_batch`` time (for ``spatial_merge_size``) and inside
            :meth:`request_cost_fn` (for activation byte estimation).
        dtype: dtype the encoder will run in. Used by the cost fn to
            compute output byte size; defaults to ``torch.bfloat16``
            which matches the encoder default. Override to match a
            different ``ServerArgs.dtype`` for cost accuracy.
    """

    thinker_config: Any
    dtype: torch.dtype = torch.bfloat16
    stage_name: str = IMAGE_STAGE

    def __post_init__(self) -> None:
        vision_cfg = self.thinker_config.vision_config
        self._merge = int(vision_cfg.spatial_merge_size) ** 2
        # Take the *base* hidden size from the HF config — the SGLang
        # wrapper folds deepstack into its own ``out_hidden_size``.
        self._base_hidden = int(vision_cfg.out_hidden_size)
        self._output_layers = 1 + len(vision_cfg.deepstack_visual_indexes)
        self._dtype_bytes = torch.empty((), dtype=self.dtype).element_size()

    # -- Cost fn (admission control) ---------------------------------------

    def request_cost_fn(self, payload: StagePayload) -> int:
        state = PipelineState.from_dict(payload.data)
        request = build_encoder_request(state, stage_name=self.stage_name)
        if request.skip_result is not None:
            return 0
        inputs = request.model_inputs
        raw_bytes = _tensor_bytes(inputs.get("pixel_values"))
        raw_bytes += _tensor_bytes(inputs.get("pixel_values_videos"))
        visual_tokens = _grid_visual_tokens(inputs.get("image_grid_thw"), self._merge)
        visual_tokens += _grid_visual_tokens(inputs.get("video_grid_thw"), self._merge)
        output_bytes = (
            visual_tokens * self._base_hidden * self._dtype_bytes * self._output_layers
        )
        return (raw_bytes + output_bytes) * QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER

    # -- build_batch -------------------------------------------------------

    def build_batch(self, messages: list[IncomingMessage]) -> BatchPlan:
        Modality, MultimodalDataItem = _ensure_multimodal_imports()

        plan = BatchPlan(adapter=self)
        for msg in messages:
            request, _payload, _state = _request_from_message(
                msg, stage_name=self.stage_name
            )
            if request.skip_result is not None:
                plan.spans.append(
                    RequestSpan(
                        request_id=msg.request_id,
                        skip_result=request.skip_result,
                        cache_key=request.cache_key,
                    )
                )
                continue

            inputs = request.model_inputs
            n_img = n_vid = 0
            img_tokens = vid_tokens = 0

            pixel_values = inputs.get("pixel_values")
            image_grid_thw = inputs.get("image_grid_thw")
            if isinstance(pixel_values, torch.Tensor) and isinstance(
                image_grid_thw, torch.Tensor
            ):
                item = MultimodalDataItem(modality=Modality.IMAGE, feature=pixel_values)
                item.image_grid_thw = image_grid_thw
                plan.image_items.append(item)
                n_img = int(image_grid_thw.shape[0])
                img_tokens = int(
                    (image_grid_thw.to(dtype=torch.long).prod(-1) // self._merge)
                    .sum()
                    .item()
                )

            pixel_values_videos = inputs.get("pixel_values_videos")
            video_grid_thw = inputs.get("video_grid_thw")
            if isinstance(pixel_values_videos, torch.Tensor) and isinstance(
                video_grid_thw, torch.Tensor
            ):
                item = MultimodalDataItem(
                    modality=Modality.VIDEO, feature=pixel_values_videos
                )
                item.video_grid_thw = video_grid_thw
                plan.video_items.append(item)
                n_vid = int(video_grid_thw.shape[0])
                vid_tokens = int(
                    (video_grid_thw.to(dtype=torch.long).prod(-1) // self._merge)
                    .sum()
                    .item()
                )

            plan.spans.append(
                RequestSpan(
                    request_id=msg.request_id,
                    cache_key=request.cache_key,
                    image_rows=n_img,
                    video_rows=n_vid,
                    image_token_count=img_tokens,
                    video_token_count=vid_tokens,
                )
            )
        return plan

    # -- run_feature -------------------------------------------------------

    def run_feature(
        self, model: Any, plan: BatchPlan
    ) -> dict[str, torch.Tensor | None]:
        if plan.is_empty:
            return {"image": None, "video": None, "audio": None}

        image_embed = (
            model.thinker.get_image_feature(plan.image_items)
            if plan.image_items
            else None
        )
        video_embed = (
            model.thinker.get_video_feature(plan.video_items)
            if plan.video_items
            else None
        )
        return {"image": image_embed, "video": video_embed, "audio": None}

    # -- slice_results -----------------------------------------------------

    def slice_results(
        self,
        raw: dict[str, torch.Tensor | None],
        plan: BatchPlan,
        messages: list[IncomingMessage],
    ) -> list[StagePayload]:
        # The upstream visual return tensor has trailing-dim
        # ``out_hidden_size * (1 + len(deepstack_visual_indexes))``;
        # split it back into a base + deepstack list so the merge_for_thinker
        # contract that the local path observes is preserved.
        out: list[StagePayload] = []
        img_token_cursor = 0
        vid_token_cursor = 0

        image_raw = raw.get("image")
        video_raw = raw.get("video")
        image_base, image_deepstack = _split_visual_output(
            image_raw, self._base_hidden, self._output_layers
        )
        video_base, video_deepstack = _split_visual_output(
            video_raw, self._base_hidden, self._output_layers
        )

        for span, msg in zip(plan.spans, messages):
            payload: StagePayload = msg.data
            state = PipelineState.from_dict(payload.data)
            if span.skip_result is not None:
                apply_encoder_result(
                    state, stage_name=self.stage_name, result=span.skip_result
                )
                out.append(_payload_with_state(payload, state))
                continue

            result: dict[str, Any] = {}
            if span.image_token_count:
                end = img_token_cursor + span.image_token_count
                result["image_embeds"] = (
                    image_base[img_token_cursor:end] if image_base is not None else None
                )
                if image_deepstack is not None:
                    result["deepstack_visual_embeds_image"] = [
                        ds[img_token_cursor:end] for ds in image_deepstack
                    ]
                img_token_cursor = end

            if span.video_token_count:
                end = vid_token_cursor + span.video_token_count
                result["video_embeds"] = (
                    video_base[vid_token_cursor:end] if video_base is not None else None
                )
                if video_deepstack is not None:
                    result["deepstack_visual_embeds_video"] = [
                        ds[vid_token_cursor:end] for ds in video_deepstack
                    ]
                vid_token_cursor = end

            apply_encoder_result(state, stage_name=self.stage_name, result=result)
            out.append(_payload_with_state(payload, state))
        return out


def _split_visual_output(
    tensor: torch.Tensor | None,
    base_hidden: int,
    output_layers: int,
) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
    """Split ``[N, base_hidden * output_layers]`` back into base + deepstack."""
    if tensor is None:
        return None, None
    if output_layers <= 1:
        return tensor, None
    parts = tensor.split(base_hidden, dim=-1)
    base = parts[0]
    deepstack = list(parts[1:])
    return base, deepstack


# ---------------------------------------------------------------------------
# Audio adapter
# ---------------------------------------------------------------------------


def _normalize_audio_request_tensors(
    request: EncoderRequestData,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(features, mask, lengths)`` ready for batched cat.

    Mirrors the v1 local helper in
    :mod:`sglang_omni_v1.models.qwen3_omni.stages` with one fix
    required by the SGLang Plan B path: the synthesized fallback mask
    allocates ``arange`` on ``lengths.device`` instead of CPU, so it
    works regardless of whether the entry rank already moved the
    request tensors to GPU.
    """
    inputs = request.model_inputs
    features = inputs["input_features"]
    if features.ndim == 2:
        features = features.unsqueeze(0)

    lengths = inputs.get("audio_feature_lengths")
    mask = inputs.get("feature_attention_mask")
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.to(dtype=torch.long).view(-1)
    elif isinstance(mask, torch.Tensor):
        lengths = mask.to(dtype=torch.long).sum(dim=1).view(-1)
    else:
        raise ValueError("audio_feature_lengths or feature_attention_mask is required")

    time_dim = features.shape[-1]
    if isinstance(mask, torch.Tensor):
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask = mask.to(dtype=torch.bool)
    else:
        # Use lengths.device — mixing CPU steps with GPU lengths would
        # raise a device-mismatch error inside the SGLang path.
        steps = torch.arange(
            time_dim, dtype=torch.long, device=lengths.device
        ).unsqueeze(0)
        mask = steps < lengths.unsqueeze(1)

    return features, mask, lengths


def _pad_audio_features(features: torch.Tensor, target_time: int) -> torch.Tensor:
    pad = target_time - int(features.shape[-1])
    if pad <= 0:
        return features
    return F.pad(features, (0, pad))


def _pad_audio_mask(mask: torch.Tensor, target_time: int) -> torch.Tensor:
    pad = target_time - int(mask.shape[-1])
    if pad <= 0:
        return mask
    return F.pad(mask, (0, pad), value=False)


@dataclass
class Qwen3OmniAudioEncoderAdapter:
    """Adapter for the audio encoder stage."""

    thinker_config: Any | None = None
    stage_name: str = AUDIO_STAGE

    # No v1 audio cost model exists yet — leave admission control
    # uncapped to match the local audio path.
    def request_cost_fn(self, payload: StagePayload) -> int:
        return 0

    def build_batch(self, messages: list[IncomingMessage]) -> BatchPlan:
        Modality, MultimodalDataItem = _ensure_multimodal_imports()
        plan = BatchPlan(adapter=self)
        normalized: list[dict[str, Any]] = []

        for msg in messages:
            request, _payload, _state = _request_from_message(
                msg, stage_name=self.stage_name
            )
            if request.skip_result is not None:
                plan.spans.append(
                    RequestSpan(
                        request_id=msg.request_id,
                        skip_result=request.skip_result,
                        cache_key=request.cache_key,
                    )
                )
                continue

            features, mask, lengths = _normalize_audio_request_tensors(request)
            from sglang.srt.models.qwen3_omni_moe import (
                _get_feat_extract_output_lengths,
            )

            output_lengths = _get_feat_extract_output_lengths(lengths)
            plan.spans.append(
                RequestSpan(
                    request_id=msg.request_id,
                    cache_key=request.cache_key,
                    audio_rows=int(lengths.shape[0]),
                    audio_feature_lengths=lengths,
                    audio_output_lengths=output_lengths,
                )
            )
            normalized.append({"features": features, "mask": mask, "lengths": lengths})

        if not normalized:
            return plan

        # Pad to the batch-wide max time so the upstream
        # ``get_audio_feature`` ``torch.cat`` on dim=0 succeeds. Padded
        # positions are masked out by ``feature_attention_mask`` and
        # discarded inside ``get_audio_feature`` via
        # ``input_features.permute(0,2,1)[mask.bool()]``.
        max_time = max(int(item["features"].shape[-1]) for item in normalized)
        for item in normalized:
            feat = _pad_audio_features(item["features"], max_time)
            m = _pad_audio_mask(item["mask"], max_time)
            mm = MultimodalDataItem(modality=Modality.AUDIO, feature=feat)
            mm.feature_attention_mask = m
            mm.audio_feature_lengths = item["lengths"]
            plan.audio_items.append(mm)
        return plan

    def run_feature(
        self, model: Any, plan: BatchPlan
    ) -> dict[str, torch.Tensor | None]:
        if plan.is_empty:
            return {"image": None, "video": None, "audio": None}
        embed = model.thinker.get_audio_feature(plan.audio_items)
        return {"image": None, "video": None, "audio": embed}

    def slice_results(
        self,
        raw: dict[str, torch.Tensor | None],
        plan: BatchPlan,
        messages: list[IncomingMessage],
    ) -> list[StagePayload]:
        out: list[StagePayload] = []
        token_cursor = 0
        audio_raw = raw.get("audio")

        for span, msg in zip(plan.spans, messages):
            payload: StagePayload = msg.data
            state = PipelineState.from_dict(payload.data)
            if span.skip_result is not None:
                apply_encoder_result(
                    state, stage_name=self.stage_name, result=span.skip_result
                )
                out.append(_payload_with_state(payload, state))
                continue

            assert (
                span.audio_output_lengths is not None
            ), "audio span built without output_lengths"
            token_end = token_cursor + int(span.audio_output_lengths.sum().item())
            result = {
                "audio_embeds": (
                    audio_raw[token_cursor:token_end] if audio_raw is not None else None
                ),
                # Unpadded per-request lengths preserved at build_batch time.
                # merge_for_thinker uses these — never the batch-wide
                # padded ones.
                "audio_feature_lengths": span.audio_feature_lengths,
                "audio_output_lengths": span.audio_output_lengths,
            }
            apply_encoder_result(state, stage_name=self.stage_name, result=result)
            out.append(_payload_with_state(payload, state))
            token_cursor = token_end
        return out


__all__ = [
    "BatchPlan",
    "QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER",
    "QWEN3_IMAGE_ENCODER_BATCH_BUDGET_BYTES",
    "Qwen3OmniAudioEncoderAdapter",
    "Qwen3OmniImageEncoderAdapter",
    "RequestSpan",
]
