# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni_v1.cli.serve import apply_mem_fraction_cli_overrides
from sglang_omni_v1.config import PipelineConfig
from sglang_omni_v1.config.compiler import _resolve_factory_args
from sglang_omni_v1.models.qwen3_omni.config import (
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)
from sglang_omni_v1.models.qwen3_omni.merge import decode_events, merge_for_thinker
from sglang_omni_v1.models.qwen3_omni.payload_types import PipelineState
from sglang_omni_v1.models.qwen3_omni.request_builders import (
    build_sglang_thinker_request,
    project_preprocessing_to_mm_aggregate,
)
from tests.unit_test.fixtures.qwen_fakes import (
    FakeQwenTokenizer,
    make_qwen_payload,
    make_qwen_state,
)


def _stage(config: PipelineConfig, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def _server_args_overrides(config: PipelineConfig, name: str) -> dict[str, object]:
    return _stage(config, name).factory_args.get("server_args_overrides", {})


def test_qwen_pipeline_config_and_state_contracts() -> None:
    """Preserves Qwen text/speech topology and PipelineState coercion behavior."""
    text_config = Qwen3OmniPipelineConfig(model_path="model")
    speech_config = Qwen3OmniSpeechPipelineConfig(model_path="model")

    assert [stage.name for stage in text_config.stages] == [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "mm_aggregate",
        "thinker",
        "decode",
    ]
    assert speech_config.terminal_stages == ["decode", "code2wav"]
    assert {stage.name: stage for stage in speech_config.stages}[
        "thinker"
    ].stream_to == ["talker_ar"]

    state = PipelineState.from_dict(
        {
            "prompt": {"input_ids": torch.tensor([1, 2]), "prompt_text": "hi"},
            "mm_inputs": "bad",
            "encoder_inputs": {"image_encoder": {"cache_key": "img"}},
            "thinker_out": {"output_ids": [3], "is_final": True},
        }
    )
    assert torch.equal(state.prompt["input_ids"], torch.tensor([1, 2]))
    assert state.mm_inputs == {}
    assert state.encoder_inputs["image_encoder"]["cache_key"] == "img"
    assert state.thinker_out["is_final"] is True


def test_qwen_cli_mem_fraction_precedence_targets_only_ar_stages() -> None:
    """Preserves per-role CLI memory overrides for Qwen AR stages only."""
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=0.70,
        talker_mem_fraction_static=None,
    )

    assert _server_args_overrides(config, "thinker")["mem_fraction_static"] == 0.70
    assert _server_args_overrides(config, "talker_ar")["mem_fraction_static"] == 0.80
    for non_ar_stage in ("image_encoder", "audio_encoder", "code2wav"):
        assert "server_args_overrides" not in _stage(config, non_ar_stage).factory_args


def test_qwen_cli_mem_fraction_survives_runtime_overrides_overlay() -> None:
    """Preserves CLI memory settings when compiler overlays runtime overrides."""
    config = Qwen3OmniSpeechPipelineConfig(
        model_path="dummy",
        runtime_overrides={
            "thinker": {
                "server_args_overrides": {"disable_cuda_graph": True},
            }
        },
    )

    apply_mem_fraction_cli_overrides(
        config,
        mem_fraction_static=0.80,
        thinker_mem_fraction_static=None,
        talker_mem_fraction_static=None,
    )

    resolved = _resolve_factory_args(_stage(config, "thinker"), config)
    assert resolved["server_args_overrides"]["mem_fraction_static"] == 0.80
    assert resolved["server_args_overrides"]["disable_cuda_graph"] is True


def test_qwen_thinker_encoder_reserve_auto_path_vs_explicit_pin() -> None:
    """Preserves Qwen thinker reserve adjustment only for auto memory sizing."""
    import sglang_omni_v1.models.qwen3_omni.stages as qwen_stages

    auto_args = SimpleNamespace(mem_fraction_static=0.929)
    pinned_args = SimpleNamespace(mem_fraction_static=0.70)

    assert (
        qwen_stages._apply_qwen_thinker_encoder_reserve(
            auto_args,
            has_explicit_mem_fraction_static=False,
            encoder_mem_reserve=0.05,
        )
        is True
    )
    assert auto_args.mem_fraction_static == 0.879

    assert (
        qwen_stages._apply_qwen_thinker_encoder_reserve(
            pinned_args,
            has_explicit_mem_fraction_static=True,
            encoder_mem_reserve=0.20,
        )
        is False
    )
    assert pinned_args.mem_fraction_static == 0.70

    with pytest.raises(ValueError, match="below the safe floor"):
        qwen_stages._apply_qwen_thinker_encoder_reserve(
            SimpleNamespace(mem_fraction_static=0.15),
            has_explicit_mem_fraction_static=False,
            encoder_mem_reserve=0.10,
        )


def test_qwen_mm_aggregate_keeps_lightweight_inputs_and_prunes_after_merge() -> None:
    """Preserves lightweight fan-in payloads and prunes consumed encoder tensors."""
    state = make_qwen_state(
        mm_inputs={
            "image": {
                "pixel_values": torch.ones((2, 3)),
                "image_grid_thw": torch.tensor([[1, 1, 2]]),
            },
            "audio": {"audio_feature_lengths": torch.tensor([1])},
        },
        encoder_inputs={"image_encoder": {"cache_key": "image-cache"}},
    )

    projected = project_preprocessing_to_mm_aggregate(make_qwen_payload(state))
    projected_state = PipelineState.from_dict(projected.data)
    assert "pixel_values" not in projected_state.mm_inputs["image"]
    assert projected_state.encoder_inputs == {
        "image_encoder": {"cache_key": "image-cache"}
    }

    image_state = PipelineState(
        encoder_outs={"image_encoder": {"image_embeds": torch.ones((2, 2))}}
    )
    merged = merge_for_thinker(
        {
            "preprocessing": make_qwen_payload(state),
            "image_encoder": make_qwen_payload(image_state),
        }
    )
    merged_state = PipelineState.from_dict(merged.data)
    assert merged_state.encoder_inputs == {}
    assert merged_state.encoder_outs == {}
    assert "image_embeds" in merged_state.thinker_inputs["model_inputs"]
    assert merged_state.thinker_inputs["media_cache_keys"] == {
        "image": "image:image-cache",
        "video": "video:image-cache",
    }


def test_qwen_thinker_request_and_decode_contracts() -> None:
    """Preserves incremental text deltas, replacement-char suppression, and final text."""
    stream_state = PipelineState()
    tokenizer = FakeQwenTokenizer(pieces={1: "A", 2: "\ufffd", 3: "B"})
    first = list(
        decode_events(
            thinker_out={"output_ids": [1]},
            state=stream_state,
            tokenizer=tokenizer,
            eos_token_id=99,
            step=1,
        )
    )
    dropped = list(
        decode_events(
            thinker_out={"output_ids": [2]},
            state=stream_state,
            tokenizer=tokenizer,
            eos_token_id=99,
            step=2,
        )
    )
    final = list(
        decode_events(
            thinker_out={"output_ids": [1, 3, 99], "is_final": True},
            state=stream_state,
            tokenizer=FakeQwenTokenizer(pieces={1: "A", 3: "B"}),
            eos_token_id=99,
            step=3,
        )
    )
    assert first[0].payload == {"text": "A"}
    assert dropped == []
    assert final[0].type == "text_final"
    assert final[0].payload == {"text": "AB"}


def test_qwen_sglang_request_hashes_media_tokens_without_changing_mrope_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves hashed media pad tokens while M-RoPE still sees original ids."""
    captured: dict[str, torch.Tensor] = {}

    def fake_mrope(input_ids, model_inputs, thinker_config):
        del model_inputs, thinker_config
        captured["input_ids"] = input_ids.clone()
        return torch.zeros((3, input_ids.numel()), dtype=torch.long), torch.tensor(0)

    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )
    monkeypatch.setattr(
        "sglang_omni_v1.models.qwen3_omni.request_builders._compute_mrope_positions",
        fake_mrope,
    )

    audio_token_id = 77
    input_ids = torch.tensor([10, audio_token_id, 11], dtype=torch.long)
    state = make_qwen_state(
        prompt={"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)},
        thinker_inputs={
            "model_inputs": {"audio_embeds": torch.ones((1, 4))},
            "media_cache_keys": {"audio": "audio:cache"},
        },
    )
    req_data = build_sglang_thinker_request(
        state,
        params={"max_new_tokens": 3, "seed": 123},
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
        request_id="rid-1",
        thinker_config=SimpleNamespace(
            image_token_id=55,
            video_token_id=66,
            audio_token_id=audio_token_id,
        ),
    )

    pad_values = req_data.req.omni_model_inputs["pad_values"]
    assert pad_values["audio"] >= 256
    assert int(req_data.input_ids[1]) == pad_values["audio"]
    assert captured["input_ids"].tolist() == input_ids.tolist()
