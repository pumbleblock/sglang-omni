# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang_omni.models.fishaudio_s2_pro.hypno_prompt import (
    build_fish_voice_clone_prompt_ids,
    strip_chatml_keep_tags,
)


class _FakeTok:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [len(text) + 1000]

    def get_vocab(self) -> dict[str, int]:
        return {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "<|audio_start|>": 3,
            "<|audio_end|>": 4,
            "<|voice|>": 5,
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.get_vocab().get(token, 99)


def test_strip_chatml_keep_tags() -> None:
    raw = (
        "<|im_start|>system\nsys<|im_end|>\n"
        "<|im_start|>assistant\n<|voice|>hello [Calm] world"
    )
    assert strip_chatml_keep_tags(raw) == "hello [Calm] world"


def test_build_fish_voice_clone_prompt_ids_shape() -> None:
    tok = _FakeTok()
    ids, norm_text, norm_ref = build_fish_voice_clone_prompt_ids(
        tok, "target line", "ref line", [200, 201]
    )
    assert len(ids) > 10
    assert "target" in norm_text
    assert "<|speaker:0|>" in norm_ref
