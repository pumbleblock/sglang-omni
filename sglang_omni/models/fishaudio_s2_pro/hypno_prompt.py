# SPDX-License-Identifier: Apache-2.0
"""Hypno-compatible Fish S2 prompt construction (matches vLLM-Omni / Hypno s10)."""

from __future__ import annotations

import re
from typing import Any

# Vendored from vllm_omni fish_speech prompt_utils (no vllm dependency).
FISH_CLONE_SYSTEM_PROMPT_PREFIX = (
    "convert the provided text to speech reference to the following:\n\nText:\n"
)
FISH_CLONE_SYSTEM_PROMPT_SUFFIX = "\n\nSpeech:\n"

_LEGACY_SPEAKER_TAG_PATTERN = re.compile(r"<speaker:(\d+)>")
_CANONICAL_SPEAKER_TAG_PATTERN = re.compile(r"<\|speaker:\d+\|>")
_CONTROL_TOKEN_PATTERN = re.compile(r"<\|[^>]+\|>")


def strip_chatml_keep_tags(text: str) -> str:
    """Match Hypno ``_strip_chatml_keep_tags`` (vllm_dualar runner)."""
    s = text.strip()
    if "<|im_start|>assistant" in s:
        s = s.split("<|im_start|>assistant", 1)[1]
    if "<|voice|>" in s:
        s = s.split("<|voice|>", 1)[1]
    for tok in ("<|im_start|>", "<|im_end|>", "<|text|>", "<|voice|>"):
        s = s.replace(tok, " ")
    return " ".join(s.strip().split())


def normalize_fish_speech_text(text: str, *, add_default_speaker: bool = False) -> str:
    normalized = _LEGACY_SPEAKER_TAG_PATTERN.sub(r"<|speaker:\1|>", text)
    disallowed = [
        t
        for t in _CONTROL_TOKEN_PATTERN.findall(normalized)
        if not _CANONICAL_SPEAKER_TAG_PATTERN.fullmatch(t)
    ]
    if disallowed:
        raise ValueError(
            f"Fish Speech input contains unsupported control token(s): {sorted(set(disallowed))}"
        )
    if add_default_speaker and not _CANONICAL_SPEAKER_TAG_PATTERN.search(normalized):
        normalized = f"<|speaker:0|>{normalized}"
    return normalized


def _encode_plain_text(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _encode_control_token(tokenizer: Any, token: str) -> list[int]:
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    token_id = vocab.get(token)
    if token_id is None:
        token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"Fish Speech tokenizer is missing required control token: {token}")
    return [int(token_id)]


def _build_message_prefix(tokenizer: Any, role: str) -> list[int]:
    return _encode_control_token(tokenizer, "<|im_start|>") + _encode_plain_text(
        tokenizer, f"{role}\n"
    )


def build_fish_voice_clone_prompt_ids(
    tokenizer: Any,
    text: str,
    ref_text: str,
    semantic_token_ids: list[int],
) -> tuple[list[int], str, str]:
    normalized_text = normalize_fish_speech_text(text)
    normalized_ref_text = normalize_fish_speech_text(ref_text, add_default_speaker=True)
    prompt_ids = (
        _build_message_prefix(tokenizer, "system")
        + _encode_plain_text(
            tokenizer,
            FISH_CLONE_SYSTEM_PROMPT_PREFIX
            + normalized_ref_text
            + FISH_CLONE_SYSTEM_PROMPT_SUFFIX,
        )
        + _encode_control_token(tokenizer, "<|audio_start|>")
        + [int(x) for x in semantic_token_ids]
        + _encode_control_token(tokenizer, "<|audio_end|>")
        + _encode_control_token(tokenizer, "<|im_end|>")
        + _encode_plain_text(tokenizer, "\n")
        + _build_message_prefix(tokenizer, "user")
        + _encode_plain_text(tokenizer, normalized_text)
        + _encode_control_token(tokenizer, "<|im_end|>")
        + _encode_plain_text(tokenizer, "\n")
        + _build_message_prefix(tokenizer, "assistant")
        + _encode_control_token(tokenizer, "<|voice|>")
    )
    return prompt_ids, normalized_text, normalized_ref_text


def build_hypno_prebuilt_inputs(
    tokenizer: Any,
    *,
    prompt: str,
    ref_text: str,
    ref_tokens: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.1,
    top_k: int = 30,
    num_codebooks: int = 10,
    codebook_size: int = 4096,
) -> dict:
    """Build preprocessing ``inputs`` dict for Hypno HTTP requests."""
    clean_text = strip_chatml_keep_tags(prompt)
    clean_ref_text = strip_chatml_keep_tags(ref_text)
    if not clean_text:
        raise ValueError("Hypno prompt is empty after ChatML sanitization")
    if not clean_ref_text:
        raise ValueError("Hypno ref_text is empty after ChatML sanitization")
    input_ids, _, _ = build_fish_voice_clone_prompt_ids(
        tokenizer,
        clean_text,
        clean_ref_text,
        [int(x) for x in ref_tokens],
    )
    return {
        "hypno_prebuilt": True,
        "input_ids": input_ids,
        "vq_mask_tokens": None,
        "vq_parts": None,
        "num_codebooks": int(num_codebooks),
        "codebook_size": int(codebook_size),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "repetition_penalty": float(repetition_penalty),
    }
