# SPDX-License-Identifier: Apache-2.0
"""Hypno-compatible Fish S2 semantic (CB0) generation over HTTP."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerFast

from sglang_omni.client import Client, GenerateRequest, SamplingParams
from sglang_omni.models.fishaudio_s2_pro.hypno_prompt import build_hypno_prebuilt_inputs
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

_TOKENIZER: PreTrainedTokenizerFast | None = None
_TOKENIZER_PATH: str | None = None


class FishGenerateSemanticRequest(BaseModel):
    prompt: str
    ref_tokens: list[int] = Field(default_factory=list)
    ref_text: str = ""
    num_samples: int = Field(default=1, ge=1, le=64)
    max_new_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.8, ge=0.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_num_seqs: int = Field(default=8, ge=1, le=64)
    repetition_penalty: float = Field(default=1.1, ge=0.0)
    top_k: int = Field(default=30, ge=-1)


class FishGenerateSemanticResponse(BaseModel):
    candidates: list[list[int]]


def _model_path_from_app(app: FastAPI) -> str:
    path = getattr(app.state, "model_path", None) or os.environ.get("SGLANG_MODEL_PATH")
    if not path:
        raise HTTPException(
            status_code=503,
            detail="Tokenizer path unknown; set app.state.model_path or SGLANG_MODEL_PATH",
        )
    return str(path)


def _get_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    global _TOKENIZER, _TOKENIZER_PATH
    if _TOKENIZER is not None and _TOKENIZER_PATH == model_path:
        return _TOKENIZER
    _TOKENIZER = PreTrainedTokenizerFast.from_pretrained(model_path)
    _TOKENIZER_PATH = model_path
    return _TOKENIZER


def extract_semantic_row0(result: Any) -> list[int]:
    """Read generated semantic token IDs (CB0 row) from terminal pipeline result."""
    data: Any = result
    if isinstance(result, StagePayload):
        data = result.data
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected terminal result type: {type(result).__name__}")

    output_codes = data.get("output_codes")
    if output_codes is None:
        raise ValueError("Terminal result missing output_codes")

    if isinstance(output_codes, list):
        if not output_codes:
            return []
        row0 = output_codes[0]
        if isinstance(row0, list):
            return [int(x) for x in row0]
        return [int(output_codes)]

    raise ValueError(f"Unsupported output_codes type: {type(output_codes).__name__}")


async def _run_one_sample(
    client: Client,
    *,
    hypno_inputs: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    top_k: int,
    seed: int | None,
) -> list[int]:
    sampling = SamplingParams(
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        top_k=int(top_k),
        seed=seed,
    )
    gen_req = GenerateRequest(
        stream=False,
        sampling=sampling,
        metadata={
            "task": "hypno_semantic",
            "hypno_inputs": dict(hypno_inputs),
        },
    )
    request_id = f"hypno-sem-{uuid.uuid4().hex[:12]}"
    omni_request = client._build_omni_request(gen_req)
    result = await client._coordinator.submit(request_id, omni_request)
    return extract_semantic_row0(result)


def register_fish_semantic_routes(app: FastAPI) -> None:
    @app.post("/v1/fish/generate_semantic", response_model=FishGenerateSemanticResponse)
    async def fish_generate_semantic(
        body: FishGenerateSemanticRequest,
        request: Request,
    ) -> FishGenerateSemanticResponse:
        client: Client = request.app.state.client
        model_path = _model_path_from_app(request.app)
        tokenizer = _get_tokenizer(model_path)

        try:
            hypno_inputs = build_hypno_prebuilt_inputs(
                tokenizer,
                prompt=body.prompt,
                ref_text=body.ref_text,
                ref_tokens=body.ref_tokens,
                max_new_tokens=body.max_new_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
                repetition_penalty=body.repetition_penalty,
                top_k=body.top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        width = min(int(body.num_samples), int(body.max_num_seqs))
        sem = asyncio.Semaphore(max(1, width))

        async def _sample(i: int) -> list[int]:
            async with sem:
                seed = int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF
                return await _run_one_sample(
                    client,
                    hypno_inputs=hypno_inputs,
                    max_new_tokens=body.max_new_tokens,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    repetition_penalty=body.repetition_penalty,
                    top_k=body.top_k,
                    seed=seed ^ i,
                )

        try:
            candidates = await asyncio.gather(
                *[_sample(i) for i in range(int(body.num_samples))]
            )
        except Exception as exc:
            logger.exception("fish_generate_semantic failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if len(candidates) < body.num_samples:
            raise HTTPException(
                status_code=500,
                detail=f"Expected {body.num_samples} candidates, got {len(candidates)}",
            )
        return FishGenerateSemanticResponse(candidates=candidates)
