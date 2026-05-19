# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.fish_semantic_api import extract_semantic_row0
from sglang_omni.serve.openai_api import create_app


class _HypnoSemanticClient:
    def health(self) -> dict:
        return {"running": True}

    def _build_omni_request(self, request):
        from sglang_omni.client.client import Client

        return Client._build_omni_request(request)

    async def submit_semantic(self, _request_id: str, _omni_request: OmniRequest):
        payload = StagePayload(
            request_id="r1",
            request=OmniRequest(inputs={}, params={}),
            data={"output_codes": [[101, 102, 103], [1, 2, 3]]},
        )
        return payload


@pytest.fixture
def semantic_app(tmp_path):
    client = _HypnoSemanticClient()
    client._coordinator = MagicMock()
    client._coordinator.submit = AsyncMock(side_effect=client.submit_semantic)
    app = create_app(client, model_name="s2-pro", model_path=str(tmp_path))
    # Tokenizer load is skipped in route if we mock build — patch for integration test
    return app, client


def test_extract_semantic_row0() -> None:
    payload = StagePayload(
        request_id="x",
        request=OmniRequest(inputs={}),
        data={"output_codes": [[10, 11], [1, 2]]},
    )
    assert extract_semantic_row0(payload) == [10, 11]


def test_fish_generate_semantic_route(monkeypatch: pytest.MonkeyPatch, semantic_app) -> None:
    app, _client = semantic_app

    def _fake_build(tokenizer, **kwargs):
        return {
            "hypno_prebuilt": True,
            "input_ids": [1, 2, 3],
            "vq_mask_tokens": None,
            "vq_parts": None,
            "num_codebooks": 10,
            "codebook_size": 4096,
            "max_new_tokens": kwargs["max_new_tokens"],
            "temperature": kwargs["temperature"],
            "top_p": kwargs["top_p"],
            "repetition_penalty": 1.1,
            "top_k": 30,
        }

    monkeypatch.setattr(
        "sglang_omni.serve.fish_semantic_api.build_hypno_prebuilt_inputs",
        _fake_build,
    )
    monkeypatch.setattr(
        "sglang_omni.serve.fish_semantic_api._get_tokenizer",
        lambda _path: MagicMock(),
    )

    http = TestClient(app)
    resp = http.post(
        "/v1/fish/generate_semantic",
        json={
            "prompt": "hello world",
            "ref_tokens": [200, 201],
            "ref_text": "reference line",
            "num_samples": 2,
            "max_new_tokens": 16,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_num_seqs": 8,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["candidates"]) == 2
    assert body["candidates"][0] == [101, 102, 103]
