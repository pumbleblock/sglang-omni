# SPDX-License-Identifier: Apache-2.0
"""Round 4 AC-7 regression: every post-launch preflight verification stage
must route over SSH when ``--host`` is set, never read orchestrator-local
state. Codex flagged that ``probe_v1_models``, ``probe_sglang_data_uri``,
and ``verify_launcher_log_references_snapshot`` all silently used local
URLs / local filesystem reads even after Round 3 wired some other probes
to ``--host``.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


def _import_preflight():
    """Import the preflight module via importlib to avoid argparse parse."""
    import importlib.util
    import sys
    from pathlib import Path

    name = "preflight_mmmu_sweep_test"
    if name in sys.modules:
        return sys.modules[name]
    path = (
        Path(__file__).resolve().parents[3]
        / "benchmarks"
        / "scripts"
        / "preflight_mmmu_sweep.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass lookups via cls.__module__ need this
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# probe_v1_models
# ---------------------------------------------------------------------------


def test_probe_v1_models_remote_uses_ssh_curl() -> None:
    mod = _import_preflight()
    captured = {}

    def fake_check_output(cmd, **kwargs):
        captured["cmd"] = cmd
        return json.dumps({"data": [{"id": "qwen3-omni"}]})

    with patch.object(mod.subprocess, "check_output", side_effect=fake_check_output):
        with patch.object(mod.request, "urlopen") as fake_urlopen:
            result = mod.probe_v1_models(
                "http://localhost:30000", host="ion8-omni"
            )

    assert result == {"data": [{"id": "qwen3-omni"}]}
    fake_urlopen.assert_not_called(), (
        "remote mode must not fall back to orchestrator-local urlopen"
    )
    cmd = captured["cmd"]
    assert cmd[0] == "ssh" and cmd[1] == "ion8-omni"
    assert "curl" in cmd
    assert any(arg.endswith("/v1/models") for arg in cmd)


def test_probe_v1_models_local_uses_urlopen() -> None:
    """Without --host, the local path is used (no SSH)."""
    mod = _import_preflight()

    class _FakeResp:
        def __init__(self, body: str) -> None:
            self._body = body.encode("utf-8")

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch.object(mod.subprocess, "check_output") as fake_check_output:
        with patch.object(
            mod.request,
            "urlopen",
            return_value=_FakeResp(json.dumps({"data": []})),
        ):
            result = mod.probe_v1_models("http://localhost:30000")

    assert result == {"data": []}
    fake_check_output.assert_not_called()


# ---------------------------------------------------------------------------
# probe_sglang_data_uri
# ---------------------------------------------------------------------------


def test_probe_sglang_data_uri_remote_uses_ssh_curl() -> None:
    mod = _import_preflight()
    captured = {}

    def fake_check_output(cmd, **kwargs):
        captured["cmd"] = cmd
        return "200"

    with patch.object(mod.subprocess, "check_output", side_effect=fake_check_output):
        with patch.object(mod.request, "urlopen") as fake_urlopen:
            result = mod.probe_sglang_data_uri(
                "http://localhost:30001", "qwen3-vl", host="ion9-omni"
            )

    assert result == {"status": 200, "ok": True}
    fake_urlopen.assert_not_called()
    cmd = captured["cmd"]
    assert cmd[0] == "ssh" and cmd[1] == "ion9-omni"
    assert "POST" in cmd
    assert any("/v1/chat/completions" in arg for arg in cmd)


# ---------------------------------------------------------------------------
# verify_launcher_log_references_snapshot
# ---------------------------------------------------------------------------


def test_verify_launcher_log_remote_uses_ssh_cat() -> None:
    mod = _import_preflight()
    captured = {}

    def fake_check_output(cmd, **kwargs):
        captured["cmd"] = cmd
        return "loaded /snapshot/abc123 ready"

    with patch.object(mod.subprocess, "check_output", side_effect=fake_check_output):
        ok = mod.verify_launcher_log_references_snapshot(
            "/tmp/sglang-omni-benchmark.log", "/snapshot/abc123", host="ion8-omni"
        )

    assert ok is True
    cmd = captured["cmd"]
    assert cmd[0] == "ssh" and cmd[1] == "ion8-omni"
    assert "cat" in cmd
    assert "/tmp/sglang-omni-benchmark.log" in cmd


def test_verify_launcher_log_remote_fails_when_snapshot_absent() -> None:
    """Remote-mode False path: SSH returns log content that does NOT mention the snapshot."""
    mod = _import_preflight()

    def fake_check_output(cmd, **kwargs):
        return "loaded /some/other/path ready"

    with patch.object(mod.subprocess, "check_output", side_effect=fake_check_output):
        ok = mod.verify_launcher_log_references_snapshot(
            "/tmp/launcher.log", "/snapshot/abc123", host="ion8-omni"
        )
    assert ok is False


def test_verify_launcher_log_remote_fails_when_ssh_errors() -> None:
    """If the SSH cat fails, the verification returns False (not crash)."""
    import subprocess

    mod = _import_preflight()

    def fake_check_output(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    with patch.object(mod.subprocess, "check_output", side_effect=fake_check_output):
        ok = mod.verify_launcher_log_references_snapshot(
            "/tmp/missing.log", "/snapshot/abc123", host="ion8-omni"
        )
    assert ok is False
