# SPDX-License-Identifier: Apache-2.0
"""Round 4 AC-10 gate: validate_mmmu_artifacts.py is a hard sweep gate.

The validator drives from sweep-status.jsonl. For every status row it
requires the cell directory to exist with the full bundle (mmmu_results
.json, preflight.json, launcher.log, stderr.log), the run_metadata to
carry all REQUIRED_FIELDS, the live AC-9 fields to be non-empty for
successful rows, and the status row's container_image_digest to be
non-empty and equal to the cell's run_metadata digest.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


VALIDATOR = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "scripts"
    / "validate_mmmu_artifacts.py"
)


def _run_validator(out_root: Path, status_log: Path) -> tuple[int, str]:
    res = subprocess.run(
        [sys.executable, str(VALIDATOR), str(out_root), str(status_log)],
        capture_output=True,
        text=True,
    )
    return res.returncode, res.stdout + res.stderr


def _write_complete_cell(cell_dir: Path, digest: str = "sha256:abc") -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "summary": {},
        "speed": {},
        "config": {},
        "run_metadata": {
            "commit_sha": "deadbeef",
            "branch": "feat/mmmu-streaming-benchmark",
            "sglang_version": "0.5.8",
            "backend": "omni",
            "model_id": "qwen3-omni",
            "model_revision": "abc123",
            "dataset_revisions": {"MMMU/MMMU": "rev1"},
            "seed": 42,
            "ignore_eos": False,
            "lane": "A",
            "stream": True,
            "max_tokens": 2048,
            "max_concurrency": 8,
            "temperature": 0.0,
            "warmup": 5,
            "request_rate": None,
            "timeout_s": 300,
            "repo_id": None,
            "max_samples": None,
            "mem_fraction_static_configured": 0.9,
            "kv_cache_capacity_tokens": 123456,
            "steady_state_gpu_gb": [80.5],
            "prefix_cache_disabled": True,
            "encoder_patches_active": False,
            "host": "ion8-omni",
            "container_name": "sglang-omni-hayden-benchmark",
            "container_image": "frankleeeee/sglang-omni:dev",
            "container_image_digest": digest,
            "server_port": 30000,
            "gpu_topology": "fake",
            "repetition_index": 0,
            "failure_count": 0,
        },
        "per_sample": [],
    }
    (cell_dir / "mmmu_results.json").write_text(json.dumps(result))
    (cell_dir / "preflight.json").write_text("{}")
    (cell_dir / "launcher.log").write_text("ready /snapshot/abc123\n")
    (cell_dir / "stderr.log").write_text("")


def _write_status(
    status_log: Path, rows: list[dict]
) -> None:
    status_log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_validator_passes_on_complete_bundle(tmp_path) -> None:
    out_root = tmp_path / "sweep"
    cell = out_root / "lane_A" / "omni" / "rep_0"
    _write_complete_cell(cell, digest="sha256:abc")
    status = tmp_path / "sweep-status.jsonl"
    _write_status(
        status,
        [
            {
                "host": "ion8-omni",
                "backend": "omni",
                "lane": "A",
                "rep": 0,
                "status": "success",
                "cell_dir": str(cell),
                "container_image_digest": "sha256:abc",
                "container_name": "sglang-omni-hayden-benchmark",
                "container_image": "frankleeeee/sglang-omni:dev",
                "server_port": 30000,
                "failure_count": 0,
            }
        ],
    )
    rc, out = _run_validator(out_root, status)
    assert rc == 0, out


def test_validator_fails_when_status_log_missing(tmp_path) -> None:
    out_root = tmp_path / "sweep"
    out_root.mkdir()
    status = tmp_path / "sweep-status.jsonl"
    # Status log empty / missing.
    rc, out = _run_validator(out_root, status)
    assert rc == 1
    assert "status log" in out


def test_validator_fails_on_missing_bundle_file(tmp_path) -> None:
    """Status row points at a cell that is missing preflight.json."""
    out_root = tmp_path / "sweep"
    cell = out_root / "lane_A" / "omni" / "rep_0"
    _write_complete_cell(cell)
    (cell / "preflight.json").unlink()  # remove a required file
    status = tmp_path / "sweep-status.jsonl"
    _write_status(
        status,
        [
            {
                "host": "ion8-omni",
                "backend": "omni",
                "lane": "A",
                "rep": 0,
                "status": "success",
                "cell_dir": str(cell),
                "container_image_digest": "sha256:abc",
            }
        ],
    )
    rc, out = _run_validator(out_root, status)
    assert rc == 1
    assert "preflight.json" in out


def test_validator_fails_on_empty_digest(tmp_path) -> None:
    """A successful cell with an empty status-row digest is rejected."""
    out_root = tmp_path / "sweep"
    cell = out_root / "lane_A" / "omni" / "rep_0"
    _write_complete_cell(cell, digest="sha256:abc")
    status = tmp_path / "sweep-status.jsonl"
    _write_status(
        status,
        [
            {
                "host": "ion8-omni",
                "backend": "omni",
                "lane": "A",
                "rep": 0,
                "status": "success",
                "cell_dir": str(cell),
                "container_image_digest": "",
            }
        ],
    )
    rc, out = _run_validator(out_root, status)
    assert rc == 1
    assert "digest" in out


def test_validator_fails_on_digest_mismatch(tmp_path) -> None:
    out_root = tmp_path / "sweep"
    cell = out_root / "lane_A" / "omni" / "rep_0"
    _write_complete_cell(cell, digest="sha256:abc")
    status = tmp_path / "sweep-status.jsonl"
    _write_status(
        status,
        [
            {
                "host": "ion8-omni",
                "backend": "omni",
                "lane": "A",
                "rep": 0,
                "status": "success",
                "cell_dir": str(cell),
                "container_image_digest": "sha256:OTHER",
            }
        ],
    )
    rc, out = _run_validator(out_root, status)
    assert rc == 1
    assert "digest" in out


def test_validator_fails_on_orphan_cell_with_no_status_row(tmp_path) -> None:
    """A cell on disk that no status row references is rejected."""
    out_root = tmp_path / "sweep"
    cell_known = out_root / "lane_A" / "omni" / "rep_0"
    cell_orphan = out_root / "lane_A" / "omni" / "rep_1"
    _write_complete_cell(cell_known, digest="sha256:abc")
    _write_complete_cell(cell_orphan, digest="sha256:def")
    status = tmp_path / "sweep-status.jsonl"
    _write_status(
        status,
        [
            {
                "host": "ion8-omni",
                "backend": "omni",
                "lane": "A",
                "rep": 0,
                "status": "success",
                "cell_dir": str(cell_known),
                "container_image_digest": "sha256:abc",
            }
        ],
    )
    rc, out = _run_validator(out_root, status)
    assert rc == 1
    assert "orphan" in out
