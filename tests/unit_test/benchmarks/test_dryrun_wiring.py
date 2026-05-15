# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test for the no-GPU sweep-wiring dry-run.

The dry-run script synthesizes a complete retained bundle and pipes it
through the real validator. This test asserts the script returns 0 when
the wiring is healthy and returns non-zero when the wiring chain is
deliberately broken (e.g. the validator regressed or the bundle shape
drifted from what the schema requires).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "scripts"
    / "dryrun_sweep_wiring.sh"
)


def test_dryrun_script_exits_zero_when_wiring_is_healthy() -> None:
    """The happy path: stock dry-run returns 0 + prints OK message."""
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"dry-run exited {result.returncode}; stderr:\n{result.stderr}"
    )
    assert "wiring OK" in result.stdout


def test_dryrun_script_detects_validator_regression(tmp_path) -> None:
    """If the validator is replaced with a stub that always fails, the
    dry-run wrapper must surface the non-zero exit. This catches the
    "validator silently downgraded to no-op" regression class.
    """
    # Build a sibling copy of the dry-run script that points at a stub
    # validator returning exit 1, so we can prove the wrapper does not
    # swallow validator failures.
    stub_validator = tmp_path / "stub_validate.py"
    stub_validator.write_text(
        "import sys\n"
        "print('[stub] always fails', file=sys.stderr)\n"
        "sys.exit(1)\n"
    )

    dryrun_text = SCRIPT.read_text()
    # Redirect the python call to our stub. Keep the rest of the script
    # untouched so we exercise the real bundle synthesis + status JSONL
    # construction code paths.
    rewired = dryrun_text.replace(
        'python "$REPO_ROOT/benchmarks/scripts/validate_mmmu_artifacts.py" "$OUT_ROOT" "$STATUS_LOG"',
        f'python {stub_validator} "$OUT_ROOT" "$STATUS_LOG"',
    )
    rewired_path = tmp_path / "rewired_dryrun.sh"
    rewired_path.write_text(rewired)
    rewired_path.chmod(0o755)

    result = subprocess.run(
        ["bash", str(rewired_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        "dry-run wrapper swallowed a validator failure; "
        "regression in the wiring chain"
    )
