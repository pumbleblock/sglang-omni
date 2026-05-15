#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Dry-run smoke test for the MMMU sweep orchestration wiring.
#
# Runs on any machine (no docker, no GPU, no SSH required). Synthesizes a
# minimal-but-valid retained bundle for a single cell — full
# `mmmu_results.json` with the required run_metadata fields, a
# `preflight.json` carrying a `launch_command` with the contracted policy
# flags, plus the expected `launcher.log` and `stderr.log`. Builds a
# matching `sweep-status.jsonl` row. Then invokes the real
# `validate_mmmu_artifacts.py` against this bundle and asserts it exits 0.
#
# Purpose: before committing ~3 GPU-h on a real H200 sweep, the operator
# can run this in <1s on macOS and confirm the orchestration code (sweep
# runner ↔ status JSONL schema ↔ retained-evidence validator) is wired
# correctly end-to-end. Failures here mean someone refactored a piece of
# the chain and broke the contract without catching it in unit tests.
#
# Usage:
#   bash benchmarks/scripts/dryrun_sweep_wiring.sh
#
# Exit codes:
#   0 = wiring is healthy
#   1 = wiring is broken (validator reported issues; see stderr for details)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_ROOT="$(mktemp -d)"
trap 'rm -rf "$OUT_ROOT"' EXIT

CELL_DIR="$OUT_ROOT/lane_A/omni/rep_0"
STATUS_LOG="$OUT_ROOT/sweep-status.jsonl"
mkdir -p "$CELL_DIR"

DIGEST="sha256:dryrun0000000000000000000000000000000000000000000000000000000000"
CONTAINER_NAME="sglang-omni-hayden-benchmark"
CONTAINER_IMAGE="frankleeeee/sglang-omni:dev"

# A complete mmmu_results.json with a run_metadata block carrying all
# REQUIRED_FIELDS and non-empty live values. The validator requires
# every key — keep this in sync with `benchmarks/scripts/run_metadata.py`.
cat >"$CELL_DIR/mmmu_results.json" <<JSON
{
  "summary": {},
  "speed": {},
  "config": {},
  "run_metadata": {
    "commit_sha": "dryrun-deadbeef",
    "branch": "dryrun-branch",
    "sglang_version": "0.5.8",
    "backend": "omni",
    "model_id": "qwen3-omni",
    "model_revision": "dryrun-model-rev",
    "dataset_revisions": {"MMMU/MMMU": "dryrun-dataset-rev"},
    "seed": 42,
    "ignore_eos": false,
    "lane": "A",
    "stream": true,
    "max_tokens": 2048,
    "max_concurrency": 8,
    "temperature": 0.0,
    "warmup": 5,
    "request_rate": null,
    "timeout_s": 300,
    "repo_id": null,
    "max_samples": null,
    "mem_fraction_static_configured": 0.9,
    "kv_cache_capacity_tokens": 123456,
    "steady_state_gpu_gb": [80.5],
    "prefix_cache_disabled": true,
    "encoder_patches_active": false,
    "host": "dryrun-host",
    "container_name": "$CONTAINER_NAME",
    "container_image": "$CONTAINER_IMAGE",
    "container_image_digest": "$DIGEST",
    "server_port": 30000,
    "gpu_topology": "dryrun-topology",
    "repetition_index": 0,
    "failure_count": 0
  },
  "per_sample": []
}
JSON

# Synthesized preflight.json with a launch_command that contains both
# --disable-radix-cache and --mem-fraction-static; the validator's
# launch-evidence check requires this exact shape.
cat >"$CELL_DIR/preflight.json" <<JSON
{
  "ok": true,
  "containers": {
    "$CONTAINER_NAME": {
      "container_image_digest": "$DIGEST",
      "container_image": "$CONTAINER_IMAGE",
      "launch_command": [
        "docker", "run", "-d", "--name", "$CONTAINER_NAME",
        "$CONTAINER_IMAGE",
        "sgl-omni", "serve", "--model-path", "/snapshot",
        "--text-only", "--port", "30000",
        "--mem-fraction-static", "0.9",
        "--disable-radix-cache"
      ]
    }
  }
}
JSON

echo "dryrun launcher.log: loaded /snapshot/dryrun ready" >"$CELL_DIR/launcher.log"
: >"$CELL_DIR/stderr.log"

# A matching sweep-status.jsonl row whose container_image_digest agrees
# with the run_metadata digest above.
cat >"$STATUS_LOG" <<JSON
{"host":"dryrun-host","backend":"omni","lane":"A","rep":0,"status":"success","cell_dir":"$CELL_DIR","container_name":"$CONTAINER_NAME","container_image":"$CONTAINER_IMAGE","container_image_digest":"$DIGEST","server_port":30000,"failure_count":0}
JSON

echo "[dryrun] synthesized bundle at $OUT_ROOT"
echo "[dryrun] invoking real validator..."
python "$REPO_ROOT/benchmarks/scripts/validate_mmmu_artifacts.py" "$OUT_ROOT" "$STATUS_LOG"
echo "[dryrun] wiring OK"
