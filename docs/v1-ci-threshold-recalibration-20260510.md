# V1 CI Threshold Recalibration - 2026-05-10

## Scope

- Models: `qwen3-omni-v1`, `s2-pro-v1`
- Stages: `ALL`
- Repeats: 5, using worst-of-5
- Hardware: 2x NVIDIA H20
- Provenance: `calibrate-v1-thresholds-20260509` at `771874a8`
- Venv: `/data/chenyang/.python/omni/bin/python`

Run artifacts are local and intentionally not committed:

- Qwen3-Omni: `.tune-runs/20260509T215050Z_qwen3-omni-v1_ALL_r5/`
- S2-Pro: `.tune-runs/20260510T000700Z_s2-pro-v1_ALL_r5_retry1/`

Raw per-run observation reports copied from those run directories:

- `docs/calibration/v1-ci-threshold-observation-qwen3-omni-20260509.md`
- `docs/calibration/v1-ci-threshold-observation-s2-pro-20260510.md`

The first S2-Pro run at `.tune-runs/20260509T215050Z_s2-pro-v1_ALL_r5/` was discarded because the report had `N/A` for run 1/2 after an OOM retry artifact issue.

## Accuracy

Qwen3-Omni accuracy did not show a systematic regression. No accuracy thresholds were loosened. Existing thresholds were kept except for speed/WER changes described below.

Notable worst-of-5 observations:

- `mmmu_accuracy`: 58.00%
- `mmsu_accuracy`: 69.40%
- `videoamme_accuracy`: 66.67%
- `videomme_accuracy`: 53.33%
- Talker accuracy stages were stable at their existing floors.

## WER

Qwen3-Omni WER was handled conservatively:

- `mmsu_talker_wer` matched the expected fix direction: `n_above_50` stayed `0` in all five runs, so it was tightened from `1` to `0`.
- `tts_wer` stayed clean with `n_above_50=0`, so it was tightened from `1` to `0`.
- `videomme_talker_wer` stayed clean with `n_above_50=0`, so it was tightened from `1` to `0`.
- `mmmu_talker_wer` did not match the expected direction: worst-of-5 `n_above_50=2`. It was not changed.
- `videoamme_talker_wer` did not return to clean output: worst-of-5 `n_above_50=1`. It was not changed.

S2-Pro WER was recalibrated for the normalizer change:

- Non-stream corpus WER tightened from `0.012` to `0.010638297872340425`.
- Non-stream max per-sample WER tightened from `0.5` to `0.25`.
- Streaming corpus WER loosened from `0.012` to `0.03`, matching the expected normalizer-driven recalibration.
- Streaming max per-sample WER tightened from `0.5` to `0.17`.

## Speed

Speed changes used smart apply:

- Speed metrics that tightened were written back automatically.
- Qwen3 non-talker speed loosening was not applied.
- Qwen3 talker/TTS speed loosening was not applied unless it tightened under smart apply.
- S2-Pro speed was skipped because `apply-plan` could not parse current nested speed baselines (`direction=unknown`), so no speed thresholds were changed for S2-Pro.

CUDA graph runtime evidence was present in Qwen3 talker logs, including `Capture cuda graph ...` and decode logs with `cuda graph: True`.

Qwen3 speed changes written:

- `mmmu_speed`: throughput, tok/s, and latency tightened.
- `mmmu_talker_speed`: throughput, tok/s, latency, and RTF tightened.
- `mmsu_speed`: throughput tightened.
- `mmsu_talker_speed`: throughput, tok/s, latency, and RTF tightened.
- `tts_speed`: throughput and latency tightened.
- `videoamme_talker_speed`: RTF tightened.
- `videomme_speed`: throughput and latency tightened.
- `videomme_talker_speed`: throughput and latency tightened.

Skipped speed loosening:

- Qwen3 `mmsu_speed.latency_mean_s`
- Qwen3 `tts_speed.rtf_mean`
- Qwen3 `videoamme_speed` throughput, tok/s, latency
- Qwen3 `videoamme_talker_speed` throughput, tok/s, latency
- Qwen3 `videomme_talker_speed` tok/s and RTF
- All S2-Pro speed metrics due `direction=unknown`

## Verification

- Ran `tune.py precheck` for both models successfully.
- Ran Qwen3-Omni `ALL` stages with 5 repeats; `tune.py` exited `0`.
- Ran S2-Pro `ALL` stages with 5 repeats in the retry run; `tune.py` exited `0`.
- Filled all report context placeholders.
- Ran IDE lint diagnostics on edited test files; no linter errors were reported.
