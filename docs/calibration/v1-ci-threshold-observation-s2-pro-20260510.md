# CI Threshold Observation Report

## 1. TTS NONSTREAM Wer

— 2× NVIDIA H20 from precheck.json, streaming_benchmark_max_samples=16, 5 runs

| Run | Samples run | Samples ok | Corpus WER (%) | Max per-sample WER (%) |
|-----|--------|--------|--------|--------|
| 1 | 50 | 50 | 0.89 | 14.29 |
| 2 | 50 | 50 | 1.06 | 16.67 |
| 3 | 50 | 50 | 1.06 | 16.67 |
| 4 | 50 | 50 | 1.06 | 25.00 |
| 5 | 50 | 50 | 1.06 | 14.29 |
| **Worst-of-5** | — | — | **1.06** | **25.00** |

## 2. TTS NONSTREAM Speed

— 2× NVIDIA H20 from precheck.json, streaming_benchmark_max_samples=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 50 | 50 | 0.802 | 71.20 | 9.220 | 2.7487 |
| 2 | 50 | 50 | 0.784 | 72.20 | 9.344 | 2.7709 |
| 3 | 50 | 50 | 0.805 | 71.40 | 9.139 | 2.7521 |
| 4 | 50 | 50 | 0.787 | 70.40 | 9.204 | 2.8008 |
| 5 | 50 | 50 | 0.785 | 72.30 | 9.420 | 2.8011 |
| **Worst-of-5** | — | — | **0.784** | **70.40** | **9.420** | **2.8011** |

## 3. TTS STREAM Wer

— 2× NVIDIA H20 from precheck.json, streaming_benchmark_max_samples=16, 5 runs

| Run | Samples run | Samples ok | Corpus WER (%) | Max per-sample WER (%) |
|-----|--------|--------|--------|--------|
| 1 | 16 | 16 | 2.58 | 16.67 |
| 2 | 16 | 16 | 2.06 | 14.29 |
| 3 | 16 | 16 | 2.58 | 16.67 |
| 4 | 16 | 16 | 2.58 | 16.67 |
| 5 | 16 | 16 | 2.06 | 14.29 |
| **Worst-of-5** | — | — | **2.58** | **16.67** |

## 4. TTS STREAM Speed

— 2× NVIDIA H20 from precheck.json, streaming_benchmark_max_samples=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 16 | 16 | 0.702 | 74.30 | 9.510 | 2.5473 |
| 2 | 16 | 16 | 0.727 | 73.60 | 9.322 | 2.5764 |
| 3 | 16 | 16 | 0.741 | 72.10 | 9.017 | 2.3978 |
| 4 | 16 | 16 | 0.754 | 72.90 | 9.057 | 2.4831 |
| 5 | 16 | 16 | 0.729 | 73.80 | 9.771 | 2.6578 |
| **Worst-of-5** | — | — | **0.702** | **72.10** | **9.771** | **2.6578** |

## Applied changes

| Stage | Metric | Old | New |
|-------|--------|-----|-----|
| tts_nonstream_wer | VC_WER_MAX_CORPUS | 0.012 | 0.010638297872340425 |
| tts_nonstream_wer | VC_WER_MAX_PER_SAMPLE | 0.5 | 0.25 |
| tts_stream_wer | VC_STREAM_WER_MAX_CORPUS | 0.012 | 0.03 |
| tts_stream_wer | VC_STREAM_WER_MAX_PER_SAMPLE | 0.5 | 0.17 |

## Provenance

- Model: s2-pro-v1
- Branch: calibrate-v1-thresholds-20260509 @ 771874a8 (dirty) — see `workspace.diff`
- Venv Python: /data/chenyang/.python/omni/bin/python (flag)
- sglang 0.5.8 · torch 2.9.1+cu128
- GPU: 2× NVIDIA H20
- tune-ci-thresholds v0.3.0
- Ran 2026-05-10T00:07:23Z – 2026-05-10T00:23:55Z
