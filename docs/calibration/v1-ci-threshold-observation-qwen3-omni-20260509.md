# CI Threshold Observation Report

## 1. MMMU Accuracy

— 2× NVIDIA H20 from precheck.json, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 50 | 50 | 58.00 |
| 2 | 50 | 50 | 60.00 |
| 3 | 50 | 50 | 62.00 |
| 4 | 50 | 50 | 62.00 |
| 5 | 50 | 50 | 60.00 |
| **Worst-of-5** | — | — | **58.00** |

## 2. MMMU Speed

— 2× NVIDIA H20 from precheck.json, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) |
|-----|--------|--------|--------|--------|--------|
| 1 | 50 | 50 | 0.667 | 53.30 | 11.337 |
| 2 | 50 | 50 | 0.668 | 53.30 | 11.284 |
| 3 | 50 | 50 | 0.661 | 53.90 | 11.272 |
| 4 | 50 | 50 | 0.704 | 52.50 | 10.623 |
| 5 | 50 | 50 | 0.667 | 52.80 | 11.383 |
| **Worst-of-5** | — | — | **0.661** | **52.50** | **11.383** |

## 3. MMMU TALKER Accuracy

— 2× NVIDIA H20 from precheck.json, 10 samples, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 10 | 10 | 70.00 |
| 2 | 10 | 10 | 70.00 |
| 3 | 10 | 10 | 70.00 |
| 4 | 10 | 10 | 70.00 |
| 5 | 10 | 10 | 70.00 |
| **Worst-of-5** | — | — | **70.00** |

## 4. MMMU TALKER Wer

— 2× NVIDIA H20 from precheck.json, 10 samples, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 10 | 10 | 12.42 | 1 |
| 2 | 10 | 10 | 12.89 | 1 |
| 3 | 10 | 10 | 13.13 | 2 |
| 4 | 10 | 10 | 13.05 | 1 |
| 5 | 10 | 10 | 17.77 | 1 |
| **Worst-of-5** | — | — | **17.77** | **2** |

## 5. MMMU TALKER Speed

— 2× NVIDIA H20 from precheck.json, 10 samples, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 10 | 10 | 0.403 | 10.00 | 13.532 | 0.3700 |
| 2 | 10 | 10 | 0.403 | 10.30 | 13.218 | 0.3646 |
| 3 | 10 | 10 | 0.415 | 9.90 | 13.641 | 0.3777 |
| 4 | 10 | 10 | 0.402 | 10.30 | 13.170 | 0.3616 |
| 5 | 10 | 10 | 0.389 | 10.10 | 13.469 | 0.3620 |
| **Worst-of-5** | — | — | **0.389** | **9.90** | **13.641** | **0.3777** |

## 6. MMSU Accuracy

— 2× NVIDIA H20 from precheck.json, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 2000 | 2000 | 70.10 |
| 2 | 2000 | 2000 | 69.60 |
| 3 | 2000 | 2000 | 69.70 |
| 4 | 2000 | 2000 | 69.40 |
| 5 | 2000 | 2000 | 69.65 |
| **Worst-of-5** | — | — | **69.40** |

## 7. MMSU Speed

— 2× NVIDIA H20 from precheck.json, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) |
|-----|--------|--------|--------|--------|--------|
| 1 | 2000 | 2000 | 29.017 | 7.50 | 0.275 |
| 2 | 2000 | 2000 | 28.982 | 7.50 | 0.275 |
| 3 | 2000 | 2000 | 29.101 | 7.50 | 0.274 |
| 4 | 2000 | 2000 | 29.449 | 7.60 | 0.271 |
| 5 | 2000 | 2000 | 28.956 | 7.50 | 0.276 |
| **Worst-of-5** | — | — | **28.956** | **7.50** | **0.276** |

## 8. MMSU TALKER Accuracy

— 2× NVIDIA H20 from precheck.json, 20 samples, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 20 | 20 | 60.00 |
| 2 | 20 | 20 | 60.00 |
| 3 | 20 | 20 | 60.00 |
| 4 | 20 | 20 | 60.00 |
| 5 | 20 | 20 | 60.00 |
| **Worst-of-5** | — | — | **60.00** |

## 9. MMSU TALKER Wer

— 2× NVIDIA H20 from precheck.json, 20 samples, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 20 | 20 | 2.26 | 0 |
| 2 | 20 | 20 | 2.48 | 0 |
| 3 | 20 | 20 | 2.39 | 0 |
| 4 | 20 | 20 | 2.43 | 0 |
| 5 | 20 | 20 | 2.49 | 0 |
| **Worst-of-5** | — | — | **2.49** | **0** |

## 10. MMSU TALKER Speed

— 2× NVIDIA H20 from precheck.json, 20 samples, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 20 | 20 | 1.028 | 8.70 | 7.022 | 0.3903 |
| 2 | 20 | 20 | 1.049 | 8.80 | 6.880 | 0.3859 |
| 3 | 20 | 20 | 1.007 | 8.40 | 7.099 | 0.4037 |
| 4 | 20 | 20 | 1.034 | 8.80 | 6.974 | 0.3796 |
| 5 | 20 | 20 | 1.060 | 8.80 | 6.833 | 0.3856 |
| **Worst-of-5** | — | — | **1.007** | **8.40** | **7.099** | **0.4037** |

## 11. TTS Wer

— 2× NVIDIA H20 from precheck.json, 50 samples, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 50 | 50 | 1.42 | 0 |
| 2 | 50 | 50 | 1.42 | 0 |
| 3 | 50 | 50 | 1.42 | 0 |
| 4 | 50 | 50 | 1.42 | 0 |
| 5 | 50 | 50 | 1.42 | 0 |
| **Worst-of-5** | — | — | **1.42** | **0** |

## 12. TTS Speed

— 2× NVIDIA H20 from precheck.json, 50 samples, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 50 | 50 | 4.052 | 7.60 | 1.918 | 0.5840 |
| 2 | 50 | 50 | 4.050 | 7.70 | 1.904 | 0.5917 |
| 3 | 50 | 50 | 4.025 | 7.70 | 1.908 | 0.5938 |
| 4 | 50 | 50 | 4.034 | 7.60 | 1.925 | 0.5997 |
| 5 | 50 | 50 | 4.064 | 7.70 | 1.912 | 0.5954 |
| **Worst-of-5** | — | — | **4.025** | **7.60** | **1.925** | **0.5997** |

## 13. VIDEOAMME Accuracy

— 2× NVIDIA H20 from precheck.json, 30 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 30 | 30 | 66.67 |
| 2 | 30 | 30 | 66.67 |
| 3 | 30 | 30 | 66.67 |
| 4 | 30 | 30 | 66.67 |
| 5 | 30 | 30 | 66.67 |
| **Worst-of-5** | — | — | **66.67** |

## 14. VIDEOAMME Speed

— 2× NVIDIA H20 from precheck.json, 30 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) |
|-----|--------|--------|--------|--------|--------|
| 1 | 30 | 30 | 0.217 | 0.80 | 55.901 |
| 2 | 30 | 30 | 0.214 | 0.80 | 56.511 |
| 3 | 30 | 30 | 0.215 | 0.80 | 56.876 |
| 4 | 30 | 30 | 0.214 | 0.80 | 56.445 |
| 5 | 30 | 30 | 0.214 | 0.80 | 56.910 |
| **Worst-of-5** | — | — | **0.214** | **0.80** | **56.910** |

## 15. VIDEOAMME TALKER Accuracy

— 2× NVIDIA H20 from precheck.json, 10 samples, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 10 | 10 | 50.00 |
| 2 | 10 | 10 | 50.00 |
| 3 | 10 | 10 | 50.00 |
| 4 | 10 | 10 | 50.00 |
| 5 | 10 | 10 | 50.00 |
| **Worst-of-5** | — | — | **50.00** |

## 16. VIDEOAMME TALKER Wer

— 2× NVIDIA H20 from precheck.json, 10 samples, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 10 | 10 | 1.33 | 1 |
| 2 | 10 | 10 | 0.82 | 1 |
| 3 | 10 | 10 | 1.10 | 1 |
| 4 | 10 | 10 | 0.27 | 1 |
| 5 | 10 | 10 | 0.00 | 1 |
| **Worst-of-5** | — | — | **1.33** | **1** |

## 17. VIDEOAMME TALKER Speed

— 2× NVIDIA H20 from precheck.json, 10 samples, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 10 | 10 | 0.229 | 1.50 | 30.733 | 4.4398 |
| 2 | 10 | 10 | 0.234 | 1.50 | 30.579 | 4.3896 |
| 3 | 10 | 10 | 0.214 | 1.30 | 33.932 | 4.9824 |
| 4 | 10 | 10 | 0.231 | 1.50 | 31.126 | 4.4729 |
| 5 | 10 | 10 | 0.234 | 1.40 | 30.644 | 4.4802 |
| **Worst-of-5** | — | — | **0.214** | **1.30** | **33.932** | **4.9824** |

## 18. VIDEOMME Accuracy

— 2× NVIDIA H20 from precheck.json, 30 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 30 | 30 | 56.67 |
| 2 | 30 | 30 | 53.33 |
| 3 | 30 | 30 | 56.67 |
| 4 | 30 | 30 | 56.67 |
| 5 | 30 | 30 | 56.67 |
| **Worst-of-5** | — | — | **53.33** |

## 19. VIDEOMME Speed

— 2× NVIDIA H20 from precheck.json, 30 samples, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) |
|-----|--------|--------|--------|--------|--------|
| 1 | 30 | 30 | 0.217 | 2.00 | 56.217 |
| 2 | 30 | 30 | 0.217 | 2.00 | 56.419 |
| 3 | 30 | 30 | 0.217 | 2.00 | 56.237 |
| 4 | 30 | 30 | 0.221 | 2.00 | 55.284 |
| 5 | 30 | 30 | 0.217 | 2.00 | 56.343 |
| **Worst-of-5** | — | — | **0.217** | **2.00** | **56.419** |

## 20. VIDEOMME TALKER Accuracy

— 2× NVIDIA H20 from precheck.json, 10 samples, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 10 | 10 | 50.00 |
| 2 | 10 | 10 | 50.00 |
| 3 | 10 | 10 | 50.00 |
| 4 | 10 | 10 | 50.00 |
| 5 | 10 | 10 | 50.00 |
| **Worst-of-5** | — | — | **50.00** |

## 21. VIDEOMME TALKER Wer

— 2× NVIDIA H20 from precheck.json, 10 samples, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 10 | 10 | 0.82 | 0 |
| 2 | 10 | 10 | 1.40 | 0 |
| 3 | 10 | 10 | 0.86 | 0 |
| 4 | 10 | 10 | 0.94 | 0 |
| 5 | 10 | 10 | 0.85 | 0 |
| **Worst-of-5** | — | — | **1.40** | **0** |

## 22. VIDEOMME TALKER Speed

— 2× NVIDIA H20 from precheck.json, 10 samples, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 10 | 10 | 0.241 | 1.50 | 29.390 | 3.7142 |
| 2 | 10 | 10 | 0.248 | 1.50 | 28.246 | 3.5969 |
| 3 | 10 | 10 | 0.246 | 1.50 | 28.540 | 3.4808 |
| 4 | 10 | 10 | 0.239 | 1.30 | 29.202 | 3.9522 |
| 5 | 10 | 10 | 0.241 | 1.50 | 29.068 | 3.7342 |
| **Worst-of-5** | — | — | **0.239** | **1.30** | **29.390** | **3.9522** |

## 23. Docs smoke

— 2× NVIDIA H20, docs smoke, 5 runs

| Run | Result |
|-----|--------|
| 1 | PASS |
| 2 | PASS |
| 3 | PASS |
| 4 | PASS |
| 5 | PASS |
| **Worst-of-5** | **PASS** |

## Applied changes

| Stage | Metric | Old | New |
|-------|--------|-----|-----|
| mmmu_speed | _MMMU_P95[8]['throughput_qps'] | 0.616 | 0.661 |
| mmmu_speed | _MMMU_P95[8]['tok_per_s_agg'] | 51.8 | 52.5 |
| mmmu_speed | _MMMU_P95[8]['latency_mean_s'] | 12.302 | 11.383 |
| mmmu_talker_speed | _MMMU_AUDIO_P95[8]['throughput_qps'] | 0.142 | 0.389 |
| mmmu_talker_speed | _MMMU_AUDIO_P95[8]['tok_per_s_agg'] | 4.3 | 9.9 |
| mmmu_talker_speed | _MMMU_AUDIO_P95[8]['latency_mean_s'] | 31.533 | 13.641 |
| mmmu_talker_speed | _MMMU_AUDIO_P95[8]['rtf_mean'] | 0.4497 | 0.3777 |
| mmsu_speed | _MMSU_P95[8]['throughput_qps'] | 28.933 | 28.956 |
| mmsu_talker_wer | MMSU_AUDIO_WER_BELOW_50_CORPUS_MAX | 0.06 | 0.024945770065075923 |
| mmsu_talker_wer | MMSU_AUDIO_N_ABOVE_50_MAX | 1 | 0 |
| mmsu_talker_speed | _MMSU_AUDIO_P95[8]['throughput_qps'] | 0.326 | 1.007 |
| mmsu_talker_speed | _MMSU_AUDIO_P95[8]['tok_per_s_agg'] | 5.2 | 8.4 |
| mmsu_talker_speed | _MMSU_AUDIO_P95[8]['latency_mean_s'] | 11.825 | 7.099 |
| mmsu_talker_speed | _MMSU_AUDIO_P95[8]['rtf_mean'] | 0.4188 | 0.4037 |
| tts_wer | VC_WER_BELOW_50_CORPUS_MAX | 0.03 | 0.014184397163120567 |
| tts_wer | VC_N_ABOVE_50_MAX | 1 | 0 |
| tts_speed | _VC_NON_STREAM_P95[8]['throughput_qps'] | 3 | 4.025 |
| tts_speed | _VC_NON_STREAM_P95[8]['latency_mean_s'] | 1.938 | 1.925 |
| videoamme_talker_speed | _VIDEOAMME_TALKER_AUDIO_P95[8]['rtf_mean'] | 5.1571 | 4.9824 |
| videomme_speed | _VIDEOMME_P95[16]['throughput_qps'] | 0.211 | 0.217 |
| videomme_speed | _VIDEOMME_P95[16]['latency_mean_s'] | 57.268 | 56.419 |
| videomme_talker_wer | VIDEOMME_TALKER_WER_BELOW_50_CORPUS_MAX | 0.02 | 0.014005602240896359 |
| videomme_talker_wer | VIDEOMME_TALKER_N_ABOVE_50_MAX | 1 | 0 |
| videomme_talker_speed | _VIDEOMME_TALKER_AUDIO_P95[8]['throughput_qps'] | 0.238 | 0.239 |
| videomme_talker_speed | _VIDEOMME_TALKER_AUDIO_P95[8]['latency_mean_s'] | 29.783 | 29.39 |

## Provenance

- Model: qwen3-omni-v1
- Branch: calibrate-v1-thresholds-20260509 @ 771874a8 (dirty) — see `workspace.diff`
- Venv Python: /data/chenyang/.python/omni/bin/python (flag)
- sglang 0.5.8 · torch 2.9.1+cu128
- GPU: 2× NVIDIA H20
- tune-ci-thresholds v0.3.0
- Ran 2026-05-09T21:51:14Z – 2026-05-09T23:36:38Z
