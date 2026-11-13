# Time to detection on the test split (hourly scoring)

Alert = first scoring point with score >= the seed's session-level 5%-FPR threshold set on VALIDATION (max aggregation). Hours pooled over the 3 seeds (median, IQR).

| Model | Thresholds (valid) | Test session FPR at them | Detected | Detected before logoff | Start to alert (h) | First malicious event to alert (h) | Alert before logoff (h) |
|---|---|---|---|---|---|---|---|
| SLM (Qwen2.5-1.5B LoRA) | 0.3951, 0.3799, 0.3865 | 0.055, 0.055, 0.054 | 41.8% | 40.1% of positives (95.7% of detected) | 2.00 (1.00 to 4.00) | 0.66 (0.04 to 2.34) | 8.82 (4.98 to 9.23) |
| XGBoost (structured v3) | 0.001638, 0.001417, 0.001278 | 0.054, 0.057, 0.057 | 63.9% | 56.9% of positives (89.1% of detected) | 2.00 (1.00 to 5.00) | 0.86 (0.17 to 3.59) | 6.98 (3.12 to 9.18) |

243 positive test sessions; first malicious event a median 0.85 h after session start.
