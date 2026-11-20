# Recorded protocol

The source is the supplied experiment notebook, not a protocol inferred from reported outcomes.

| Partition | Negative | Positive | Uncertain | Total |
|---|---:|---:|---:|---:|
| Training | 300 | 299 | 301 | 900 |
| Model validation | 50 | 50 | 50 | 150 |
| Calibration | 50 | 50 | 50 | 150 |
| Threshold selection | 50 | 49 | 51 | 150 |
| Exploratory evaluation | 50 | 50 | 50 | 150 |

The split seed is 2607, with successive stratified divisions using 2608, 2609 and 2610. Training seeds are 42, 123 and 2026. Patient/specimen grouping is unavailable. Source manifest SHA-256: `3e8e49d6f7ec2bbf30d36d0795bee99c5261bae5d9318276182d56d75ce5055c`. A portable-path export will have a different file hash while retaining the original identities and memberships.

Training uses ImageNet-pretrained ResNet-18, EfficientNetV2-S and Swin-T; CE, label smoothing 0.1 or focal loss gamma 2; 20 epochs; AdamW learning rate and weight decay 1e-4; no scheduler or mixed precision. ResNet batch size is 32. EfficientNet and Swin use physical batch size 16 and two-step gradient accumulation. The original final incomplete accumulation group is retained, including division of its loss by two. This is a fidelity choice, not an assertion that it is optimal. Validation Macro-F1 selects checkpoints; the focal gamma pilot is separate and uses 10 epochs on ResNet seed 42.

Six probability conditions are uncalibrated, temperature scaling, Vector Scaling, Dirichlet, Negative-event Beta and Negative-event Isotonic calibration. Vector Scaling uses positive exp(log-scale) multipliers, centered biases and L2 strength 1e-3. The primary and sensitivity implementations retain their respective numerical clipping conventions. Selection maximizes development coverage subject to zero observed unsafe releases; tied policies use the higher threshold. A no-release policy is feasible, but its NPV and conditional unsafe rate are undefined.

The corrected historical ensemble output records a threshold of 0.967785046190, 42/150 development releases and 39/150 evaluation releases. These are verification targets, not constants inserted into prediction code. The missing external correction source prevents claiming exact historical ensemble reconstruction before checking the saved inputs.

## Statistical endpoints

| Endpoint | Historical observation | One-sided exact 95% upper bound |
|---|---:|---:|
| Operational Positive FNR | 0/50 | 5.82% |
| Non-Negative unsafe release rate | 0/100 | 2.95% |
| Unsafe among releases | 0/39 | 7.39% |
| Overall unsafe release rate | 0/150 | 1.98% |

Pre-abstention Positive FNR is 5/50=10%: all five were classified Uncertain, none Negative, and all were referred. Coverage is 39/150=26%. These endpoints have different denominators and must not be interchanged. Exact bounds assume binomial independence, which cannot be established at patient level from this dataset.

Six comparisons are ordered ResNet42, Swin42, ResNet123, Swin123, ResNet2026, Swin2026. Exact two-sided McNemar uses paired correctness among all 150 images and paired release among the 50 reference Negatives. Holm adjustment is separate within each six-test family. Reported adjusted values are Accuracy `[1, .625, .750, 1, .625, .421875]` and TNRR `[1, 1, 1, 1, .1875, 1]`. None is significant at .05 after Holm; this does not prove equivalence or superiority.

The paired percentile bootstrap uses 3,000 resamples, seed 20261112, separately stratifies all three classes, and shares each sampled index set across all seven policies. Its 30 metric-difference intervals are marginal, not multiplicity-adjusted. The earlier individual-policy bootstrap instead uses 2,000 resamples, seed 20260730, and Negative versus pooled non-Negative strata. Reliability-bin intervals use 2,000 resamples and seed 20260729. The original shuffling/draw order is retained.

The development sensitivity analysis uses 100 overlapping 150/150 repartitions of 300 calibration-plus-threshold images (100 Negative, 99 Positive, 101 Uncertain), random state 2026. Networks remain frozen. These repartitions and the three training seeds are not independent external cohorts.
