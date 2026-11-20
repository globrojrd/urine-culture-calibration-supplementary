# Supplement and manuscript reconciliation

The supplied three-page `Supplementary_Material.pdf` was read and visually checked. It remains the authors' existing file; no replacement PDF is manufactured from incomplete observations.

1. Page 1 repeats the explanatory paragraph for Table S2. Keep one explanation, including the distinction between individual seed policies and the single calibrated ensemble.
2. Figure S2 depicts a no-release point at (0,0). Its conditional unsafe-release probability is undefined because its denominator is zero. Label the symbol as a plotting convention; do not describe it as measured zero conditional risk. The cleaned plotting command marks that explicitly.
3. Table S2 contains actual individual release counts 36, 41 and 45. Do not replace these with the ensemble's 39 releases repeated three times. The misleading candidate listing in notebook cell104 is excluded.
4. Figure S1 in the supplied PDF has a reliability panel and a separate bin-count panel. Notebook cell100 instead uses count annotations on one panel. The clean script preserves the cell100 numerical estimator and bootstrap seed while providing the supplied PDF's two-panel layout. Pixel-identical historical rendering is not claimed.
5. Table S3 agrees with the notebook's recorded final confusion matrix: Positive-to-Uncertain 5, Positive-to-Negative 0; 39 Negative releases and 111 referrals.
6. Table S4 pools 27 seed-specific evaluations per condition. Its totals are computational counts across reused images, not independent clinical observations.
7. Notebook cells72–74 and101 confirm gradient accumulation of two for EfficientNetV2-S as well as Swin-T. The manuscript training sentence should say both architectures used accumulation.
8. The earlier single-policy bootstrap code computes NPV, producing a degenerate interval when no unsafe release is present. The manuscript should say that this bootstrap interval was not interpreted as safety evidence; saying NPV was never bootstrapped contradicts the source. Report the exact one-sided unsafe-release bound instead.
9. Threshold selection has 50 Negative, 49 Positive and 51 Uncertain images. Evaluation and calibration have 50 per class. The combined sensitivity cohort has 100/99/101; avoid describing every partition as exactly balanced.

The manuscript's EfficientNet accumulation and NPV-bootstrap sentences were corrected on 20 November 2026 after checking the recovered files. The existing supplementary PDF has not been regenerated; its duplicated paragraph and no-release caption remain editorial items to correct in its source. The regenerated numerical supplementary assets were checked against the recovered original tables.
