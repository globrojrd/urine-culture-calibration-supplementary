# Validation with recovered study artifacts

Reanalysed on **20 November 2026** using the original saved predictions, logits, split manifest and training histories. The recovery archive contained 1,777 experiment files; all hashes matched its recovery report. The original split-file hash also matches the notebook. No network training was repeated.

## Numerical reproduction

| Analysis | Verified result |
|---|---|
| Seven frozen evaluation policies | Same 150 images and labels, paired by sample ID. |
| Exact McNemar and Holm | All six rows match exactly; no endpoint is significant after its six-test Holm correction. |
| Paired bootstrap | All 30 metric summaries and 90,000 differences (3,000 resamples x 6 comparisons x 5 metrics) agree within floating-point precision. |
| Single-policy bootstrap | All 2,000 iterations and six interval summaries agree within floating-point precision. |
| Clinical/operating denominators | 0/50 Positive releases, 0/100 non-Negative releases, 0/39 unsafe among releases; exact upper bounds 5.82%, 2.95%, 7.39%. Pre-abstention Positive FNR 5/50, all assigned Uncertain. |
| Primary factorial refit | All 162 conditions complete without optimizer failure. All development/evaluation classification and release decisions match when historical files are replayed with their original reader. |
| Original fitted ensemble replay | All 150 predictions and release decisions match; maximum probability difference 4.44e-16. Original threshold 0.967785046190242; 39/150 releases, zero unsafe. |
| Independent ensemble refit | Same predictions and releases; threshold difference 2.62e-9 and maximum probability difference 9.62e-8. |
| Sensitivity | All 16,200 seed-level operating counts and Macro-F1 values match; all 100 winners and selected-configuration ranks match. Selected configuration wins 15 times and ranks in the top three 25 times. All 100 ensemble release counts match. No optimizer failure. |
| Histories and split | All 27 runs contain 20 epochs; best epochs/scores agree with saved metadata. All original identities, partition counts and the focal-pilot gamma choice agree. |
| Reviewer curves | Regenerated means/SD agree with saved values to about 1.11e-16. The plotted data are measured histories. |
| Supplement | S1 bin counts/intervals, S2 individual release counts 36/41/45 and S3 confusion/referral tables agree with saved outputs. |

See [the machine-readable report](../results/reproduction_validation.json) for exact differences and provenance. `python scripts/verify_reproduction.py` checks all 96 input/provenance hashes, regenerates the main statistical tables and replays the original frozen ensemble without a GPU or Drive access.

## Numerical details and retained limitations

Refits are numerically close, not byte-identical. Across the primary factorial, the largest probability difference is approximately 1.65e-7. Across all sensitivity seed policies, the largest threshold difference is approximately 3.0e-6; every operating count and winner remains unchanged. The maximum ensemble sensitivity threshold difference is approximately 1.81e-7.

Historical CSV boundary behavior is documented in `PROVENANCE.md` and `results/serialization_boundary_audit.json`. Switching the old probability-table reader to round-trip parsing can lose an exactly-on-threshold development release at a difference of 1.11e-16. Replaying the actual historical parser restores all recorded decisions; no final evaluation decision changes. New comparisons use the preserved original frozen-policy tables, rather than mixing old thresholds with differently parsed probabilities.

The external corrected-ensemble script source is still absent. Its saved parameters and complete mathematical specification were recovered and its outputs reproduced; source-code recovery itself is not claimed. Images and checkpoint weights were not recovered, so image readability/hash audits, network inference, full retraining and GPU latency were not rerun. Training-duration summaries were checked against the saved logs; these are historical measurements, not fresh benchmarks.

The local validation used Python 3.11.7, NumPy 1.26.4, pandas 2.2.3, SciPy 1.16.0, scikit-learn 1.7.1, PyTorch 2.7.0, torchvision 0.22.0 and Matplotlib 3.10.3. This differs from the recorded experiment environment in the requirements files. Successful reanalysis does not establish patient independence, undo previous evaluation-set inspection, or imply external clinical validity.

## Software checks

The extracted implementations were also compared with original notebook functions using clearly identified synthetic fixtures. Calibrator transforms, training updates, focal loss and incomplete accumulation groups agree. All three architectures perform a CPU forward pass. Protocol tests cover pairing, no-release behavior, exact bounds, shared bootstrap samples, frozen evaluation and leakage rejection. These software fixtures are not supplied as study observations.

The stricter sensitivity optimizer guard was exercised on a nonconvergent synthetic fixture and stopped as intended. The full real-data run completed all 100 repartitions without that failure. Nothing was relaxed to obtain the historical winner or desired significance.
