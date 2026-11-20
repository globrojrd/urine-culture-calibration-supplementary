# Validation and limits

Checked on 20 November 2026. All numerical fixtures used for software testing are explicitly synthetic and are not included as study results.

## Checks completed

- All Python source files parse, and all 13 command-line entry points open their help successfully.
- Original-versus-extracted ResNet and EfficientNet accumulation loops produce identical parameters and metrics on synthetic mini-batches, including the final incomplete accumulation group. Focal loss matches the original implementation; gamma zero reduces to cross-entropy.
- All three network constructors perform a CPU forward pass with three outputs. Pretrained weights were not downloaded for these checks.
- The fixed-split implementation reproduces the expected class counts and deterministic membership on a synthetic 1,500-row input. This does not identify the actual study images; the original manifest is still required.
- All six calibrator transforms agree with the original notebook functions on a fixed synthetic fixture. Maximum probability difference was zero for uncalibrated/TS/VS, approximately 1e-14 for Dirichlet, and at most 2.3e-16 for Beta/Isotonic. Argmax predictions and JSON round trips agree.
- A synthetic factorial workflow exercises all 27 checkpoint configurations and six probability conditions, with separate development and frozen evaluation phases. Separate checks withhold test files during development and reject modified frozen calibrator files.
- The ensemble tests verify sample-ID alignment, reference labels, fixed thresholds, no fitting during evaluation, rejection of development/evaluation overlap, and mismatch rejection.
- The cleaned paired analysis matches execution of original notebook cell123 on a common synthetic fixture: all 30 metric-difference summaries, six exact-test rows and 90,000 bootstrap-difference rows. McNemar also agrees with SciPy in 625 discordance cases; Holm agrees with statsmodels.
- Aggregate notebook discordances reproduce the six reported Holm-adjusted Accuracy values and six TNRR values. The four zero-event confidence bounds have been independently checked. This validates arithmetic from supplied counts, not the underlying prediction files.
- Synthetic file exports, comparison staging, statistics and supplementary/reviewer-curve generation are exercised end to end. No real image panel is generated without the original files.

## Sensitivity optimizer behavior

The cleaned sensitivity code retains the original objective and numerical settings, but stops on reported nonconvergence rather than silently using a failed optimizer result. A noisy synthetic integration fixture triggered a SciPy `ABNORMAL` termination and was correctly rejected. A separate symmetric fixture exercises one complete development repartition with all 27 checkpoints and with test files absent. The 100-repartition analysis has not been rerun on original data.

An optimizer stop on the original inputs requires inspection of the saved historical fit, its convergence status and environment. Do not remove that check merely to obtain a preferred ranking. Numerical equivalence of the paper's original sensitivity results remains unverified.

## Local verification environment

Python 3.11.7; NumPy 1.26.4; pandas 2.2.3; SciPy 1.16.0; scikit-learn 1.7.1; PyTorch 2.7.0; torchvision 0.22.0; Matplotlib 3.10.3; statsmodels 0.14.0. This is different from the notebook-recorded training environment. Source-equivalence tests used both source and extracted functions in this same local environment.

## Not validated yet

- Numerical regeneration of the article from original observation-level files.
- Exact corrected-ensemble parameters and threshold from the missing external script and original logits.
- Actual image memberships or image byte hashes against the recorded manifest.
- Full 27-model GPU retraining, historical checkpoint equivalence or interrupted-run recovery.
- Reproduction of recorded NVIDIA L4 latencies, GPU memory and 27-run training times.
- The 100-repartition sensitivity result on original inputs.
- Public GitHub publication: preparing these files does not itself update the remote repository.

The published observations (39/150 released, no observed unsafe releases and the reported p-values) remain author-supplied results until the saved inputs have been recovered and the commands rerun. Code availability, numerical reproducibility and external clinical validation are different claims.
