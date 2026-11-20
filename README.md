# A Factorial Evaluation of Calibration Methods for Selective Automated Urine Culture Screening

Research code and supplementary material for the manuscript submitted to *Computación y Sistemas*.

The code was extracted from the authors' experiment notebook, retaining the final five-part image split, 27 trained configurations, six probability treatments, frozen selective-release policies, and reviewer-requested analyses. Superseded experiments, interrupted runs, repeated notebook state and presentation-only inspections were excluded. See [provenance](docs/PROVENANCE.md) and [validation](docs/VALIDATION.md).

## Reproduction status

The code has been checked against the notebook implementations on synthetic fixtures. **The original image-level prediction CSVs, split manifest, logit tables, training histories, images and checkpoints are not included in this initial code package.** Consequently, the reported study results have not yet been independently regenerated from this repository. Exporters below recover the existing experiment files without retraining. Update this status only after those artifacts are added and the commands succeed.

The notebook called an external corrected-ensemble script whose source was not embedded. `urineculture/ensemble.py` is a documented reconstruction from the available mean-logit, calibration and threshold-selection implementations; it is not a recovered copy of that missing file. Numerical agreement with the historical fitted ensemble remains to be checked using the original logits.

## Contents

| Location | Purpose |
|---|---|
| `Supplementary_Material.pdf` | Existing supplementary document: reliability, computational measurements, seed policies, risk–coverage and confusion tables. |
| `urineculture/` | Reusable data, training, calibration, policy, ensemble, statistics, figures and sensitivity functions. |
| `scripts/` | Commands to export inputs and run each analysis. |
| `tests/` | Synthetic protocol and integration tests, not study observations. |
| `data/` | Input specification; actual study CSVs must be exported and added. |
| `docs/` | Source provenance, reproduction protocol, supplementary corrections and validation scope. |

The public image dataset is [Mendeley Data, version 3, DOI 10.17632/xrbtd74pfj.3](https://doi.org/10.17632/xrbtd74pfj.3). It is not redistributed here. Images must be obtained from that source and kept in `Negative/`, `Positive/`, and `Uncertain/` directories. Reference labels describe culture images, not symptomatic infection.

## Install and check the code

Use a separate Python environment. The notebook recorded Python 3.12.13. `requirements-analysis.txt` records the reported direct analysis-library versions; it is not a complete historical lockfile.

```bash
python -m pip install -r requirements-analysis.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

For training or checkpoint inference, also install `requirements.txt`. The recorded training environment used PyTorch 2.11.0, torchvision 0.26.0, CUDA 12.8 and an NVIDIA L4. These original dependencies were not reinstalled during extraction; local verification used the versions listed in `docs/VALIDATION.md`. GPU timing and freshly trained weights can differ across environments.

## Recover the authors' saved experiment artifacts

These commands are for the authors with access to the original experiment folder. They read the saved files and write a new, portable export outside that folder. They do not modify the experiment or retrain any model. In Colab, mount Google Drive first.

```bash
python scripts/export_artifacts.py \
  --project /content/drive/MyDrive/Extension_articulo_urocultivo \
  --output /content/public_urineculture_data
python scripts/export_review_inputs.py \
  --project /content/drive/MyDrive/Extension_articulo_urocultivo \
  --output /content/public_urineculture_data/review
```

Copy the exported directory's contents into this repository's `data/`. The exporters preserve sample IDs, reference labels and saved row order; remove machine-specific image paths; and record source/output hashes. Exported splits contain class-relative paths. Do not regenerate the study manifest by scanning a directory: the original input order and path-derived IDs matter.

## Recompute the reviewer-requested statistics

Requires `data/review/ensemble.csv` and the six named comparator CSVs listed in `data/README.md`.

```bash
python scripts/compare_policies.py --inputs data/review --output outputs/statistics
```

Outputs include exact two-sided McNemar tests, Holm-adjusted p-values, 3,000 paired bootstrap difference samples/intervals, the separate 2,000-resample single-policy bootstrap, and false-negative/unsafe-release rates with one-sided exact bounds. The original row order is retained because it affects the exact random draws. A degenerate NPV bootstrap interval is not used as evidence of zero population risk.

## Refit calibration and frozen policies from saved logits

Requires the portable `data/experiments.csv` and 81 logit tables. Network training is not needed for this route.

```bash
python scripts/evaluate_factorial.py develop --manifest data/experiments.csv --output outputs/factorial
python scripts/evaluate_factorial.py evaluate --manifest data/experiments.csv --output outputs/factorial
python scripts/run_ensemble.py fit --manifest data/experiments.csv --output outputs/ensemble_policy
python scripts/run_ensemble.py evaluate --manifest data/experiments.csv \
  --frozen-policy outputs/ensemble_policy/frozen_ensemble.json --output outputs/ensemble_evaluation
```

The first factorial command fits calibrators on `calibration`, selects thresholds and ranks configurations on `threshold_selection`, then saves frozen artifacts. The second evaluates them. The ensemble reproduces the manuscript's selected **ResNet-18/focal/seeds 42, 123 and 2026** configuration: average raw logits, fit one Vector Scaling model, select one threshold. It does not switch to another configuration if a new training run changes the ranking; inspect and report any such difference.

To compare these freshly refitted policies:

```bash
python scripts/stage_comparisons.py --factorial outputs/factorial \
  --ensemble outputs/ensemble_evaluation/ensemble.csv --output outputs/refitted_comparison_inputs
python scripts/compare_policies.py --inputs outputs/refitted_comparison_inputs --output outputs/refitted_statistics
```

Keep refitted outputs distinct from the original frozen predictions when verifying historical results.

## Figures, supplementary assets and sensitivity

```bash
python scripts/make_figures.py --inputs data --only curves --output outputs/reviewer_curves
python scripts/make_figures.py --inputs data --only plates \
  --dataset-root /path/to/image_dataset --output outputs/reviewer_plates
python scripts/make_supplement.py --inputs data/review --factorial outputs/factorial --output outputs/supplement
python scripts/development_sensitivity.py --manifest data/experiments.csv --output outputs/sensitivity
```

The two reviewer figures use actual measured histories and actual dataset images; no images or curves are synthesized. Supplementary commands generate S1 reliability assets, S2 seed-policy/risk–coverage assets, S3 confusion/release tables and S4 counts. They do not overwrite the existing supplementary PDF. See `docs/SUPPLEMENT_NOTES.md` for required textual corrections. The 100-repartition sensitivity analysis uses only the pooled calibration/threshold cohort and fixed network logits; it may take substantial CPU time.

The optional CUDA benchmark is `python scripts/benchmark_inference.py --output outputs/benchmark`. As in the source notebook, this measures network architecture forward passes with random weights and synthetic inputs after 50 warm-ups and 300 timed passes. It does not regenerate recorded training times or accuracy.

## Full training route

Requires the actual dataset and original split manifest. Download of torchvision pretrained weights is also required on first use.

```bash
python scripts/prepare_data.py --existing-splits data/splits/urocultivo_extension_split_v1.csv \
  --image-root /path/to/image_dataset --output-dir outputs/verified_splits
python scripts/train_models.py --split-csv outputs/verified_splits/urocultivo_extension_split_v1.csv \
  --image-root /path/to/image_dataset --output-dir outputs/training --pilot
python scripts/train_models.py --split-csv outputs/verified_splits/urocultivo_extension_split_v1.csv \
  --image-root /path/to/image_dataset --output-dir outputs/training
python scripts/export_logits.py --split-manifest outputs/verified_splits/urocultivo_extension_split_v1.csv \
  --image-root /path/to/image_dataset --checkpoint-root outputs/training/02_checkpoints --output outputs/logits
```

Use `outputs/logits/experiments.csv` for subsequent calibration commands. Training runs from scratch and refuses to overwrite completed/nonempty run directories. The original interrupted-run recovery did not preserve all data-loader RNG state; exact continuation is therefore not claimed.

## Interpretation

Automatic release requires both a Negative argmax and calibrated `P(Negative) >= threshold`; everything else is referred. Three training seeds are not independent cohorts. Patient/specimen/acquisition groups are unknown, and the evaluation split had been inspected previously. Zero observed unsafe releases do not establish zero population risk. Neither marginal bootstrap intervals nor unadjusted tests establish ensemble superiority. See `docs/PROTOCOL.md` for denominators, seeds and endpoints.
