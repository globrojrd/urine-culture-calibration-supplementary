# Notebook extraction provenance

Prepared 20 November 2026 from `CalibracionExt (1).ipynb` supplied by the authors. Source SHA-256: `c939beca6db3afc81df20d964806b5b1ca116abfa6aa3d537fa99f8d478dd57d`. All 127 code cells have null execution counters. Cell position and saved output establish correspondence, not a complete execution chronology. Cell numbers below are **zero-based**; add one for their ordinal position in the notebook.

| Source cells | Decision / public counterpart |
|---|---|
| 0–54 | Exclude prior pipelines, unrelated image-restoration work, ViT/KAN experiments and presentation plots. Some old plotting data are explicitly simulated; none enter this extraction. |
| 55–59 | Exclude preliminary three-part extension; superseded by the five-part protocol. |
| 60–64 | Keep dataset audit, fixed split and data loaders in `data.py`; replace Drive-specific setup with parameters. |
| 65–69 | Keep completed ResNet CE/LS runs. Discard the interrupted LS seed-2026 run; retain its fresh restart. No destructive cleanup is copied. |
| 70–77 | Keep gamma pilot and final ResNet/EfficientNet/Swin training in `training.py`. |
| 78 | Keep checkpoint-to-logit export, strict class mapping, row-order checks and hashes. |
| 79–82, 84–89 | Keep six calibrators, Negative-event metrics, threshold rule and factorial selection/evaluation. Repeated functions are consolidated. |
| 83 | Keep correction for non-estimable calibration slopes/intercepts; do not reuse earlier uncorrected fits. |
| 90–97 | Inspection/summary outputs are evidence, not additional computational steps. Individual-seed mean coverage 27.11% is not the final ensemble coverage 26%. |
| 98 | Calls missing external `corregir_ensemble_vector_scaling_colab.py`. Saved output is available; original script source is not. Ensemble reconstruction is explicitly labeled. |
| 99–103 | Keep single-policy bootstrap, reliability-bin estimator and architecture benchmark; omit repeated Drive setup. |
| 104 | Exclude candidate listing that points all three purported seed rows to the same ensemble CSV. It is not evidence of seed-specific policies. |
| 105–107 | Keep actual confusion/operating tables and seed risk–coverage logic. |
| 108–109 | Environment and file inspections are provenance. |
| 110–112 | Keep fixed-checkpoint development sensitivity, separate numerical helpers and mean-logit ensemble construction. |
| 113–115 | Supplement/file packaging and downloads; replace with explicit exporters. Old ZIP output reported missing files and cannot establish complete availability. |
| 116–120 | Initial review paths, bounds, history checks and preparatory statistics; use the finalized aligned comparison below. |
| 121–124 | Keep exact six-policy alignment, final McNemar/Holm and 3,000 paired bootstrap. |
| 125 | Keep real dataset panels and measured training/validation curves. |
| 126 | Download operation only; not an analysis. |

The refactor adds explicit inputs, main guards, non-overwriting training, sample-identity validation, frozen-artifact hashes, strict JSON for undefined quantities, and convergence failures that stop rather than being silently accepted. These safeguards do not erase previous test inspection or unknown grouping. Numerical reanalysis from recovered predictions and logits was completed on 20 November 2026; whole-network training was not repeated.

For the missing corrected ensemble script, averaging is supported by cell112, Vector Scaling by cell85 and the final sensitivity implementation, and the threshold rule by cells80/86. Exact source recovery is not claimed. If the missing script is found, preserve it privately and compare its estimator/settings before changing the documented reconstruction.

The recovery archive contained 1,777 experiment files, all matching their recorded SHA-256 hashes. It recovered the exact original split manifest, 27 completed 20-epoch histories, all 81 logits tables and the original ensemble calibrator/threshold. Those fitted parameters provide a separate no-refit replay route. The public release strips absolute machine paths and supplies only the portable subset needed for the analyses.

Historical threshold replay requires attention to CSV parsing: changing the parser from the original pandas default to `round_trip` moves one exactly-on-threshold probability below the threshold by 1.11e-16 in 15 development conditions. Replaying the historical reader reproduces every stored count and decision. Final evaluation decisions are unaffected. The newly fitted pipeline uses its own jointly generated probabilities/thresholds and matches the historical counts; no threshold tolerance was introduced to force agreement. See `results/serialization_boundary_audit.json`.
