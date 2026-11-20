# Required study inputs

No observation-level study data are fabricated or bundled as placeholders. The exporters create the files below from the authors' saved experiment.

| Path | Required content |
|---|---|
| `splits/urocultivo_extension_split_v1.csv` | 1,500 original sample identities, filenames, class labels, five split memberships and class-relative paths. |
| `experiments.csv` | 27 rows: `architecture,loss,seed,calibration,threshold_selection,final_test`; logit paths relative to this file. |
| `logits/.../*.csv` | 81 tables, each 150 rows: `sample_id,true_label,logit_0_negative,logit_1_positive,logit_2_uncertain`. |
| `histories/resnet18_focal_seed_42.csv` | Original 20-epoch selected-configuration history; analogous files for seeds 123 and 2026. |
| `review/ensemble.csv` | 150 frozen ensemble decisions in their original stored row order. |
| `review/ResNet_individual_seed_42.csv` | Individual ResNet/focal/VectorScaling decisions; analogous files for seeds 123 and 2026. |
| `review/Swin_uncalibrated_seed_42.csv` | Swin/CE/uncalibrated decisions; analogous files for seeds 123 and 2026. |
| `artifact_provenance.json`, `review/input_provenance.json` | Hashes and source correspondence produced by the exporters. |

All seven review CSVs require `sample_id,true_label,predicted_label,prob_negative,released,operating_threshold`. Classes are Negative=0, Positive=1, Uncertain=2. `released` is a boolean or 0/1. Each file must contain the same 150 images, 50 per class. Thresholds must be constant within a policy, and stored release flags must agree with the frozen decision rule. The comparison loader aligns by sample ID and checks reference labels; it never pairs rows by position alone.

Histories require `epoch,train_loss,validation_loss,train_macro_f1,validation_macro_f1`. Image panels additionally require the original dataset files. Raw images and model checkpoints are not required for the frozen-prediction statistical route; they are required for image panels/full training or inference, respectively.
