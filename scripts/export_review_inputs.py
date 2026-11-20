"""Export minimal review inputs from the original Drive experiment, without modifying it.

Run in Colab after mounting Drive, or locally against a downloaded experiment.
Only model outputs, numeric labels and sample IDs are exported, not absolute image
paths. Review the sample identifiers and dataset sharing rights before publishing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from urineculture.statistics import normalize_predictions, validate_study_cohort, align_comparator


def export_review_inputs(project, output):
    project, output = Path(project), Path(output)
    if output.resolve() == project.resolve() or project.resolve() in output.resolve().parents:
        raise ValueError("Export to a new directory outside the original experiment folder")
    policies, audit = {}, []
    ensemble_file = project / "10_ensemble_resnet18_focal_vector_scaling/03_predictions/final_test_ensemble_frozen_policy_decisions.csv"
    base = normalize_predictions(pd.read_csv(ensemble_file, dtype={"sample_id": str}, float_precision="round_trip"),
                                pred_col="predicted_label_ensemble_vector_scaling",
                                prob_col="prob_0_negative_ensemble_vector_scaling")
    validate_study_cohort(base)
    policies["ensemble"] = base
    audit.append({"policy": "ensemble", "source": str(ensemble_file.relative_to(project)),
                  "source_sha256": hashlib.sha256(ensemble_file.read_bytes()).hexdigest()})
    for seed in (42, 123, 2026):
        sub = Path(f"resnet18/focal_loss/gamma_2/seed_{seed}")
        path = project / "05_calibrated_predictions_extended" / sub / "final_test_extended_posthoc.csv"
        threshold_file = project / "04_thresholds_extended" / sub / "vector_scaling/selected_threshold.json"
        threshold = float(json.loads(threshold_file.read_text())["selected_threshold"])
        name = f"ResNet_individual_seed_{seed}"
        policies[name] = normalize_predictions(pd.read_csv(path, dtype={"sample_id": str}, float_precision="round_trip"),
            pred_col="predicted_label_vector_scaling", prob_col="prob_0_negative_vector_scaling",
            release_col=None, threshold=threshold)
        audit.append({"policy": name, "source": str(path.relative_to(project)),
                      "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "threshold_source": str(threshold_file.relative_to(project)),
                      "threshold_sha256": hashlib.sha256(threshold_file.read_bytes()).hexdigest()})
        path = project / f"06_results/final_test/sample_level_decisions/swin_t/cross_entropy/seed_{seed}/uncalibrated_final_test_decisions.csv"
        name = f"Swin_uncalibrated_seed_{seed}"
        policies[name] = normalize_predictions(pd.read_csv(path, dtype={"sample_id": str}, float_precision="round_trip"),
            release_col="auto_release_negative", threshold_col="frozen_threshold")
        audit.append({"policy": name, "source": str(path.relative_to(project)),
                      "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for name in policies:
        validate_study_cohort(policies[name])
        if name != "ensemble":
            policies[name] = align_comparator(base, policies[name])
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f"{name}.csv").exists() for name in policies):
        raise FileExistsError("The output already contains review inputs; choose a new directory")
    for name, data in policies.items():
        data.to_csv(output / f"{name}.csv", index=False)
    # The notebook bootstrap uses existing ensemble row order; sorting would change draws.
    (output / "input_provenance.json").write_text(json.dumps({
        "source_cells_zero_based": [99, 118, 121, 122, 123],
        "row_order": "Original ensemble row order; comparators aligned by sample_id",
        "labels": {"0": "Negative", "1": "Positive", "2": "Uncertain"},
        "inputs": audit}, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Exported verified paired inputs to {export_review_inputs(args.project, args.output)}")


if __name__ == "__main__":
    main()
