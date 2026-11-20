"""Recompute final paired tests, individual intervals and safety endpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urineculture.statistics import (COMPARATOR_NAMES, false_negative_analysis,
    paired_comparisons, read_predictions, single_policy_bootstrap, validate_study_cohort,
    confusion_tables, operating_summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True,
                        help="Directory containing ensemble.csv and six comparator CSVs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {"ensemble": args.inputs / "ensemble.csv", **{name: args.inputs / f"{name}.csv" for name in COMPARATOR_NAMES}}
    base = read_predictions(files["ensemble"])
    validate_study_cohort(base)
    comparators = {name: read_predictions(files[name]) for name in COMPARATOR_NAMES}
    metrics, tests, iterations = paired_comparisons(base, comparators)
    summary, boot = single_policy_bootstrap(base)
    safety = false_negative_analysis(base)
    multiclass, operational = confusion_tables(base)
    operating = operating_summary({"ensemble": base, **comparators})
    if args.output.resolve() == args.inputs.resolve():
        raise ValueError("Use a separate output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in {"paired_bootstrap_metric_differences": metrics,
                        "paired_exact_tests_holm": tests,
                        "paired_bootstrap_iterations": iterations,
                        "single_policy_bootstrap_summary": summary,
                        "single_policy_bootstrap_iterations": boot,
                        "false_negative_analysis": safety,
                        "multiclass_confusion_matrix": multiclass,
                        "operational_release_table": operational,
                        "policy_operating_summary": operating}.items():
        frame.to_csv(args.output / f"{name}.csv", index=False)
    metadata = {
        "paired_bootstrap": {"resamples": 3000, "seed": 20261112,
                             "strata": ["Negative", "Positive", "Uncertain"],
                             "same_draws_for_all_policies": True, "intervals": "marginal percentile 95%; not multiplicity-adjusted"},
        "single_policy_bootstrap": {"resamples": 2000, "seed": 20260730,
                                    "strata": ["Negative", "non-Negative"],
                                    "warning": "Degenerate NPV bootstrap does not establish zero population risk"},
        "exact_tests": "Two-sided McNemar; Holm separately over 6 Accuracy tests and 6 TNRR tests",
        "limitations": ["Fitted policies are fixed", "Test images previously inspected",
                        "Unknown patient/acquisition groups", "Training seeds are not independent cohorts",
                        "No multiplicity-adjusted claim follows from marginal bootstrap intervals"],
        "inputs": {name: {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in files.items()}}
    (args.output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(tests.to_string(index=False))
    print(f"Saved recomputed analyses to {args.output}")


if __name__ == "__main__":
    main()
