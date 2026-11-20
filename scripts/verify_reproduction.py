"""Verify published numerical results from the supplied saved study artifacts.

Uses frozen predictions and logits; does not retrain networks or require a GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from urineculture.statistics import (COMPARATOR_NAMES, read_predictions, paired_comparisons,
                                     single_policy_bootstrap, false_negative_analysis)
from urineculture.ensemble import evaluate_ensemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--expected", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reproduction_check"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty verification output directory")
    checksums = json.loads((args.data / "file_checksums.json").read_text())
    for relative, digest in checksums["sha256"].items():
        if hashlib.sha256((args.data / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Input checksum mismatch: {relative}")
    base = read_predictions(args.data / "review/ensemble.csv")
    comparators = {name: read_predictions(args.data / "review" / f"{name}.csv") for name in COMPARATOR_NAMES}
    metric, exact, iterations = paired_comparisons(base, comparators)
    single, single_iterations = single_policy_bootstrap(base)
    safety = false_negative_analysis(base)
    frames = {"paired_bootstrap_metric_differences": metric, "paired_exact_tests_holm": exact,
              "single_policy_bootstrap_summary": single, "false_negative_analysis": safety}
    for name, frame in frames.items():
        reference = pd.read_csv(args.expected / f"{name}.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), reference,
                                      check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-14)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(args.output / f"{name}.csv", index=False)
    iterations.to_csv(args.output / "paired_bootstrap_iterations.csv", index=False)
    single_iterations.to_csv(args.output / "single_policy_bootstrap_iterations.csv", index=False)
    metrics = evaluate_ensemble(args.data / "experiments.csv", args.data / "frozen_ensemble.json",
                                args.output / "original_frozen_ensemble")
    replay = read_predictions(args.output / "original_frozen_ensemble/ensemble.csv").set_index("sample_id").loc[base.sample_id]
    np.testing.assert_array_equal(replay.predicted_label, base.predicted_label)
    np.testing.assert_array_equal(replay.released, base.released)
    np.testing.assert_allclose(replay.prob_negative, base.prob_negative, rtol=1e-12, atol=1e-14)
    report = {"status": "PASS", "input_files_verified": len(checksums["sha256"]),
              "paired_test_rows": len(exact), "paired_metric_rows": len(metric),
              "paired_bootstrap_difference_rows": len(iterations), "single_policy_resamples": len(single_iterations),
              "frozen_ensemble_predictions_and_decisions_match": True,
              "releases": metrics["number_of_releases"], "unsafe_releases": metrics["unsafe_releases"],
              "network_retraining": False, "calibrator_refitting": False}
    (args.output / "verification_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
