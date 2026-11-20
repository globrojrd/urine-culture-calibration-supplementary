"""Three-checkpoint mean-logit ensemble with one frozen calibration policy.

Reconstruction: averaging from notebook cell 112, Vector Scaling from cell 85,
threshold rule from cells 80/86. Cell 98 called an external correction script
whose source was not included in the supplied notebook. This is a documented
reconstruction, not a recovered copy of that external script.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import Calibrator, fit_calibrator
from .factorial import LOGIT_COLUMNS, read_logits
from .metrics import evaluate_predictions
from .policy import evaluate_policy, release_mask, select_threshold

SEEDS = (42, 123, 2026)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean_aligned_logits(paths):
    """Align by sample ID, verify labels, and preserve seed-42 row order."""
    if len(paths) != 3:
        raise ValueError("Exactly three checkpoint-logit files are required.")
    frames = [read_logits(path) for path in paths]
    reference = frames[0]
    ids = reference["sample_id"].tolist()
    values = []
    for frame in frames:
        if set(frame["sample_id"]) != set(ids):
            raise ValueError("The three checkpoints did not evaluate the same images.")
        frame = frame.set_index("sample_id").loc[ids].reset_index()
        if not np.array_equal(frame["true_label"], reference["true_label"]):
            raise ValueError("Reference labels differ between checkpoints.")
        values.append(frame[list(LOGIT_COLUMNS)].to_numpy(dtype=np.float64))
    result = reference[["sample_id", "true_label"]].copy()
    result[list(LOGIT_COLUMNS)] = np.stack(values).mean(axis=0)
    return result


def _paths(manifest, split):
    path = Path(manifest).resolve()
    frame = pd.read_csv(path)
    required = {"architecture", "loss", "seed", split}
    if not required.issubset(frame):
        raise ValueError(f"Missing manifest fields: {sorted(required - set(frame))}")
    frame = frame[(frame.architecture == "resnet18") & (frame.loss == "focal_loss")]
    if len(frame) != 3 or set(frame.seed) != set(SEEDS) or frame.seed.duplicated().any():
        raise ValueError("Manifest must contain one ResNet-18/focal row per selected seed.")
    return [path.parent / str(frame.set_index("seed").loc[s, split]) for s in SEEDS]


def _json(path, obj):
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value
    Path(path).write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + "\n")


def fit_ensemble(manifest, output):
    """Fit using development only; never open final-test logits here."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    calibration_paths = _paths(manifest, "calibration")
    threshold_paths = _paths(manifest, "threshold_selection")
    cal = mean_aligned_logits(calibration_paths)
    dev = mean_aligned_logits(threshold_paths)
    if set(cal.sample_id) & set(dev.sample_id):
        raise ValueError("Calibration and threshold-selection images overlap.")
    calibrator = fit_calibrator("vector_scaling", cal[list(LOGIT_COLUMNS)], cal.true_label)
    probabilities = calibrator.transform(dev[list(LOGIT_COLUMNS)])
    policy, candidates = select_threshold(dev.true_label, probabilities)
    state = {"calibrator": calibrator.to_dict(), "policy": policy,
             "architecture": "resnet18", "loss": "focal_loss", "seeds": list(SEEDS),
             "averaging": "arithmetic mean of raw logits before one calibrator",
             "experiment_manifest_sha256": sha256(manifest),
             "calibration_sample_ids": cal.sample_id.tolist(),
             "threshold_sample_ids": dev.sample_id.tolist(),
             "source_hashes": {str(p): sha256(p) for p in calibration_paths + threshold_paths},
             "provenance": "Documented reconstruction; external cell-98 source unavailable."}
    _json(output / "frozen_ensemble.json", state)
    pd.DataFrame(candidates).to_csv(output / "threshold_candidates.csv", index=False)
    cal.to_csv(output / "calibration_mean_logits.csv", index=False)
    dev.to_csv(output / "threshold_selection_mean_logits.csv", index=False)
    return state


def evaluate_ensemble(manifest, frozen_policy, output):
    """Evaluate a previously saved policy without fitting or selecting anything."""
    state = json.loads(Path(frozen_policy).read_text())
    if sha256(manifest) != state["experiment_manifest_sha256"]:
        raise ValueError("Experiment manifest changed after ensemble development was frozen.")
    paths = _paths(manifest, "final_test")
    test = mean_aligned_logits(paths)
    if set(test.sample_id) & set(state["calibration_sample_ids"] + state["threshold_sample_ids"]):
        raise ValueError("Evaluation images overlap with ensemble development images.")
    if test.true_label.value_counts().to_dict() != {0: 50, 1: 50, 2: 50}:
        raise ValueError("The original evaluation protocol requires 50 images per class.")
    calibrator = Calibrator.from_dict(state["calibrator"])
    p = calibrator.transform(test[list(LOGIT_COLUMNS)])
    threshold = state["policy"]["selected_threshold"]
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    predictions = test[["sample_id", "true_label"]].copy()
    predictions["predicted_label"] = p.argmax(axis=1)
    predictions["prob_negative"] = p[:, 0]
    predictions["prob_positive"] = p[:, 1]
    predictions["prob_uncertain"] = p[:, 2]
    predictions["released"] = release_mask(p, threshold)
    predictions["operating_threshold"] = threshold
    predictions.to_csv(output / "ensemble.csv", index=False)
    metrics = {**evaluate_predictions(test.true_label, p),
               **evaluate_policy(test.true_label, p, threshold),
               "frozen_policy_sha256": sha256(frozen_policy),
               "test_logit_hashes": {str(p): sha256(p) for p in paths}}
    _json(output / "ensemble_metrics.json", metrics)
    return metrics
