"""Auditable two-phase factorial evaluation, extracted from cells 79–89.

Development and evaluation are separate entry points. The historical test split
was inspected previously; separating commands cannot undo that limitation.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .calibration import METHODS, Calibrator, fit_calibrator
from .metrics import evaluate_predictions, calibration_bins
from .policy import evaluate_policy, select_threshold, release_mask, risk_coverage_curve

LOGIT_COLUMNS = ("logit_0_negative", "logit_1_positive", "logit_2_uncertain")
PROBABILITY_COLUMNS = ("p_negative", "p_positive", "p_uncertain")
ARCHITECTURES = ("resnet18", "efficientnetv2_s", "swin_t")
LOSSES = ("cross_entropy", "label_smoothing", "focal_loss")
SEEDS = (42, 123, 2026)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, obj):
    """Write strict JSON, using null for undefined rates, never nonstandard NaN."""
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        return value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_logits(path, expected_rows=150):
    frame = pd.read_csv(path, dtype={"sample_id": str}, float_precision="round_trip")
    required = ["sample_id", "true_label", *LOGIT_COLUMNS]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Logit table {path} requires columns {required}.")
    if frame[required].isna().any().any() or frame["sample_id"].duplicated().any():
        raise ValueError(f"Missing values or repeated sample IDs in {path}.")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {path}, found {len(frame)}.")
    if not np.isin(frame.true_label, [0, 1, 2]).all():
        raise ValueError(f"Invalid class labels in {path}.")
    if not np.isfinite(frame[list(LOGIT_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite logits in {path}.")
    return frame


def load_experiments(manifest, require_full_factorial=True):
    manifest = Path(manifest).resolve()
    frame = pd.read_csv(manifest)
    needed = ["architecture", "loss", "seed", "calibration", "threshold_selection", "final_test"]
    if not set(needed).issubset(frame.columns) or frame[needed].isna().any().any():
        raise ValueError(f"Experiment manifest requires complete columns {needed}.")
    if frame.duplicated(["architecture", "loss", "seed"]).any():
        raise ValueError("Experiment manifest repeats an architecture/loss/seed.")
    if not frame.architecture.isin(ARCHITECTURES).all() or not frame.loss.isin(LOSSES).all():
        raise ValueError("Unknown architecture or loss in experiment manifest.")
    if not frame.seed.isin(SEEDS).all():
        raise ValueError("Experiment seeds must be 42, 123, or 2026.")
    if require_full_factorial and len(frame) != 27:
        raise ValueError("The full factorial analysis requires all 27 checkpoints.")
    experiments = frame.to_dict("records")
    for row in experiments:
        row["seed"] = int(row["seed"])
        for split in ("calibration", "threshold_selection", "final_test"):
            row[split] = (manifest.parent / str(row[split])).resolve()
    return experiments


def experiment_directory(output, row):
    return Path(output) / row["architecture"] / row["loss"] / f"seed_{row['seed']}"


def _identity(row, method):
    return {key: row[key] for key in ("architecture", "loss", "seed")} | {"calibration_method": method}


def save_predictions(path, frame, p, threshold, identity):
    out = frame[["sample_id", "true_label"]].copy()
    out[list(PROBABILITY_COLUMNS)] = p
    out["predicted_label"] = p.argmax(axis=1)
    out["released"] = release_mask(p, threshold).astype(int)
    out["prob_negative"] = p[:, 0]
    out["operating_threshold"] = float(threshold)
    for name, value in identity.items():
        out[name] = value
    out.to_csv(path, index=False)


def rank_development(individual):
    """Sample SD across seeds is descriptive, not independent replication."""
    grouping = ["architecture", "loss", "calibration_method"]
    if individual.duplicated(grouping + ["seed"]).any():
        raise ValueError("Repeated seed in development summaries.")
    records = []
    for keys, group in individual.groupby(grouping, sort=True):
        if set(group.seed) != set(SEEDS) or len(group) != 3:
            raise ValueError("Ranking requires the same three seeds per configuration.")
        row = dict(zip(grouping, keys))
        row["unsafe_releases_total"] = int(group.unsafe_releases.sum())
        row["zero_unsafe_across_all_three_seeds"] = bool((group.unsafe_releases == 0).all())
        for metric in ("coverage", "negative_fixed_ece", "negative_binary_brier",
                       "negative_binary_nll", "macro_f1", "true_negative_release_rate"):
            row[metric + "_mean"] = float(group[metric].mean())
            row[metric + "_std"] = float(group[metric].std(ddof=1))
        records.append(row)
    aggregate = pd.DataFrame(records)
    criteria = [("coverage_mean", False), ("negative_fixed_ece_mean", True),
                ("negative_binary_brier_mean", True), ("negative_binary_nll_mean", True),
                ("macro_f1_mean", False), ("coverage_std", True)]
    ranking = aggregate.sort_values(
        ["zero_unsafe_across_all_three_seeds"] + [x[0] for x in criteria] + grouping,
        ascending=[False] + [x[1] for x in criteria] + [True] * 3).reset_index(drop=True)
    ranking.insert(0, "lexicographic_rank", np.arange(1, len(ranking) + 1))
    current = aggregate[aggregate.zero_unsafe_across_all_three_seeds].copy()
    if current.empty:
        raise ValueError("No configuration meets the development constraint.")
    trace = [{"criterion": "zero_unsafe_across_all_three_seeds", "number_before": len(aggregate),
              "number_after": len(current), "best_value": True}]
    for metric, ascending in criteria:
        best = current[metric].min() if ascending else current[metric].max()
        remaining = current[np.isclose(current[metric], best, rtol=0, atol=1e-12)]
        trace.append({"criterion": metric, "number_before": len(current),
                      "number_after": len(remaining), "best_value": float(best)})
        current = remaining
    winner = current.sort_values(grouping).iloc[0].to_dict()
    # The historical notebook compared the exact-sort and tolerance-trace winners.
    if tuple(winner[k] for k in grouping) != tuple(ranking.iloc[0][k] for k in grouping):
        raise RuntimeError("Exact and tolerance-based ranking disagree; inspect numerical ties.")
    return ranking, winner, trace


def develop(manifest, output, require_full_factorial=True):
    """Fit all calibrators and thresholds, then freeze ranking before test access."""
    experiments = load_experiments(manifest, require_full_factorial)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "development_frozen.json").exists():
        raise FileExistsError("Development is already frozen; use a fresh output directory to refit.")
    summaries, input_hashes, frozen_files = [], [], []
    reference_labels = {}
    for row in experiments:
        calibration = read_logits(row["calibration"])
        threshold = read_logits(row["threshold_selection"])
        if set(calibration.sample_id) & set(threshold.sample_id):
            raise ValueError("Calibration and threshold-selection samples overlap.")
        for split, frame in (("calibration", calibration), ("threshold_selection", threshold)):
            ids_labels = dict(zip(frame.sample_id, frame.true_label))
            if split in reference_labels and reference_labels[split] != ids_labels:
                raise ValueError(f"{split} images or labels differ between checkpoints.")
            reference_labels[split] = ids_labels
        zcal, ycal = calibration[list(LOGIT_COLUMNS)].to_numpy(), calibration.true_label.to_numpy()
        zthr, ythr = threshold[list(LOGIT_COLUMNS)].to_numpy(), threshold.true_label.to_numpy()
        for method in METHODS:
            destination = experiment_directory(output, row) / method
            destination.mkdir(parents=True, exist_ok=True)
            calibrator = fit_calibrator(method, zcal, ycal)
            p = calibrator.transform(zthr)
            selection, candidates = select_threshold(ythr, p)
            identity = _identity(row, method)
            save_json(destination / "calibrator.json", calibrator.to_dict())
            save_json(destination / "threshold.json", selection)
            for filename in ("calibrator.json", "threshold.json"):
                artifact = destination / filename
                frozen_files.append({"path": artifact.relative_to(output).as_posix(),
                                     "sha256": sha256_file(artifact)})
            pd.DataFrame(candidates).to_csv(destination / "threshold_candidates.csv", index=False)
            save_predictions(destination / "threshold_selection_predictions.csv", threshold, p,
                             selection["selected_threshold"], identity)
            summaries.append(identity | evaluate_predictions(ythr, p) | selection)
        input_hashes.append(_identity(row, "all") | {
            "calibration_sha256": sha256_file(row["calibration"]),
            "threshold_selection_sha256": sha256_file(row["threshold_selection"])})
    summaries = pd.DataFrame(summaries)
    summaries.to_csv(output / "development_by_seed.csv", index=False)
    ranking, winner, trace = rank_development(summaries)
    ranking.to_csv(output / "development_ranking.csv", index=False)
    save_json(output / "selection_trace.json", trace)
    save_json(output / "development_frozen.json", {
        "experiment_manifest_sha256": sha256_file(manifest), "winner": winner,
        "methods": list(METHODS), "input_hashes": input_hashes, "frozen_files": frozen_files,
        "development_sample_labels": reference_labels,
        "limitation": "Historical evaluation images were previously inspected; ranking is exploratory."})
    return winner


def evaluate(manifest, output, require_full_factorial=True):
    """Apply frozen files to final_test without fitting or reselecting anything."""
    output = Path(output)
    frozen = json.loads((output / "development_frozen.json").read_text())
    if sha256_file(manifest) != frozen["experiment_manifest_sha256"]:
        raise ValueError("Experiment manifest changed after development was frozen.")
    for artifact in frozen["frozen_files"]:
        if sha256_file(output / artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"Frozen calibrator/threshold was modified: {artifact['path']}")
    summaries, reference_labels = [], None
    development_ids = set().union(*(set(mapping) for mapping in frozen["development_sample_labels"].values()))
    for row in load_experiments(manifest, require_full_factorial):
        frame = read_logits(row["final_test"])
        if development_ids & set(frame.sample_id):
            raise ValueError("Evaluation images overlap development images.")
        labels = dict(zip(frame.sample_id, frame.true_label))
        if reference_labels is not None and reference_labels != labels:
            raise ValueError("Evaluation sample IDs/labels differ between checkpoints.")
        reference_labels = labels
        z, y = frame[list(LOGIT_COLUMNS)].to_numpy(), frame.true_label.to_numpy()
        for method in METHODS:
            destination = experiment_directory(output, row) / method
            calibrator = Calibrator.from_dict(json.loads((destination / "calibrator.json").read_text()))
            threshold = json.loads((destination / "threshold.json").read_text())["selected_threshold"]
            p = calibrator.transform(z)
            identity = _identity(row, method)
            save_predictions(destination / "final_test_predictions.csv", frame, p, threshold, identity)
            summaries.append(identity | evaluate_predictions(y, p) | evaluate_policy(y, p, threshold))
            bins = calibration_bins(p[:, 0], (y == 0).astype(int))[2]
            pd.DataFrame(bins).to_csv(destination / "final_test_reliability_bins.csv", index=False)
            pd.DataFrame(risk_coverage_curve(y, p)).to_csv(
                destination / "final_test_risk_coverage_descriptive.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "final_test_by_seed.csv", index=False)
    metrics = ["accuracy", "macro_f1", "coverage", "true_negative_release_rate",
               "negative_fixed_ece", "negative_adaptive_ece", "negative_binary_brier",
               "negative_binary_nll", "negative_auroc"]
    aggregate = summary.groupby(["architecture", "loss", "calibration_method"])[metrics].agg(["mean", "std"])
    aggregate.columns = ["_".join(pair) for pair in aggregate.columns]
    aggregate.to_csv(output / "final_test_by_configuration.csv")
    return summary
