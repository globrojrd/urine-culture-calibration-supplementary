"""Development-only sensitivity, extracted from notebook cells 110-112 (zero-based).

This historical analysis used EPS=1e-12 and right-closed ECE bins. These helpers
are intentionally isolated from the primary calibration implementation because
silently substituting its numerical conventions can change a tied ranking.
All 27 checkpoints are frozen. Only calibration and threshold-selection inputs
are read; evaluation data are not used by this module.
"""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

ARCHITECTURES = ["resnet18", "efficientnetv2_s", "swin_t"]
LOSSES = ["cross_entropy", "label_smoothing", "focal_loss"]
SEEDS = [42, 123, 2026]
CALIBRATION_METHODS = ["uncalibrated", "temperature_scaling", "dirichlet", "vector_scaling", "beta_negative", "isotonic_negative"]
LOGIT_COLUMNS = ["logit_0_negative", "logit_1_positive", "logit_2_uncertain"]
NEGATIVE_LABEL, POSITIVE_LABEL, UNCERTAIN_LABEL = 0, 1, 2
NUM_CLASSES, FIXED_ECE_BINS = 3, 15
PROBABILITY_EPSILON = 1e-12
DIRICHLET_C, DIRICHLET_MAX_ITER, DIRICHLET_SOLVER = 1.0, 5000, "lbfgs"
VECTOR_SCALING_L2, BETA_CALIBRATION_C = 1e-3, 1.0

def stable_log_softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    return logits - logsumexp(logits, axis=1, keepdims=True)

def stable_softmax(logits):
    return np.exp(stable_log_softmax(logits))

def normalize_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, PROBABILITY_EPSILON, 1.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    return probabilities / row_sums

def multiclass_nll(probabilities, labels):
    probabilities = normalize_probabilities(probabilities)
    selected = probabilities[np.arange(len(labels)), labels]
    selected = np.clip(selected, PROBABILITY_EPSILON, 1.0)
    return float(-np.mean(np.log(selected)))

def binary_nll(probabilities, targets):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    probabilities = np.clip(probabilities, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    losses = -(targets * np.log(probabilities) + (1.0 - targets) * np.log(1.0 - probabilities))
    return float(np.mean(losses))

def binary_brier(probabilities, targets):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    return float(np.mean((probabilities - targets) ** 2))

def fixed_ece(probabilities, targets, number_of_bins=FIXED_ECE_BINS):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, number_of_bins + 1)
    number_of_samples = len(targets)
    ece = 0.0
    for bin_index in range(number_of_bins):
        lower_bound = bin_edges[bin_index]
        upper_bound = bin_edges[bin_index + 1]
        if bin_index == 0:
            in_bin = (probabilities >= lower_bound) & (probabilities <= upper_bound)
        else:
            in_bin = (probabilities > lower_bound) & (probabilities <= upper_bound)
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        mean_probability = float(np.mean(probabilities[in_bin]))
        observed_frequency = float(np.mean(targets[in_bin]))
        ece += count / number_of_samples * abs(observed_frequency - mean_probability)
    return float(ece)

def fit_temperature_scaling(calibration_logits, calibration_labels):

    def objective(log_temperature):
        temperature = math.exp(float(log_temperature))
        probabilities = stable_softmax(calibration_logits / temperature)
        return multiclass_nll(probabilities, calibration_labels)
    result = minimize_scalar(objective, method='bounded', bounds=(-5.0, 5.0), options={'xatol': 1e-10, 'maxiter': 1000})
    if not result.success:
        raise RuntimeError(f'Temperature Scaling did not converge: {result.message}')
    return float(math.exp(float(result.x)))

def fit_dirichlet_calibration(calibration_logits, calibration_labels):
    log_probabilities = stable_log_softmax(calibration_logits)
    model = LogisticRegression(C=DIRICHLET_C, penalty='l2', solver=DIRICHLET_SOLVER, max_iter=DIRICHLET_MAX_ITER, fit_intercept=True, random_state=42)
    model.fit(log_probabilities, calibration_labels)
    return model

def apply_dirichlet_calibration(model, logits):
    log_probabilities = stable_log_softmax(logits)
    raw_probabilities = model.predict_proba(log_probabilities)
    ordered_probabilities = np.zeros((len(logits), NUM_CLASSES), dtype=np.float64)
    for source_index, class_label in enumerate(model.classes_):
        ordered_probabilities[:, int(class_label)] = raw_probabilities[:, source_index]
    return normalize_probabilities(ordered_probabilities)

def vector_scaling_transform(logits, scales, biases):
    transformed_logits = logits * scales.reshape(1, NUM_CLASSES) + biases.reshape(1, NUM_CLASSES)
    return softmax(transformed_logits, axis=1)

def fit_vector_scaling(calibration_logits, calibration_labels, l2_strength=VECTOR_SCALING_L2):
    calibration_logits = np.asarray(calibration_logits, dtype=np.float64)
    calibration_labels = np.asarray(calibration_labels, dtype=np.int64)

    def objective(parameters):
        log_scales = parameters[:NUM_CLASSES]
        raw_biases = parameters[NUM_CLASSES:]
        scales = np.exp(log_scales)
        biases = raw_biases - np.mean(raw_biases)
        probabilities = vector_scaling_transform(calibration_logits, scales, biases)
        nll = multiclass_nll(probabilities, calibration_labels)
        regularization = l2_strength * (np.sum(log_scales ** 2) + np.sum(biases ** 2))
        return float(nll + regularization)
    initial_parameters = np.zeros(2 * NUM_CLASSES, dtype=np.float64)
    result = minimize(objective, x0=initial_parameters, method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-08})
    if not result.success:
        raise RuntimeError(f'Vector Scaling optimizer failed: {result.message}')
    log_scales = result.x[:NUM_CLASSES]
    raw_biases = result.x[NUM_CLASSES:]
    scales = np.exp(log_scales)
    biases = raw_biases - np.mean(raw_biases)
    return {'scales': scales, 'biases': biases, 'success': bool(result.success)}

def beta_features(negative_probabilities):
    negative_probabilities = np.clip(negative_probabilities, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return np.column_stack([np.log(negative_probabilities), -np.log(1.0 - negative_probabilities)])

def reconstruct_multiclass_from_negative(calibrated_negative_probability, original_probabilities):
    original_probabilities = normalize_probabilities(original_probabilities)
    calibrated_negative_probability = np.clip(np.asarray(calibrated_negative_probability, dtype=np.float64), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    original_positive = original_probabilities[:, POSITIVE_LABEL]
    original_uncertain = original_probabilities[:, UNCERTAIN_LABEL]
    original_non_negative = original_positive + original_uncertain
    conditional_positive = np.divide(original_positive, original_non_negative, out=np.full_like(original_positive, 0.5), where=original_non_negative > PROBABILITY_EPSILON)
    conditional_uncertain = 1.0 - conditional_positive
    new_non_negative_mass = 1.0 - calibrated_negative_probability
    reconstructed = np.column_stack([calibrated_negative_probability, new_non_negative_mass * conditional_positive, new_non_negative_mass * conditional_uncertain])
    return normalize_probabilities(reconstructed)

def fit_beta_negative(calibration_probabilities, calibration_labels):
    binary_targets = (calibration_labels == NEGATIVE_LABEL).astype(np.int64)
    negative_probabilities = calibration_probabilities[:, NEGATIVE_LABEL]
    model = LogisticRegression(penalty='l2', C=BETA_CALIBRATION_C, solver='lbfgs', max_iter=5000, random_state=0)
    model.fit(beta_features(negative_probabilities), binary_targets)
    return model

def apply_beta_negative(model, original_probabilities):
    original_probabilities = normalize_probabilities(original_probabilities)
    original_negative = original_probabilities[:, NEGATIVE_LABEL]
    calibrated_negative = model.predict_proba(beta_features(original_negative))[:, 1]
    return reconstruct_multiclass_from_negative(calibrated_negative, original_probabilities)

def fit_isotonic_negative(calibration_probabilities, calibration_labels):
    binary_targets = (calibration_labels == NEGATIVE_LABEL).astype(np.float64)
    negative_probabilities = calibration_probabilities[:, NEGATIVE_LABEL]
    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip')
    model.fit(negative_probabilities, binary_targets)
    return model

def apply_isotonic_negative(model, original_probabilities):
    original_probabilities = normalize_probabilities(original_probabilities)
    original_negative = original_probabilities[:, NEGATIVE_LABEL]
    calibrated_negative = model.predict(original_negative)
    calibrated_negative = np.clip(calibrated_negative, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return reconstruct_multiclass_from_negative(calibrated_negative, original_probabilities)

def fit_and_apply_all_methods(calibration_logits, calibration_labels, evaluation_logits):
    calibration_uncalibrated = stable_softmax(calibration_logits)
    evaluation_uncalibrated = stable_softmax(evaluation_logits)
    output = {'uncalibrated': evaluation_uncalibrated}
    temperature = fit_temperature_scaling(calibration_logits, calibration_labels)
    output['temperature_scaling'] = stable_softmax(evaluation_logits / temperature)
    dirichlet_model = fit_dirichlet_calibration(calibration_logits, calibration_labels)
    output['dirichlet'] = apply_dirichlet_calibration(dirichlet_model, evaluation_logits)
    vector_parameters = fit_vector_scaling(calibration_logits, calibration_labels)
    output['vector_scaling'] = vector_scaling_transform(evaluation_logits, vector_parameters['scales'], vector_parameters['biases'])
    beta_model = fit_beta_negative(calibration_uncalibrated, calibration_labels)
    output['beta_negative'] = apply_beta_negative(beta_model, evaluation_uncalibrated)
    isotonic_model = fit_isotonic_negative(calibration_uncalibrated, calibration_labels)
    output['isotonic_negative'] = apply_isotonic_negative(isotonic_model, evaluation_uncalibrated)
    return output

def select_maximum_safe_coverage_threshold(probabilities, true_labels):
    probabilities = normalize_probabilities(probabilities)
    true_labels = np.asarray(true_labels, dtype=np.int64)
    predicted_labels = np.argmax(probabilities, axis=1)
    negative_probabilities = probabilities[:, NEGATIVE_LABEL]
    predicted_negative_mask = predicted_labels == NEGATIVE_LABEL
    candidate_values = np.unique(negative_probabilities[predicted_negative_mask])
    candidate_values = np.sort(candidate_values)[::-1]
    no_release_threshold = float(np.nextafter(1.0, np.inf))
    candidate_thresholds = np.concatenate([np.array([no_release_threshold], dtype=np.float64), candidate_values])
    records = []
    total_true_negatives = int(np.sum(true_labels == NEGATIVE_LABEL))
    for threshold in candidate_thresholds:
        release_mask = (predicted_labels == NEGATIVE_LABEL) & (negative_probabilities >= threshold)
        released_labels = true_labels[release_mask]
        number_of_releases = int(np.sum(release_mask))
        unsafe_releases = int(np.sum(released_labels != NEGATIVE_LABEL))
        true_negative_releases = int(np.sum(released_labels == NEGATIVE_LABEL))
        coverage = float(number_of_releases / len(true_labels))
        true_negative_release_rate = float(true_negative_releases / total_true_negatives)
        records.append({'selected_threshold': float(threshold), 'number_of_releases': number_of_releases, 'coverage': coverage, 'unsafe_releases': unsafe_releases, 'true_negative_releases': true_negative_releases, 'true_negative_release_rate': true_negative_release_rate})
    candidates = pd.DataFrame(records)
    safe_candidates = candidates.loc[candidates['unsafe_releases'] == 0].copy()
    selected = safe_candidates.sort_values(by=['coverage', 'selected_threshold'], ascending=[False, False]).iloc[0]
    binary_negative_targets = (true_labels == NEGATIVE_LABEL).astype(np.float64)
    return {'selected_threshold': float(selected['selected_threshold']), 'number_of_releases': int(selected['number_of_releases']), 'coverage': float(selected['coverage']), 'unsafe_releases': int(selected['unsafe_releases']), 'true_negative_release_rate': float(selected['true_negative_release_rate']), 'negative_fixed_ece': fixed_ece(negative_probabilities, binary_negative_targets), 'negative_binary_brier': binary_brier(negative_probabilities, binary_negative_targets), 'negative_binary_nll': binary_nll(negative_probabilities, binary_negative_targets), 'macro_f1': float(f1_score(true_labels, predicted_labels, average='macro', zero_division=0))}


def rank_configurations(seed_metrics):
    """Retrospective lexicographic development objective from cells110-111."""
    keys = ["architecture", "loss", "calibration_method"]
    grouped = seed_metrics.groupby(keys, as_index=False).agg(
        unsafe_releases_total=("unsafe_releases", "sum"),
        number_of_safe_seeds=("unsafe_releases", lambda x: int(np.sum(np.asarray(x) == 0))),
        coverage_mean=("coverage", "mean"), coverage_std=("coverage", "std"),
        negative_fixed_ece_mean=("negative_fixed_ece", "mean"),
        negative_binary_brier_mean=("negative_binary_brier", "mean"),
        negative_binary_nll_mean=("negative_binary_nll", "mean"), macro_f1_mean=("macro_f1", "mean"))
    grouped["safety_priority"] = ((grouped["unsafe_releases_total"] == 0) & (grouped["number_of_safe_seeds"] == 3)).astype(int)
    rank = grouped.sort_values(["safety_priority", "coverage_mean", "negative_fixed_ece_mean",
                                "negative_binary_brier_mean", "negative_binary_nll_mean", "macro_f1_mean", "coverage_std"],
                               ascending=[False, False, True, True, True, False, True]).reset_index(drop=True)
    rank.insert(0, "lexicographic_rank", np.arange(1, len(rank) + 1))
    return rank


def _load_development(manifest_path):
    """Manifest columns: architecture,loss,seed,calibration,threshold_selection."""
    from .statistics import LABELS, _mapped
    manifest_path = Path(manifest_path)
    manifest = pd.read_csv(manifest_path)
    columns = {"architecture", "loss", "seed", "calibration", "threshold_selection"}
    if not columns.issubset(manifest):
        raise ValueError(f"Missing manifest columns: {columns - set(manifest)}")
    if manifest.duplicated(["architecture", "loss", "seed"]).any():
        raise ValueError("Duplicate checkpoint configurations in manifest")
    records = manifest.set_index(["architecture", "loss", "seed"])
    expected = {(a, l, s) for a in ARCHITECTURES for l in LOSSES for s in SEEDS}
    if set(records.index) != expected:
        raise ValueError("Sensitivity analysis requires all 27 architecture/loss/seed checkpoints")
    combined, original_splits = {}, {}
    for key in sorted(expected):
        parts = []
        for split in ("calibration", "threshold_selection"):
            path = Path(str(records.loc[key, split]))
            path = path if path.is_absolute() else manifest_path.parent / path
            frame = pd.read_csv(path, dtype={"sample_id": str}, float_precision="round_trip")
            needed = {"sample_id", "true_label", *LOGIT_COLUMNS}
            if not needed.issubset(frame) or len(frame) != 150:
                raise ValueError(f"Expected 150 rows and required logit columns: {path}")
            frame = frame[list(needed)].copy()
            if frame["sample_id"].isna().any():
                raise ValueError(f"Missing sample identifiers: {path}")
            frame["true_label"] = _mapped(frame["true_label"], LABELS, "true label").astype(int)
            logits = frame[LOGIT_COLUMNS].to_numpy(dtype=float)
            if not np.isfinite(logits).all():
                raise ValueError(f"Nonfinite logits: {path}")
            frame["source_split"] = split
            parts.append(frame)
        both = pd.concat(parts, ignore_index=True)
        if both["sample_id"].duplicated().any():
            raise ValueError(f"Calibration/threshold split overlap or duplicated IDs: {key}")
        combined[key] = both
    master = combined[("resnet18", "cross_entropy", 42)].sort_values("sample_id").reset_index(drop=True)
    ids, labels = master["sample_id"].tolist(), master["true_label"].to_numpy(dtype=int)
    cache = {}
    for key, frame in combined.items():
        if set(frame["sample_id"]) != set(ids):
            raise ValueError(f"Development IDs differ: {key}")
        aligned = frame.set_index("sample_id").loc[ids].reset_index()
        if not np.array_equal(labels, aligned["true_label"]):
            raise ValueError(f"Development reference labels differ: {key}")
        if not np.array_equal(master["source_split"], aligned["source_split"]):
            raise ValueError(f"Original development split memberships differ: {key}")
        cache[key] = aligned[LOGIT_COLUMNS].to_numpy(dtype=float)
    return master, cache


def _evaluate_partition(cache, labels, cal_idx, threshold_idx):
    records = []
    for architecture in ARCHITECTURES:
        for loss in LOSSES:
            for seed in SEEDS:
                logits = cache[(architecture, loss, seed)]
                probabilities = fit_and_apply_all_methods(logits[cal_idx], labels[cal_idx], logits[threshold_idx])
                for method in CALIBRATION_METHODS:
                    record = select_maximum_safe_coverage_threshold(probabilities[method], labels[threshold_idx])
                    records.append({"architecture": architecture, "loss": loss, "seed": seed,
                                    "calibration_method": method, **record})
    return pd.DataFrame(records)


def run_sensitivity(manifest_path, output, n_repartitions=100, seed=2026):
    """Reproduce original development ranking and repeated calibration/threshold fits.

    The same 300 sorted development images and same 100 stratified splits are used
    for the factorial configurations and the mean-logit ensemble. These overlapping
    repartitions describe development sensitivity, not independent validation.
    """
    master, cache = _load_development(manifest_path)
    y = master["true_label"].to_numpy(dtype=int)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cal = np.flatnonzero(master["source_split"].eq("calibration"))
    threshold = np.flatnonzero(master["source_split"].eq("threshold_selection"))
    original = _evaluate_partition(cache, y, cal, threshold)
    original.to_csv(output / "original_development_seed_level.csv", index=False)
    rank_configurations(original).to_csv(output / "original_development_ranking.csv", index=False)
    ensemble = np.mean(np.stack([cache[("resnet18", "focal_loss", s)] for s in SEEDS]), axis=0)
    splitter = StratifiedShuffleSplit(n_splits=n_repartitions, test_size=150, random_state=seed)
    seed_rows, rank_rows, ensemble_rows, partition_rows = [], [], [], []
    for repeat, (cal, threshold) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        evaluated = _evaluate_partition(cache, y, cal, threshold)
        evaluated.insert(0, "repartition", repeat)
        seed_rows.append(evaluated)
        ranked = rank_configurations(evaluated)
        ranked.insert(0, "repartition", repeat)
        rank_rows.append(ranked)
        fitted = fit_vector_scaling(ensemble[cal], y[cal])
        probabilities = vector_scaling_transform(ensemble[threshold], fitted["scales"], fitted["biases"])
        metrics = select_maximum_safe_coverage_threshold(probabilities, y[threshold])
        ensemble_rows.append({"repartition": repeat, **metrics,
                              **{f"scale_{label}": fitted["scales"][c] for c, label in enumerate(("negative", "positive", "uncertain"))},
                              **{f"bias_{label}": fitted["biases"][c] for c, label in enumerate(("negative", "positive", "uncertain"))}})
        for subset, indices in (("calibration", cal), ("threshold_selection", threshold)):
            partition_rows.extend({"repartition": repeat, "sample_id": master.iloc[i]["sample_id"],
                                   "true_label": int(y[i]), "assigned_subset": subset} for i in indices)
        if repeat == 1 or repeat % 10 == 0 or repeat == n_repartitions:
            print(f"Completed development repartition {repeat}/{n_repartitions}")
    all_seeds, all_ranks, ensembles = pd.concat(seed_rows, ignore_index=True), pd.concat(rank_rows, ignore_index=True), pd.DataFrame(ensemble_rows)
    selection = all_ranks[(all_ranks["architecture"] == "resnet18") & (all_ranks["loss"] == "focal_loss") & (all_ranks["calibration_method"] == "vector_scaling")]
    summary = {"repartitions": n_repartitions, "selection_rank_1_count": int((selection["lexicographic_rank"] == 1).sum()),
               "selection_top_3_count": int((selection["lexicographic_rank"] <= 3).sum()),
               "selection_rank_median": float(selection["lexicographic_rank"].median())}
    for col, prefix in (("selected_threshold", "threshold"), ("coverage", "coverage")):
        summary.update({f"ensemble_{prefix}_{name}": float(getattr(ensembles[col], func)(*args))
                        for name, func, args in [("median", "median", ()), ("q25", "quantile", (.25,)),
                                                 ("q75", "quantile", (.75,)), ("min", "min", ()), ("max", "max", ())]})
    for name, frame in {"sensitivity_seed_level": all_seeds, "sensitivity_rankings": all_ranks,
                        "ensemble_threshold_sensitivity_100_repartitions": ensembles,
                        "development_repartition_assignments": pd.DataFrame(partition_rows),
                        "sensitivity_summary": pd.DataFrame([summary])}.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    metadata = {"source_cells_zero_based": [110, 111, 112], "repartitions": n_repartitions, "seed": seed,
                "source_splits": ["calibration", "threshold_selection"], "evaluation_used": False,
                "checkpoint_retraining": False, "with_replacement": False,
                "class_counts": {str(c): int(np.sum(y == c)) for c in (0, 1, 2)},
                "historical_numerical_epsilon": PROBABILITY_EPSILON,
                "limitations": "Overlapping image-level repartitions are not independent cohorts"}
    (output / "sensitivity_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return summary
