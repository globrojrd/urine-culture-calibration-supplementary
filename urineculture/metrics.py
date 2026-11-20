"""Classification and Negative-event calibration metrics; notebook cells 81–88.

Fixed ECE bins are [0, 1/15], (1/15, 2/15], ... . Adaptive bins use
numpy.argsort followed by numpy.array_split, as in the original notebook.
"""
import numpy as np


def validate_predictions(labels, probabilities):
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0:
        raise ValueError("Expected nonempty (n, 3) probabilities.")
    if y.shape != (len(p),) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Labels must be integers 0, 1, or 2 and match probabilities.")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    if not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-8):
        raise ValueError("Each probability row must sum to one.")
    return y.astype(np.int64), p


def binary_nll(probabilities, targets):
    p = np.clip(np.asarray(probabilities), 1e-12, 1 - 1e-12)
    y = np.asarray(targets)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def binary_brier(probabilities, targets):
    return float(np.mean((np.asarray(probabilities) - np.asarray(targets)) ** 2))


def calibration_bins(probabilities, targets, number_of_bins=15, adaptive=False):
    p, y = np.asarray(probabilities), np.asarray(targets)
    if p.ndim != 1 or y.shape != p.shape or len(p) == 0:
        raise ValueError("Expected matching nonempty probability and target vectors.")
    if not isinstance(number_of_bins, int) or number_of_bins < 1:
        raise ValueError("number_of_bins must be a positive integer.")
    if adaptive:
        groups = np.array_split(np.argsort(p), number_of_bins)
        edges = None
    else:
        edges = np.linspace(0, 1, number_of_bins + 1)
        groups = [np.flatnonzero((p >= edges[i] if i == 0 else p > edges[i])
                                & (p <= edges[i + 1])) for i in range(number_of_bins)]
    records = []
    for i, group in enumerate(groups):
        count = len(group)
        if adaptive and count == 0:
            continue
        confidence = float(p[group].mean()) if count else float("nan")
        frequency = float(y[group].mean()) if count else float("nan")
        records.append({"bin_index": i + 1, "count": count, "weight": count / len(p),
                        "lower_bound": float(p[group].min()) if adaptive else float(edges[i]),
                        "upper_bound": float(p[group].max()) if adaptive else float(edges[i + 1]),
                        "mean_probability": confidence, "observed_frequency": frequency,
                        "absolute_gap": abs(confidence - frequency)})
    ece = sum(row["weight"] * row["absolute_gap"] for row in records if row["count"])
    mce = max(row["absolute_gap"] for row in records if row["count"])
    return float(ece), float(mce), records


def negative_auroc(probabilities, binary_targets):
    """Pairwise AUC with half credit for ties; equivalent to rank-based AUC."""
    p, y = np.asarray(probabilities), np.asarray(binary_targets)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum())
                 / (len(pos) * len(neg)))


def evaluate_predictions(labels, probabilities):
    y, p = validate_predictions(labels, probabilities)
    pred = p.argmax(axis=1)
    confusion = np.zeros((3, 3), dtype=int)
    np.add.at(confusion, (y, pred), 1)
    denominators = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * np.diag(confusion), denominators,
                   out=np.zeros(3, dtype=float), where=denominators != 0)
    target = (y == 0).astype(float)
    fixed, fixed_mce, _ = calibration_bins(p[:, 0], target)
    adaptive, adaptive_mce, _ = calibration_bins(p[:, 0], target, adaptive=True)
    top_ece, _, _ = calibration_bins(p.max(axis=1), (pred == y).astype(float))
    clipped = np.clip(p, 1e-12, 1)
    clipped /= clipped.sum(axis=1, keepdims=True)
    result = {"number_of_samples": len(y), "accuracy": float((pred == y).mean()),
              "macro_f1": float(f1.mean()), "negative_fixed_ece": fixed,
              "negative_adaptive_ece": adaptive, "negative_fixed_mce": fixed_mce,
              "negative_adaptive_mce": adaptive_mce,
              "negative_binary_brier": binary_brier(p[:, 0], target),
              "negative_binary_nll": binary_nll(p[:, 0], target),
              "negative_auroc": negative_auroc(p[:, 0], target),
              "multiclass_nll": float(-np.log(clipped[np.arange(len(y)), y]).mean()),
              "multiclass_brier": float(((p - np.eye(3)[y]) ** 2).sum(axis=1).mean()),
              "top_label_ece": top_ece,
              "positive_misclassified_count": int(((y == 1) & (pred != 1)).sum()),
              "positive_to_negative_count": int(confusion[1, 0]),
              "positive_to_uncertain_count": int(confusion[1, 2])}
    result.update({f"confusion_{i}{j}": int(confusion[i, j]) for i in range(3) for j in range(3)})
    return result


def descriptive_calibration_slope(probabilities, targets):
    """Cell 82 estimate with cell 83 correction; never used to recalibrate.

    AUROC >= .999 is the notebook's diagnostic flag, not a mathematical
    definition of quasi-separation. Unstable estimates remain missing.
    """
    from scipy.optimize import minimize
    from scipy.special import expit
    p, y = np.asarray(probabilities), np.asarray(targets)
    if set(np.unique(y)) != {0, 1}:
        return {"calibration_intercept": None, "calibration_slope": None,
                "calibration_slope_status": "not_estimable_missing_class"}
    positive, negative = p[y == 1], p[y == 0]
    perfect = positive.min() > negative.max() or negative.min() > positive.max()
    if perfect or negative_auroc(p, y) >= .999:
        status = "perfect_separation" if perfect else "near_separation_auroc_flag"
        return {"calibration_intercept": None, "calibration_slope": None,
                "calibration_slope_status": "not_estimable_due_to_" + status}
    p = np.clip(p, 1e-8, 1 - 1e-8)
    x = np.log(p / (1 - p))
    result = minimize(lambda theta: binary_nll(expit(theta[0] + theta[1] * x), y),
                      [0., 1.], method="BFGS", options={"maxiter": 2000, "gtol": 1e-8})
    intercept, slope = result.x
    valid = result.success and np.isfinite(result.x).all() and abs(intercept) <= 10 and abs(slope) <= 10
    return {"calibration_intercept": float(intercept) if valid else None,
            "calibration_slope": float(slope) if valid else None,
            "calibration_slope_status": "estimable" if valid else "not_estimable_unstable_fit"}
