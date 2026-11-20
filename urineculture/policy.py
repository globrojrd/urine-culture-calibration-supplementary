"""Frozen selective Negative-release policy; notebook cells 80 and 86.

An image is released only when Negative is its calibrated argmax and
P(Negative) >= threshold. Zero observed unsafe releases does not imply zero
population risk. Non-released images are retained for manual review.
"""
import numpy as np
from .metrics import validate_predictions

NO_RELEASE_THRESHOLD = float(np.nextafter(1.0, np.inf))


def release_mask(probabilities, threshold):
    p = np.asarray(probabilities, dtype=float)
    validate_predictions(np.zeros(len(p), dtype=int), p)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative.")
    return (p.argmax(axis=1) == 0) & (p[:, 0] >= threshold)


def exact_upper_bound(events, trials, confidence=.95):
    if trials == 0:
        return float("nan")
    if not 0 <= events <= trials or not 0 < confidence < 1:
        raise ValueError("Invalid binomial count or confidence level.")
    if events == 0:
        return float(1 - (1 - confidence) ** (1 / trials))
    if events == trials:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(confidence, events + 1, trials - events))


def evaluate_policy(labels, probabilities, threshold):
    y, p = validate_predictions(labels, probabilities)
    released = release_mask(p, threshold)
    n = len(y)
    count = int(released.sum())
    correct = int((released & (y == 0)).sum())
    missed_positive = int((released & (y == 1)).sum())
    missed_uncertain = int((released & (y == 2)).sum())
    unsafe = missed_positive + missed_uncertain
    n_neg, n_pos, n_unc = [int((y == i).sum()) for i in range(3)]
    def ratio(a, b):
        return a / b if b else float("nan")
    return {"selected_threshold": float(threshold), "number_of_samples": n,
            "number_of_releases": count, "coverage": count / n,
            "manual_review_rate": 1 - count / n, "true_negative_releases": correct,
            "unsafe_releases": unsafe, "unsafe_positive_releases": missed_positive,
            "unsafe_uncertain_releases": missed_uncertain,
            "true_negative_release_rate": ratio(correct, n_neg),
            "negative_predictive_value": ratio(correct, count),
            "unsafe_rate_among_releases": ratio(unsafe, count),
            "unsafe_rate_upper_95_one_sided": exact_upper_bound(unsafe, count),
            "operational_positive_fnr": ratio(missed_positive, n_pos),
            "operational_positive_fnr_upper_95": exact_upper_bound(missed_positive, n_pos),
            "operational_nonnegative_fnr": ratio(unsafe, n_pos + n_unc),
            "operational_nonnegative_fnr_upper_95": exact_upper_bound(unsafe, n_pos + n_unc),
            "total_true_negatives": n_neg, "total_true_positives": n_pos,
            "total_true_uncertain": n_unc, "zero_observed_unsafe_releases": unsafe == 0}


def select_threshold(labels, probabilities):
    """Fit on threshold-selection images only. Returns selected row and audit."""
    y, p = validate_predictions(labels, probabilities)
    values = np.unique(p[p.argmax(axis=1) == 0, 0])[::-1]
    thresholds = np.concatenate(([NO_RELEASE_THRESHOLD], values))
    candidates = [evaluate_policy(y, p, t) for t in thresholds]
    feasible = [row for row in candidates if row["unsafe_releases"] == 0]
    selected = max(feasible, key=lambda row: (row["number_of_releases"], row["selected_threshold"]))
    selected = dict(selected)
    selected["selection_status"] = ("zero_observed_unsafe_threshold_selected" if selected["number_of_releases"]
                                     else "no_release_selected")
    return selected, candidates


def risk_coverage_curve(labels, probabilities):
    """All distinct attainable thresholds including the no-release policy."""
    y, p = validate_predictions(labels, probabilities)
    values = np.unique(p[p.argmax(axis=1) == 0, 0])[::-1]
    return [evaluate_policy(y, p, t) for t in np.concatenate(([NO_RELEASE_THRESHOLD], values))]
