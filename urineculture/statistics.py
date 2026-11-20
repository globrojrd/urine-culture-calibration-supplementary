"""Paired image-level analyses from notebook cells 99, 118, 122 and 123 (zero-based).

Policies are already fitted and frozen. These intervals do not cover model selection,
training variability, or unknown within-patient dependence. Three training seeds are
not independent evaluation cohorts. Negative=0, Positive=1, Uncertain=2.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {"0": 0, "0.0": 0, "negative": 0, "1": 1, "1.0": 1,
          "positive": 1, "2": 2, "2.0": 2, "uncertain": 2}
BOOLEAN = {"true": True, "1": True, "1.0": True, "yes": True,
           "false": False, "0": False, "0.0": False, "no": False}
COMPARATOR_NAMES = tuple(name for seed in (42, 123, 2026) for name in
                         (f"ResNet_individual_seed_{seed}", f"Swin_uncalibrated_seed_{seed}"))


def _mapped(series, mapping, kind):
    result = series.astype(str).str.strip().str.lower().map(mapping)
    if result.isna().any():
        raise ValueError(f"Unknown or missing {kind}: {series[result.isna()].tolist()[:5]}")
    return result


def normalize_predictions(raw, pred_col="predicted_label", prob_col="prob_negative",
                          release_col="released", threshold_col="operating_threshold",
                          threshold=None):
    """Validate a frozen policy without changing its row order or refitting it."""
    needed = {"sample_id", "true_label", pred_col, prob_col}
    if release_col is not None:
        needed.add(release_col)
    if not needed.issubset(raw):
        raise ValueError(f"Missing prediction columns: {sorted(needed - set(raw))}")
    if raw["sample_id"].isna().any():
        raise ValueError("Missing sample identifiers")
    result = pd.DataFrame({
        "sample_id": raw["sample_id"].astype(str),
        "true_label": _mapped(raw["true_label"], LABELS, "true label").astype(int),
        "predicted_label": _mapped(raw[pred_col], LABELS, "prediction").astype(int),
        "prob_negative": pd.to_numeric(raw[prob_col], errors="raise"),
    })
    if result.empty or result["sample_id"].duplicated().any():
        raise ValueError("Predictions must be nonempty with one row per unique sample_id")
    probabilities = result["prob_negative"].to_numpy()
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Negative probabilities must be finite and in [0,1]")
    if threshold is None and threshold_col in raw:
        values = pd.to_numeric(raw[threshold_col], errors="raise").unique()
        if len(values) != 1:
            raise ValueError("A frozen policy must contain exactly one threshold")
        threshold = float(values[0])
    if threshold is None or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("A finite nonnegative frozen threshold is required")
    expected = (result["predicted_label"] == 0) & (result["prob_negative"] >= threshold)
    if release_col is None:
        result["released"] = expected
    else:
        result["released"] = _mapped(raw[release_col], BOOLEAN, "release flag").astype(bool)
        if not np.array_equal(expected, result["released"]):
            raise ValueError("Release flags do not match predicted Negative AND probability >= threshold")
    result["operating_threshold"] = float(threshold)
    if "unsafe_release" in raw:
        unsafe = _mapped(raw["unsafe_release"], BOOLEAN, "unsafe flag").astype(bool)
        if not np.array_equal(unsafe, result["released"] & (result["true_label"] != 0)):
            raise ValueError("Stored unsafe_release flags disagree with the policy definition")
    return result.reset_index(drop=True)


def read_predictions(path):
    return normalize_predictions(pd.read_csv(Path(path), dtype={"sample_id": str}, float_precision="round_trip"))


def validate_study_cohort(frame):
    counts = frame["true_label"].value_counts().sort_index().to_dict()
    if len(frame) != 150 or counts != {0: 50, 1: 50, 2: 50}:
        raise ValueError(f"Article evaluation requires 150 images, 50 per class; obtained {counts}")


def align_comparator(base, comp):
    if set(base["sample_id"]) != set(comp["sample_id"]):
        raise ValueError("Policies were not evaluated on exactly the same sample IDs")
    aligned = comp.set_index("sample_id").loc[base["sample_id"]].reset_index()
    if not np.array_equal(base["true_label"], aligned["true_label"]):
        raise ValueError("Reference labels disagree after pairing by sample_id")
    return aligned


def exact_mcnemar(a, b):
    """Two-sided exact binomial test on paired binary discordances."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Paired binary endpoints must be equal-length vectors")
    a_only, b_only = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = a_only + b_only
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / 2**n) if n else 1.0
    return a_only, b_only, float(p)


def holm_adjust(p_values):
    """Holm adjustment for one explicitly supplied family of tests."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p-values")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return adjusted


def _metrics(data, indices):
    y = data["true_label"].to_numpy(dtype=int)[indices]
    pred = data["predicted_label"].to_numpy(dtype=int)[indices]
    release = data["released"].to_numpy(dtype=bool)[indices]
    prob = data["prob_negative"].to_numpy(dtype=float)[indices]
    f1 = []
    for cls in (0, 1, 2):
        tp = np.sum((y == cls) & (pred == cls), axis=1)
        denom = np.sum(y == cls, axis=1) + np.sum(pred == cls, axis=1)
        f1.append(np.divide(2 * tp, denom, out=np.zeros(len(y)), where=denom > 0))
    return {"Accuracy": np.mean(y == pred, axis=1), "Macro-F1": np.mean(f1, axis=0),
            "Coverage": release.mean(axis=1),
            "TNRR": np.sum(release & (y == 0), axis=1) / np.sum(y == 0, axis=1),
            "Negative Brier": np.mean(((y == 0).astype(float) - prob)**2, axis=1)}


def paired_comparisons(base, comparators, n_boot=3000, seed=20261112):
    """Six paired comparisons with shared three-class-stratified bootstrap draws.

    Exact tests are adjusted separately across six Accuracy tests and six TNRR
    tests. Percentile difference intervals are marginal, not multiplicity-adjusted.
    The vectorized RNG draw ordering reproduces notebook cell 123 exactly.
    """
    validate_study_cohort(base)
    if tuple(comparators) != COMPARATOR_NAMES:
        raise ValueError(f"Supply these six comparators in order: {COMPARATOR_NAMES}")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    y = base["true_label"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.choice(ix, size=(n_boot, len(ix)), replace=True)
                              for ix in [np.flatnonzero(y == c) for c in (0, 1, 2)]], axis=1)
    point_idx = np.arange(len(y))[None, :]
    point_a, boot_a = _metrics(base, point_idx), _metrics(base, indices)
    metrics, tests, iterations = [], [], []
    for name, original in comparators.items():
        comp = align_comparator(base, original)
        validate_study_cohort(comp)
        point_b, boot_b = _metrics(comp, point_idx), _metrics(comp, indices)
        a_only, b_only, p_acc = exact_mcnemar(base["predicted_label"].to_numpy() == y,
                                             comp["predicted_label"].to_numpy() == y)
        negative = y == 0
        ra, rb, p_tnrr = exact_mcnemar(base["released"].to_numpy()[negative],
                                      comp["released"].to_numpy()[negative])
        tests.append({"Comparison": name, "Accuracy_A_only_correct": a_only,
                      "Accuracy_B_only_correct": b_only, "McNemar_p_accuracy": p_acc,
                      "Negative_released_only_A": ra, "Negative_released_only_B": rb,
                      "McNemar_p_TNRR": p_tnrr,
                      "Unsafe_ensemble": int(np.sum(base["released"] & (y != 0))),
                      "Unsafe_comparator": int(np.sum(comp["released"] & (y != 0)))})
        for metric in point_a:
            delta = boot_a[metric] - boot_b[metric]
            lo, hi = np.quantile(delta, [0.025, 0.975])
            metrics.append({"Comparison": name, "Metric": metric,
                            "Ensemble": point_a[metric][0], "Comparator": point_b[metric][0],
                            "Delta_A_minus_B": point_a[metric][0] - point_b[metric][0],
                            "CI_lower_95": lo, "CI_upper_95": hi,
                            "CI_excludes_zero": bool(lo > 0 or hi < 0)})
            iterations.append(pd.DataFrame({"Comparison": name, "Metric": metric,
                                            "Iteration": np.arange(1, n_boot + 1),
                                            "Delta_A_minus_B": delta}))
    tests = pd.DataFrame(tests)
    for col in ("McNemar_p_accuracy", "McNemar_p_TNRR"):
        tests[col + "_Holm"] = holm_adjust(tests[col].to_numpy())
    return pd.DataFrame(metrics), tests, pd.concat(iterations, ignore_index=True)


def negative_ece(y_negative, probability, bins=15):
    """Cell 99 bins: [lower,upper), final bin includes 1; empty bins contribute zero."""
    bin_id = np.digitize(probability, np.linspace(0, 1, bins + 1)[1:-1], right=False)
    return float(sum(np.mean(bin_id == b) * abs(np.mean(probability[bin_id == b]) -
                    np.mean(y_negative[bin_id == b])) for b in range(bins) if np.any(bin_id == b)))


def _single_metrics(y, p, release):
    safe = int(np.sum(release & y))
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return {"coverage": float(release.mean()), "negative_ece": negative_ece(y, p),
            "negative_brier": float(np.mean((p - y)**2)),
            "negative_nll": float(-np.mean(y * np.log(clipped) + (1-y) * np.log(1-clipped))),
            "npv": safe / int(release.sum()) if release.any() else np.nan,
            "tnrr": safe / int(y.sum()) if y.any() else np.nan,
            "released": int(release.sum()), "unsafe": int(np.sum(release & ~y))}


def single_policy_bootstrap(base, n_boot=2000, seed=20260730):
    """Historical Negative/non-Negative-stratified bootstrap, distinct from paired analysis.

    A degenerate NPV interval is retained to reproduce the notebook output; it is
    not evidence of zero population risk. Use the exact unsafe-release bound.
    Each iteration includes the original RNG shuffle even though metrics ignore
    order: omitting it changes subsequent draws and the published percentiles.
    """
    validate_study_cohort(base)
    y = base["true_label"].to_numpy(dtype=int) == 0
    p = base["prob_negative"].to_numpy(dtype=float)
    release = base["released"].to_numpy(dtype=bool)
    point = _single_metrics(y, p, release)
    neg, nonneg = np.flatnonzero(y), np.flatnonzero(~y)
    rng, records = np.random.default_rng(seed), []
    for iteration in range(1, n_boot + 1):
        ix = np.concatenate([rng.choice(neg, len(neg), replace=True),
                             rng.choice(nonneg, len(nonneg), replace=True)])
        rng.shuffle(ix)
        records.append({**_single_metrics(y[ix], p[ix], release[ix]), "bootstrap_iteration": iteration})
    boot = pd.DataFrame(records)
    summary = []
    for metric in ("coverage", "negative_ece", "negative_brier", "negative_nll", "npv", "tnrr"):
        values = boot[metric].dropna()
        lo, hi = np.percentile(values, [2.5, 97.5]) if len(values) else (np.nan, np.nan)
        summary.append({"metric": metric, "estimate": point[metric], "ci_lower_95": lo,
                        "ci_upper_95": hi, "bootstrap_iterations": n_boot, "bootstrap_seed": seed,
                        "bootstrap_method": "Stratified percentile bootstrap preserving Negative/non-Negative counts",
                        "interpretation": "Degenerate NPV bootstrap is not a safety guarantee" if metric == "npv" else "Conditional image-level interval"})
    return pd.DataFrame(summary), boot


def exact_upper_bound(events, denominator, confidence=0.95):
    """One-sided Clopper-Pearson upper bound, conditional on binomial independence."""
    if denominator < 0 or events < 0 or events > denominator or not 0 < confidence < 1:
        raise ValueError("Invalid binomial counts or confidence level")
    if denominator == 0:
        return np.nan
    if events == denominator:
        return 1.0
    if events == 0:
        return float(1 - (1-confidence)**(1/denominator))
    from scipy.stats import beta
    return float(beta.ppf(confidence, events + 1, denominator - events))


def false_negative_analysis(base):
    y = base["true_label"].to_numpy(dtype=int)
    pred = base["predicted_label"].to_numpy(dtype=int)
    release = base["released"].to_numpy(dtype=bool)
    pos, nonneg = y == 1, y != 0
    unsafe = int(np.sum(release & nonneg))
    counts = [
        ("Pre-abstention Positive FNR", int(np.sum(pos & (pred != 1))), int(pos.sum())),
        ("Operational Positive FNR", int(np.sum(pos & release)), int(pos.sum())),
        ("Non-Negative unsafe release rate", unsafe, int(nonneg.sum())),
        ("Unsafe rate among releases", unsafe, int(release.sum())),
        ("Overall unsafe release rate", unsafe, len(y)),
        ("Positive classified as Negative", int(np.sum(pos & (pred == 0))), int(pos.sum())),
        ("Positive classified as Uncertain", int(np.sum(pos & (pred == 2))), int(pos.sum())),
    ]
    return pd.DataFrame([{"Metric": name, "Events": k, "Denominator": n,
                          "Observed rate": k/n if n else np.nan,
                          "One-sided 95% upper bound": exact_upper_bound(k, n)} for name, k, n in counts])


def confusion_tables(base):
    """Multiclass predictions before abstention and operational release/referral."""
    labels = ("Negative", "Positive", "Uncertain")
    y = base["true_label"].to_numpy(dtype=int)
    pred = base["predicted_label"].to_numpy(dtype=int)
    release = base["released"].to_numpy(dtype=bool)
    multiclass = pd.DataFrame([[int(np.sum((y == a) & (pred == b))) for b in range(3)]
                               for a in range(3)], columns=[f"Predicted {v}" for v in labels])
    multiclass.insert(0, "Reference", labels)
    operational = pd.DataFrame([{"Reference": name, "Released": int(np.sum(mask & release)),
                                 "Referred": int(np.sum(mask & ~release))}
                                for name, mask in (("Negative", y == 0), ("Non-Negative", y != 0))])
    return multiclass, operational


def operating_summary(policies):
    rows = []
    for name, frame in policies.items():
        y = frame["true_label"].to_numpy(dtype=int)
        released = frame["released"].to_numpy(dtype=bool)
        n, k = int(released.sum()), int(np.sum(released & (y != 0)))
        rows.append({"Policy": name, "Threshold": float(frame["operating_threshold"].iloc[0]),
                     "Released": n, "Coverage": n/len(y), "Unsafe": k,
                     "Observed NPV": (n-k)/n if n else np.nan,
                     "TNRR": (n-k)/int(np.sum(y == 0)),
                     "Unsafe among releases one-sided 95% upper": exact_upper_bound(k, n)})
    return pd.DataFrame(rows)
