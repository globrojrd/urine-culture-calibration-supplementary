"""Supplementary reliability, seed-policy and confusion outputs from measured inputs.

Reliability estimator and seed are from cell100; two-panel layout follows the
supplied Supplementary_Material.pdf. This regenerates assets, not that PDF.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from urineculture.statistics import read_predictions, validate_study_cohort, operating_summary, confusion_tables


def reliability(frame, output):
    import matplotlib.pyplot as plt
    y = frame.true_label.to_numpy() == 0
    p = frame.prob_negative.to_numpy()
    edges = np.linspace(0., 1., 16)
    bins = np.digitize(p, edges[1:-1], right=False)
    neg, non = np.flatnonzero(y), np.flatnonzero(~y)
    rng = np.random.default_rng(20260729)
    values = np.full((2000, 15), np.nan)
    for b in range(2000):
        indices = np.concatenate([rng.choice(neg, len(neg), replace=True), rng.choice(non, len(non), replace=True)])
        rng.shuffle(indices)
        for k in range(15):
            members = indices[bins[indices] == k]
            if len(members):
                values[b, k] = y[members].mean()
    rows = []
    for k in range(15):
        mask = bins == k
        finite = values[:, k][np.isfinite(values[:, k])]
        rows.append({"bin": k + 1, "lower_edge": edges[k], "upper_edge": edges[k+1],
                     "count": int(mask.sum()), "mean_probability": p[mask].mean() if mask.any() else np.nan,
                     "observed_frequency": y[mask].mean() if mask.any() else np.nan,
                     "ci_lower_95": np.quantile(finite, .025) if len(finite) else np.nan,
                     "ci_upper_95": np.quantile(finite, .975) if len(finite) else np.nan})
    bins_table = pd.DataFrame(rows)
    bins_table.to_csv(output / "S1_reliability_bins.csv", index=False)
    shown = bins_table[bins_table["count"] > 0]
    fig, ax = plt.subplots(2, 1, figsize=(6.2, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot([0, 1], [0, 1], "--", label="Perfect calibration")
    errors = np.clip(np.vstack([shown.observed_frequency-shown.ci_lower_95,
                               shown.ci_upper_95-shown.observed_frequency]), 0, None)
    ax[0].errorbar(shown.mean_probability, shown.observed_frequency, yerr=errors, fmt="o-", capsize=3,
                   label="Operational ensemble")
    ax[0].set(ylabel="Observed proportion of Negative", ylim=(-.02, 1.02))
    ax[0].legend(frameon=False)
    ax[1].bar((edges[:-1]+edges[1:])/2, bins_table["count"], width=1/15*.9)
    for _, row in shown.iterrows():
        ax[1].text((row.lower_edge+row.upper_edge)/2, row["count"], str(int(row["count"])), ha="center", va="bottom", fontsize=8)
    ax[1].set(xlabel="Predicted probability of Negative", ylabel="Bin count", xlim=(0, 1))
    fig.tight_layout()
    fig.savefig(output / "S1_reliability.pdf", bbox_inches="tight")
    fig.savefig(output / "S1_reliability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Seven frozen-policy CSVs")
    parser.add_argument("--factorial", type=Path, help="Optional completed factorial output directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty supplementary-output directory")
    ensemble = read_predictions(args.inputs / "ensemble.csv")
    validate_study_cohort(ensemble)
    seeds = {str(s): read_predictions(args.inputs / f"ResNet_individual_seed_{s}.csv") for s in (42, 123, 2026)}
    args.output.mkdir(parents=True, exist_ok=True)
    operating_summary(seeds).to_csv(args.output / "S2_seed_policies.csv", index=False)
    multiclass, operational = confusion_tables(ensemble)
    multiclass.to_csv(args.output / "S3a_confusion.csv", index=False)
    operational.to_csv(args.output / "S3b_release_referral.csv", index=False)
    reliability(ensemble, args.output)
    if args.factorial:
        import matplotlib.pyplot as plt
        dev = pd.read_csv(args.factorial / "development_by_seed.csv")
        s4 = dev.groupby("calibration_method").agg(
            releases=("number_of_releases", "sum"), mean_coverage=("coverage", "mean"),
            zero_release_runs=("number_of_releases", lambda x: int((x == 0).sum())),
            seed_evaluations=("seed", "size"))
        if not (s4.seed_evaluations == 27).all():
            raise ValueError("S4 requires all 27 seed evaluations per condition")
        s4.to_csv(args.output / "S4_release_counts.csv")
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        for seed in (42, 123, 2026):
            curve = pd.read_csv(args.factorial / f"resnet18/focal_loss/seed_{seed}/vector_scaling/threshold_candidates.csv")
            # Conditional risk is undefined with no releases; only the plotting symbol is at zero.
            active = curve[curve.number_of_releases > 0].sort_values("coverage")
            ax.plot(active.coverage * 100, active.unsafe_rate_among_releases * 100, label=f"Seed {seed}")
        ax.scatter([0], [0], marker="s", facecolors="none", edgecolors="black", label="No release (risk undefined)")
        ax.set(xlabel="Coverage (%)", ylabel="Observed unsafe releases / releases (%)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.output / "S2_risk_coverage.pdf", bbox_inches="tight")
        plt.close(fig)
    print("Supplementary assets saved in", args.output)


if __name__ == "__main__":
    main()
