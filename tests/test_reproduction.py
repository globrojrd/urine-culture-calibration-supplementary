"""Protocol checks on synthetic fixtures, never substituted for study results."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from urineculture.ensemble import fit_ensemble, evaluate_ensemble, mean_aligned_logits
from urineculture.factorial import LOGIT_COLUMNS
from urineculture.policy import release_mask, select_threshold, exact_upper_bound
from urineculture.statistics import exact_mcnemar, holm_adjust, paired_comparisons, COMPARATOR_NAMES


class ProtocolTests(unittest.TestCase):
    def test_paired_bootstrap_keeps_identical_policies_identical_after_alignment(self):
        y = np.repeat([0, 1, 2], 50)
        frame = pd.DataFrame({"sample_id": [f"synthetic-{i}" for i in range(150)],
                              "true_label": y, "predicted_label": y,
                              "prob_negative": np.where(y == 0, .95, .02),
                              "released": y == 0, "operating_threshold": .9})
        comparators = {name: frame.iloc[::-1].copy() for name in COMPARATOR_NAMES}
        metrics, tests, samples = paired_comparisons(frame, comparators)
        self.assertEqual(len(samples), 3000 * 6 * 5)
        np.testing.assert_array_equal(samples.Delta_A_minus_B, 0)
        np.testing.assert_array_equal(metrics.CI_lower_95, 0)
        np.testing.assert_array_equal(metrics.CI_upper_95, 0)
        np.testing.assert_array_equal(tests.McNemar_p_accuracy_Holm, 1)
        np.testing.assert_array_equal(tests.McNemar_p_TNRR_Holm, 1)

    def test_reported_exact_bounds_and_holm(self):
        # Aggregate discordances were reported in the supplied notebook output.
        accuracy = [(4, 3), (3, 9), (3, 0), (8, 7), (4, 0), (1, 7)]
        tnrr = [(3, 0), (3, 5), (1, 3), (5, 4), (0, 6), (3, 5)]
        def pvalues(pairs):
            result = []
            for a, b in pairs:
                left = np.array([True] * a + [False] * b)
                right = ~left
                result.append(exact_mcnemar(left, right)[2])
            return result
        np.testing.assert_allclose(holm_adjust(pvalues(accuracy)), [1, .625, .75, 1, .625, .421875])
        np.testing.assert_allclose(holm_adjust(pvalues(tnrr)), [1, 1, 1, 1, .1875, 1])
        for n, expected in [(50, .058155079), (100, .029513050), (39, .073937588)]:
            self.assertAlmostEqual(exact_upper_bound(0, n), expected, places=7)
        self.assertTrue(np.isnan(exact_upper_bound(0, 0)))

    def test_policy_argmax_constraint_inclusive_threshold_and_no_release(self):
        p = np.array([[.8, .1, .1], [.7, .2, .1], [.4, .5, .1], [.6, .3, .1]])
        y = np.array([0, 0, 0, 1])
        np.testing.assert_array_equal(release_mask(p, .4), [True, True, False, True])
        selected, _ = select_threshold(y, p)
        self.assertEqual(selected["selected_threshold"], .7)
        self.assertEqual(selected["number_of_releases"], 2)
        unsafe, _ = select_threshold(np.ones(4, dtype=int), p)
        self.assertEqual(unsafe["number_of_releases"], 0)
        self.assertTrue(np.isnan(unsafe["negative_predictive_value"]))

    def test_ensemble_alignment_freezing_and_leakage_checks(self):
        rng = np.random.default_rng(904)
        y = np.repeat([0, 1, 2], 50)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for seed in (42, 123, 2026):
                record = {"architecture": "resnet18", "loss": "focal_loss", "seed": seed}
                for split in ("calibration", "threshold_selection", "final_test"):
                    z = rng.normal(size=(150, 3)) + np.eye(3)[y] * 2
                    frame = pd.DataFrame(z, columns=LOGIT_COLUMNS)
                    frame.insert(0, "true_label", y)
                    frame.insert(0, "sample_id", [f"synthetic-{split}-{i}" for i in range(150)])
                    if seed != 42:
                        frame = frame.iloc[::-1]
                    name = f"{split}-{seed}.csv"
                    frame.to_csv(root / name, index=False)
                    record[split] = name
                rows.append(record)
            manifest = root / "experiments.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            means = mean_aligned_logits([root / r["calibration"] for r in rows])
            self.assertEqual(means.sample_id.iloc[0], "synthetic-calibration-0")
            # Fitting succeeds even when test files are temporarily unavailable.
            for row in rows:
                (root / row["final_test"]).rename(root / (row["final_test"] + ".hidden"))
            fit_ensemble(manifest, root / "policy")
            frozen = root / "policy/frozen_ensemble.json"
            snapshot = frozen.read_bytes()
            for row in rows:
                (root / (row["final_test"] + ".hidden")).rename(root / row["final_test"])
            evaluate_ensemble(manifest, frozen, root / "evaluation")
            self.assertEqual(snapshot, frozen.read_bytes())
            output = pd.read_csv(root / "evaluation/ensemble.csv")
            expected = (output.predicted_label == 0) & (output.prob_negative >= output.operating_threshold)
            np.testing.assert_array_equal(output.released, expected)
            # A forged overlap cannot silently pass the frozen evaluation guard.
            state = json.loads(snapshot)
            state["calibration_sample_ids"].append(output.sample_id.iloc[0])
            frozen.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "overlap"):
                evaluate_ensemble(manifest, frozen, root / "should_fail")
            # Same row count with different identities is also rejected.
            bad = pd.read_csv(root / rows[1]["calibration"])
            bad.loc[0, "sample_id"] = "unmatched-synthetic-id"
            bad.to_csv(root / rows[1]["calibration"], index=False)
            with self.assertRaisesRegex(ValueError, "same images"):
                mean_aligned_logits([root / r["calibration"] for r in rows])


if __name__ == "__main__":
    unittest.main()
