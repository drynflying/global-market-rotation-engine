from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.clean_sheet_predictive.r3_walk_forward import (
    _candidate_features_by_horizon,
    _cross_sectional_rank_transform,
    _eligible_training_rows,
    _make_avoid_label,
    _fit_fold,
)


class R3Tests(unittest.TestCase):
    def _spec(self):
        return {
            "horizons_bars": [63],
            "models": {
                "return_ranking": {"alpha": 1.0, "fit_intercept": True},
                "avoid_classifier": {
                    "C": 1.0,
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "max_iter": 2000,
                },
            },
        }

    def test_candidate_gate_uses_only_selected(self):
        c = pd.DataFrame(
            {
                "horizon_bars": [21, 21, 63],
                "feature": ["a", "b", "c"],
                "status": ["SELECTED", "EXCLUDED", "SELECTED"],
            }
        )
        spec = {"horizons_bars": [21, 63]}
        got = _candidate_features_by_horizon(c, spec)
        self.assertEqual(got[21], ["a"])
        self.assertEqual(got[63], ["c"])

    def test_cross_sectional_rank_is_date_local(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-31"] * 3 + ["2024-02-29"] * 3),
                "ticker": ["A", "B", "C"] * 2,
                "x": [1, 2, 3, 100, 200, 300],
            }
        )
        ranked = _cross_sectional_rank_transform(frame, ["x"])
        jan = ranked[ranked["date"].dt.month.eq(1)]["x"].tolist()
        feb = ranked[ranked["date"].dt.month.eq(2)]["x"].tolist()
        self.assertEqual(jan, feb)
        self.assertAlmostEqual(max(jan), 1.0)

    def test_avoid_label_is_bottom_quintile(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-31"] * 100),
                "y": np.arange(100, dtype=float),
            }
        )
        label = _make_avoid_label(frame, "y", 0.20)
        self.assertEqual(int(label.sum()), 20)
        self.assertEqual(float(label.iloc[0]), 1.0)
        self.assertEqual(float(label.iloc[-1]), 0.0)

    def test_training_cutoff_excludes_immature_outcomes(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2019-01-31", "2019-02-28", "2019-03-31"]),
                "ticker": ["A", "A", "A"],
                "x": [0.1, 0.2, 0.3],
                "target": [0.01, 0.02, 0.03],
                "end": pd.to_datetime(["2019-04-01", "2020-01-02", "2019-06-01"]),
                "avoid_label": [0.0, 1.0, 0.0],
            }
        )
        train = _eligible_training_rows(
            frame,
            features=["x"],
            target="target",
            outcome_end_col="end",
            cutoff=pd.Timestamp("2020-01-01"),
        )
        self.assertEqual(len(train), 2)
        self.assertTrue((train["end"] < pd.Timestamp("2020-01-01")).all())

    def test_synthetic_fold_learns_return_and_avoid_signal(self):
        rng = np.random.default_rng(123)
        train_rows = []
        test_rows = []
        for date in pd.date_range("2016-01-31", periods=36, freq="ME"):
            for i in range(80):
                x = i / 79
                y = 0.08 * x + rng.normal(0, 0.01)
                avoid = float(x <= 0.20)
                train_rows.append({"date": date, "ticker": f"T{i:02d}", "x": x, "y": y, "avoid_label": avoid})
        for date in pd.date_range("2020-01-31", periods=6, freq="ME"):
            for i in range(80):
                x = i / 79
                y = 0.08 * x + rng.normal(0, 0.01)
                avoid = float(x <= 0.20)
                test_rows.append({"date": date, "ticker": f"T{i:02d}", "x": x, "y": y, "avoid_label": avoid})
        train = pd.DataFrame(train_rows)
        test = pd.DataFrame(test_rows)
        pred, _, _ = _fit_fold(
            train,
            test,
            features=["x"],
            target="y",
            horizon=63,
            year=2020,
            spec=self._spec(),
        )
        self.assertGreater(pred["ridge_prediction"].corr(pred["y"]), 0.8)
        avoid_mean_low_x = pred.loc[test["x"].to_numpy() <= 0.20, "avoid_probability"].mean()
        avoid_mean_high_x = pred.loc[test["x"].to_numpy() >= 0.80, "avoid_probability"].mean()
        self.assertGreater(avoid_mean_low_x, avoid_mean_high_x)


if __name__ == "__main__":
    unittest.main()
