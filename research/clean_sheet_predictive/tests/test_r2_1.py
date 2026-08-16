from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.clean_sheet_predictive.r2_1_independence import (
    _actionability_row,
    _exact_clusters,
    _rank_redundancy,
)


class R21Tests(unittest.TestCase):
    def _spec(self):
        return {
            "minimum_cross_section_rows": 20,
            "extreme_fraction": 0.10,
            "eligible_r2_labels": ["ROBUST_CANDIDATE", "PROMISING", "WEAK"],
            "redundancy": {
                "exact_rank_equivalence_abs_rho": 0.9999,
                "exact_rank_equivalence_month_fraction": 0.95,
                "near_redundancy_median_abs_rho": 0.90,
                "near_redundancy_abs_rho_floor": 0.85,
                "near_redundancy_month_fraction": 0.80,
            },
            "bootstrap": {
                "block_months": 3,
                "repetitions": 300,
                "seed": 7,
            },
        }

    def test_within_date_equivalence_detects_constant_shift(self):
        rows = []
        for date in pd.date_range("2020-01-31", periods=24, freq="ME"):
            for i in range(40):
                a = float(i)
                rows.append({
                    "date": date,
                    "ticker": f"T{i:02d}",
                    "a": a,
                    "b": a - 12.5,   # same within-date rank
                    "c": float((i * 7) % 31),
                })
        frame = pd.DataFrame(rows)
        r = _rank_redundancy(
            frame,
            ["a", "b", "c"],
            min_rows=20,
            spec=self._spec(),
        )
        pair = r[
            ((r.feature_a == "a") & (r.feature_b == "b"))
            | ((r.feature_a == "b") & (r.feature_b == "a"))
        ].iloc[0]
        self.assertEqual(pair["relation"], "EXACT_RANK_EQUIVALENT")

    def test_near_redundancy_is_not_called_exact(self):
        rng = np.random.default_rng(11)
        rows = []
        for date in pd.date_range("2020-01-31", periods=24, freq="ME"):
            for i in range(60):
                a = rng.normal()
                b = a + rng.normal(0, 0.25)
                rows.append({"date": date, "ticker": f"T{i:02d}", "a": a, "b": b})
        r = _rank_redundancy(
            pd.DataFrame(rows),
            ["a", "b"],
            min_rows=20,
            spec=self._spec(),
        )
        self.assertNotEqual(r.iloc[0]["relation"], "EXACT_RANK_EQUIVALENT")

    def test_exact_representative_prefers_raw_return_over_cs_and_rs(self):
        dates = pd.date_range("2020-01-31", periods=12, freq="ME")
        rows = []
        for date in dates:
            for i in range(40):
                v = float(i)
                rows.append({
                    "date": date,
                    "ticker": f"T{i:02d}",
                    "ret_126": v,
                    "rs_spy_126": v - 5,
                    "cs_ret_126_pct": v / 40,
                })
        anchors = pd.DataFrame(rows)
        redundancy = _rank_redundancy(
            anchors,
            ["ret_126", "rs_spy_126", "cs_ret_126_pct"],
            min_rows=20,
            spec=self._spec(),
        )
        r2 = pd.DataFrame({
            "feature": ["ret_126", "rs_spy_126", "cs_ret_126_pct"],
            "mean_monthly_ic": [0.05, 0.05, 0.05],
        })
        clusters, mapping = _exact_clusters(
            ["ret_126", "rs_spy_126", "cs_ret_126_pct"],
            redundancy,
            anchors,
            r2,
        )
        self.assertEqual(mapping["rs_spy_126"], "ret_126")
        self.assertEqual(mapping["cs_ret_126_pct"], "ret_126")

    def test_actionability_can_be_avoid_only(self):
        rng = np.random.default_rng(4)
        rows = []
        dates = pd.date_range("2017-01-31", periods=72, freq="ME")
        for date in dates:
            for i in range(80):
                x = i / 79
                # Favor side is around zero; avoid side clearly negative.
                y = rng.normal(0, 0.01) if x > 0.5 else rng.normal(-0.06, 0.01)
                rows.append({
                    "date": date,
                    "ticker": f"T{i:02d}",
                    "x": x,
                    "fwd_spy_relative_return_63": y,
                    "fwd_max_drawdown_63": -0.10,
                    "fwd_spy_rel_mae_63": -0.05,
                })
        anchors = pd.DataFrame(rows)
        r2row = pd.Series({
            "feature": "x",
            "family": "test",
            "horizon_bars": 63,
            "horizon_label": "3M",
            "evidence_label": "ROBUST_CANDIDATE",
            "mean_monthly_ic": 0.10,
            "direction_if_used": "HIGHER_FEATURE_FAVORABLE",
        })
        result, _ = _actionability_row(anchors, r2row, self._spec())
        self.assertTrue(result["avoid_actionable"])
        self.assertFalse(result["favor_actionable"])
        self.assertEqual(result["actionability_label"], "AVOID_ACTIONABLE")


if __name__ == "__main__":
    unittest.main()
