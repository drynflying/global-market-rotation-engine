from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research.clean_sheet_predictive.r2_univariate import (
    _cross_section_month_stats,
    _evidence_label,
    _market_regime_evaluation,
    _month_end_anchor_dates,
    _require_valid_r1,
)


class R2Tests(unittest.TestCase):
    def test_month_end_anchor_is_last_available_date(self):
        dates = pd.to_datetime([
            "2024-01-02", "2024-01-31",
            "2024-02-01", "2024-02-28", "2024-02-29",
        ])
        anchors = _month_end_anchor_dates(pd.Series(dates))
        self.assertEqual(
            list(anchors.strftime("%Y-%m-%d")),
            ["2024-01-31", "2024-02-29"],
        )

    def test_positive_cross_sectional_feature_produces_positive_ic_and_spread(self):
        rows = []
        rng = np.random.default_rng(7)
        for month in pd.date_range("2020-01-31", periods=12, freq="ME"):
            for i in range(60):
                x = i / 59
                y = 0.10 * x + rng.normal(0, 0.003)
                rows.append({"date": month, "ticker": f"T{i:02d}", "x": x, "y": y})
        frame = pd.DataFrame(rows)
        stats = _cross_section_month_stats(frame, "x", "y", min_rows=30)
        self.assertGreater(stats["ic"].mean(), 0.90)
        self.assertGreater(stats["high_minus_low_mean"].mean(), 0)

    def test_negative_cross_sectional_feature_is_not_flipped(self):
        rows = []
        for month in pd.date_range("2020-01-31", periods=8, freq="ME"):
            for i in range(50):
                x = i / 49
                y = -0.05 * x
                rows.append({"date": month, "ticker": f"T{i:02d}", "x": x, "y": y})
        stats = _cross_section_month_stats(pd.DataFrame(rows), "x", "y", min_rows=30)
        self.assertLess(stats["ic"].mean(), -0.99)
        self.assertLess(stats["high_minus_low_mean"].mean(), 0)

    def test_r1_integrity_gate_blocks_incomplete_universe(self):
        spec = {
            "dataset_requirement": {
                "r1_validation_status": "ok",
                "universe_completeness_status": "passed",
                "expected_tickers": 167,
            }
        }
        manifest = {"dataset_tickers": 127}
        validation = {
            "status": "ok",
            "universe_completeness": {
                "status": "failed",
                "missing_symbols": ["AAA"],
            },
        }
        with self.assertRaises(RuntimeError):
            _require_valid_r1(manifest, validation, spec)

    def test_evidence_gate_requires_consistent_spread_and_ci(self):
        spec = {
            "evidence_labels": {
                "ROBUST_CANDIDATE": {
                    "minimum_months": 60,
                    "abs_mean_monthly_ic_min": 0.03,
                    "annual_direction_consistency_min": 0.70,
                },
                "PROMISING": {
                    "minimum_months": 60,
                    "abs_mean_monthly_ic_min": 0.02,
                    "annual_direction_consistency_min": 0.60,
                },
                "WEAK": {
                    "minimum_months": 48,
                    "abs_mean_monthly_ic_min": 0.01,
                },
            }
        }
        label = _evidence_label(
            mean_ic=0.04,
            ci_lo=0.02,
            ci_hi=0.06,
            annual_consistency=0.80,
            spread=0.03,
            months=100,
            spec=spec,
        )
        self.assertEqual(label, "ROBUST_CANDIDATE")

        inconsistent = _evidence_label(
            mean_ic=0.04,
            ci_lo=0.02,
            ci_hi=0.06,
            annual_consistency=0.80,
            spread=-0.03,
            months=100,
            spec=spec,
        )
        self.assertNotEqual(inconsistent, "ROBUST_CANDIDATE")


if __name__ == "__main__":
    unittest.main()
