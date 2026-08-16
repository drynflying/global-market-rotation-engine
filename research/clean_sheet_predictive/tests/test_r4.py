from __future__ import annotations

import copy
import unittest

import numpy as np
import pandas as pd

from research.clean_sheet_predictive.r4_prospective import (
    _completed_anchor_dates,
    _prediction_id,
    _score_anchor_horizon,
    _status_for,
)


class R4ProspectiveTests(unittest.TestCase):
    def _spec(self):
        return {
            "horizon_labels": {"21": "1M"},
            "minimum_prediction_cross_section_rows": 5,
        }

    def test_completed_anchor_requires_a_later_calendar_month(self):
        dates = pd.Series(
            pd.to_datetime(
                [
                    "2026-08-28",
                    "2026-08-31",
                    "2026-09-01",
                    "2026-09-02",
                ]
            )
        )
        anchors = _completed_anchor_dates(dates, "2026-08-31")
        self.assertEqual(
            [d.strftime("%Y-%m-%d") for d in anchors],
            ["2026-08-31"],
        )

    def test_no_pre_inception_anchor_is_backfilled(self):
        dates = pd.Series(
            pd.to_datetime(
                [
                    "2026-06-30",
                    "2026-07-31",
                    "2026-08-31",
                    "2026-09-01",
                ]
            )
        )
        anchors = _completed_anchor_dates(dates, "2026-08-31")
        self.assertEqual(
            [d.strftime("%Y-%m-%d") for d in anchors],
            ["2026-08-31"],
        )

    def test_prediction_id_is_stable_and_model_version_specific(self):
        a = _prediction_id(
            "CLEAN_SHEET_R4_2026_V1",
            pd.Timestamp("2026-08-31"),
            126,
            "XLK",
        )
        b = _prediction_id(
            "CLEAN_SHEET_R4_2026_V1",
            pd.Timestamp("2026-08-31"),
            126,
            "XLK",
        )
        c = _prediction_id(
            "CLEAN_SHEET_R4_2027_V1",
            pd.Timestamp("2026-08-31"),
            126,
            "XLK",
        )
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_scoring_uses_frozen_standardization_and_orientation(self):
        rows = []
        for i in range(10):
            rows.append(
                {
                    "date": pd.Timestamp("2026-08-31"),
                    "ticker": "SPY" if i == 0 else f"T{i}",
                    "close": 100 + i,
                    "f1": float(i),
                    "f2": float(10 - i),
                }
            )
        anchor = pd.DataFrame(rows)
        model_year = {
            "model_version": "TEST_2026",
            "model_year": 2026,
            "horizons": {
                "21": {
                    "features": ["f1", "f2"],
                    "scaler_mean": [0.5, 0.5],
                    "scaler_scale": [0.25, 0.25],
                    "ridge": {
                        "intercept": 0.0,
                        "coefficients": [1.0, -1.0],
                    },
                    "logistic": {
                        "intercept": 0.0,
                        "coefficients": [-1.0, 1.0],
                    },
                    "equal_weight_orientation": {"f1": 1, "f2": -1},
                }
            },
        }
        registry = {"frozen_universe_count": 10}
        pred, manifest = _score_anchor_horizon(
            anchor,
            model_year_record=model_year,
            horizon=21,
            spec=self._spec(),
            registry=registry,
            issued_at="2026-09-01T22:00:00+00:00",
        )
        self.assertEqual(len(pred), 10)
        self.assertEqual(manifest["eligible_rows"], 10)
        # Highest f1 / lowest f2 must rank highest for Ridge and equal weight.
        top_ridge = pred.nlargest(1, "ridge_prediction").iloc[0]["ticker"]
        top_equal = pred.nlargest(1, "equal_weight_score").iloc[0]["ticker"]
        self.assertEqual(top_ridge, "T9")
        self.assertEqual(top_equal, "T9")

    def test_evidence_status_does_not_confirm_early(self):
        self.assertEqual(_status_for(0, False), "INSUFFICIENT")
        self.assertEqual(_status_for(5, True), "EARLY")
        self.assertEqual(_status_for(12, True), "PROMISING")
        self.assertEqual(_status_for(12, False), "MIXED")
        self.assertEqual(_status_for(24, True), "CONFIRMED_PROSPECTIVE")
        self.assertEqual(_status_for(24, False), "FAILED_TO_CONFIRM")


if __name__ == "__main__":
    unittest.main()
