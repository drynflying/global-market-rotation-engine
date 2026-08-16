from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.clean_sheet_predictive.build_features import build_features
from research.clean_sheet_predictive.build_outcomes import add_forward_outcomes
from research.clean_sheet_predictive.split_rules import assert_training_cutoff_integrity, training_rows_as_of
from research.clean_sheet_predictive.validate_point_in_time import universe_completeness_check


class R1Tests(unittest.TestCase):
    def _sample(self, n=320):
        dates = pd.bdate_range("2020-01-01", periods=n)
        frames = []
        for ticker, drift in [("SPY", 0.0010), ("AAA", 0.0015), ("BBB", 0.0005)]:
            close = 100 * np.cumprod(np.full(n, 1 + drift))
            frames.append(pd.DataFrame({
                "date": dates,
                "ticker": ticker,
                "open": close * 0.999,
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "volume": np.linspace(1_000_000, 1_300_000, n),
            }))
        raw = pd.concat(frames, ignore_index=True)
        cfg = pd.DataFrame({
            "ticker": ["SPY", "AAA", "BBB"],
            "primary_benchmark": ["SPY", "SPY", "SPY"],
            "exposure": ["S&P 500", "AAA", "BBB"],
            "universe": ["US", "TEST", "TEST"],
            "rotation_group": ["TEST", "TEST", "TEST"],
            "level": ["broad", "test", "test"],
            "asset_type": ["ETF", "ETF", "ETF"],
        })
        return raw, cfg

    def test_features_are_future_invariant(self):
        raw, cfg = self._sample()
        cutoff = pd.Timestamp("2020-12-31")
        a = build_features(raw, cfg)
        mutated = raw.copy()
        mask = mutated["date"] > cutoff
        mutated.loc[mask, ["open", "high", "low", "close", "volume"]] *= 10
        b = build_features(mutated, cfg)
        cols = ["ret_63", "ret_126", "sma_200_slope_20", "vol_126", "cmf63", "mkt_dispersion_ret63"]
        aa = a[a["date"] <= cutoff].sort_values(["ticker", "date"])[cols].reset_index(drop=True)
        bb = b[b["date"] <= cutoff].sort_values(["ticker", "date"])[cols].reset_index(drop=True)
        self.assertTrue(np.allclose(aa.to_numpy(float), bb.to_numpy(float), equal_nan=True))

    def test_forward_return_alignment(self):
        raw, cfg = self._sample()
        ds = add_forward_outcomes(build_features(raw, cfg))
        row = ds[(ds["ticker"] == "AAA")].sort_values("date").iloc[250]
        expected = (1.0015 ** 21) - 1
        self.assertAlmostEqual(float(row["fwd_return_21"]), expected, places=10)
        self.assertGreater(float(row["fwd_spy_relative_return_21"]), 0.0)

    def test_training_rows_require_mature_outcome(self):
        raw, cfg = self._sample()
        ds = add_forward_outcomes(build_features(raw, cfg))
        cutoff = pd.Timestamp("2021-02-01")
        train = training_rows_as_of(ds, 63, cutoff)
        assert_training_cutoff_integrity(train, 63, cutoff)
        self.assertTrue((pd.to_datetime(train["outcome_end_date_63"]) <= cutoff).all())

    def test_universe_completeness_fails_when_active_symbol_is_missing(self):
        raw, _ = self._sample()
        cfg = pd.DataFrame({
            "ticker": ["SPY", "AAA", "BBB", "CCC"],
            "query_symbol": ["SPY", "AAA", "BBB", "CCC"],
            "enabled": [True, True, True, True],
            "query_ohlcv": [True, True, True, True],
        })
        report = universe_completeness_check(raw, cfg)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["missing_symbols"], ["CCC"])

    def test_universe_completeness_passes_when_all_active_symbols_exist(self):
        raw, _ = self._sample()
        cfg = pd.DataFrame({
            "ticker": ["SPY", "AAA", "BBB"],
            "query_symbol": ["SPY", "AAA", "BBB"],
            "enabled": [True, True, True],
            "query_ohlcv": [True, True, True],
        })
        report = universe_completeness_check(raw, cfg)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["completeness_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
