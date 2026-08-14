from __future__ import annotations

import sys
import types
import unittest

import pandas as pd

# src.backfill_10y imports yfinance at module import time. Unit tests exercise
# only pure merge/quality helpers, so a minimal stub is enough locally.
if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = types.SimpleNamespace(download=None)

from src.backfill_10y import _assess_symbol_quality, _merge_frames


class R11BackfillTests(unittest.TestCase):
    def _frame(self, ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        close = pd.Series(range(100, 100 + len(dates)), dtype=float)
        return pd.DataFrame({
            "date": dates,
            "ticker": ticker,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000_000,
        })

    def test_merge_preserves_successful_rows_across_passes(self):
        dates = pd.bdate_range("2024-01-01", periods=300)
        first = self._frame("AAA", dates)
        second = self._frame("BBB", dates)
        merged = _merge_frames([first, second])
        self.assertEqual(set(merged["ticker"]), {"AAA", "BBB"})
        self.assertEqual(len(merged), 600)

    def test_quality_fails_missing_symbol(self):
        dates = pd.bdate_range("2024-01-01", periods=300)
        data = self._frame("AAA", dates)
        quality = _assess_symbol_quality(
            data,
            ["AAA", "BBB"],
            global_latest=pd.Timestamp(dates[-1]),
            min_bars=260,
            min_coverage_ratio=0.85,
            max_gap_days=21,
            max_stale_days=7,
            allowed_missing=set(),
        )
        bad = quality.set_index("ticker").loc["BBB"]
        self.assertFalse(bool(bad["quality_ok"]))
        self.assertIn("no usable OHLCV", bad["reason"])

    def test_quality_rejects_material_internal_gap(self):
        d1 = pd.bdate_range("2024-01-01", periods=150)
        d2 = pd.bdate_range("2024-10-01", periods=150)
        data = _merge_frames([self._frame("AAA", d1), self._frame("AAA", d2)])
        quality = _assess_symbol_quality(
            data,
            ["AAA"],
            global_latest=pd.Timestamp(data["date"].max()),
            min_bars=260,
            min_coverage_ratio=0.50,
            max_gap_days=21,
            max_stale_days=7,
            allowed_missing=set(),
        )
        row = quality.iloc[0]
        self.assertFalse(bool(row["quality_ok"]))
        self.assertIn("max internal calendar gap", row["reason"])


if __name__ == "__main__":
    unittest.main()
