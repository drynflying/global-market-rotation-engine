from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.clean_sheet_predictive.r4_model_builder import _date_weights


class R4ModelBuilderTests(unittest.TestCase):
    def test_date_weights_balance_each_training_month(self):
        dates = pd.Series(
            pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-01-31",
                    "2020-01-31",
                    "2020-02-29",
                    "2020-02-29",
                    "2020-03-31",
                ]
            )
        )
        w = _date_weights(dates)
        frame = pd.DataFrame({"date": dates, "w": w})
        totals = frame.groupby("date")["w"].sum().to_numpy()
        self.assertTrue(np.allclose(totals, totals[0]))
        self.assertAlmostEqual(float(w.sum()), float(len(w)), places=10)


if __name__ == "__main__":
    unittest.main()
