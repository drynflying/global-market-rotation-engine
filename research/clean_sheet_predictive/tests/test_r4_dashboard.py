from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research.clean_sheet_predictive.r4_dashboard import build_dashboard


class R41DashboardTests(unittest.TestCase):
    def test_empty_prospective_state_renders_without_fake_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = root / "results"
            result.mkdir()
            spec = root / "spec.json"
            output = root / "r4.html"

            spec.write_text(json.dumps({
                "inception_date": "2026-08-15",
                "first_eligible_anchor_date": "2026-08-31",
                "primary_research_horizon_bars": 126,
                "horizons_bars": [21, 63, 126, 189],
                "horizon_labels": {"21":"1M","63":"3M","126":"6M","189":"9M"},
                "candidate_features_by_horizon": {
                    "21":["a"],"63":["b"],"126":["c"],"189":["d"]
                }
            }))
            (result / "r4_status.json").write_text(json.dumps({
                "prospective_inception_date":"2026-08-15",
                "first_eligible_anchor_date":"2026-08-31",
                "latest_market_date":"2026-08-14",
                "latest_issued_anchor_date":None,
                "prediction_rows":0,
                "matured_outcome_rows":0,
                "frozen_universe_count":167
            }))
            (result / "r4_assessment.json").write_text(json.dumps({
                "by_horizon": {
                    str(h): {
                        "horizon_label": label,
                        "evaluated_months": 0,
                        "evidence_status": "INSUFFICIENT",
                        "ranking": {
                            "ridge":{"favor_gate_pass":False},
                            "equal_weight":{"favor_gate_pass":False}
                        },
                        "avoid":{"avoid_gate_pass":False}
                    }
                    for h,label in [(21,"1M"),(63,"3M"),(126,"6M"),(189,"9M")]
                }
            }))
            (result / "r4_latest.json").write_text(json.dumps({
                "status":"waiting_for_first_completed_month",
                "latest_anchor_date":None,
                "by_horizon":{}
            }))

            build_dashboard(result, spec, output)
            text = output.read_text()
            self.assertIn("R4 Prospective Research Dashboard", text)
            self.assertIn("Waiting for the first completed month", text)
            self.assertIn("Aug 31, 2026", text)
            self.assertIn("Prediction rows", text)
            self.assertNotIn("T9</strong>", text)


if __name__ == "__main__":
    unittest.main()
