import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import export_tara as E


class RecordExportTests(unittest.TestCase):
    def test_refresh_preserves_forecasts_and_moves_only_resolved_ids(self):
        previous = {
            "generated": "2026-10-05T15:00:00Z", "record": {"n": 1},
            "board": [{"ledger_id": 2, "tracked": True}, {"ledger_id": 3, "tracked": True},
                      {"ledger_id": None, "mkt": .4}],
            "open": [{"id": 2, "start": "2026-10-04T17:00:00Z"}], "settled": [],
            "upcoming": [{"mkt": .42}], "predictions": {"stale": True}, "totals": {"n": 5}}
        original = copy.deepcopy(previous)
        fresh = {"generated": "2026-10-05T21:00:00Z", "record": {"n": 2},
                 "open": [], "settled": [{"id": 2, "start": "2026-10-04"}],
                 "disclosures": [{"id": 3, "kind": "non-binary"}]}
        with patch.object(E, "build", return_value=fresh):
            result = E.refresh_settlements(previous)
        self.assertEqual(previous, original)
        for key in ("generated", "upcoming", "predictions", "totals"):
            self.assertEqual(result[key], original[key])
        self.assertEqual(result["record_updated"], fresh["generated"])
        self.assertEqual(result["settled"][0]["start"], original["open"][0]["start"])
        self.assertEqual(result["board"], [{"ledger_id": None, "mkt": .4}])
        self.assertEqual(result["board_counts"], {"total": 1, "tracked": 0, "high_conviction": 0})

    def test_missing_board_rejected(self):
        with self.assertRaises(ValueError):
            E.refresh_settlements({})
