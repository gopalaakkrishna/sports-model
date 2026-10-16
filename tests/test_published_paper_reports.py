import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import export_tara as E
import open_lane as O
import totals_ledger as T
from paper_reporting import paper_performance
import verify_paper_reports as V


class PublishedPaperContractTests(unittest.TestCase):
    def rows(self):
        return [
            {"id": 1, "won": True, "ask": .6, "market_prob": .8},
            {"id": 2, "won": False, "ask": .5, "market_prob": .7},
            {"id": 3, "won": True, "ask": None, "market_prob": None},
            {"id": 4, "won": True, "ask": None, "market_prob": None},
        ]

    def test_independent_check_rejects_original_mixed_cohort_regression(self):
        rows = self.rows()
        report = paper_performance(rows)
        self.assertEqual(report["win_rate"], .75)
        self.assertEqual(report["priced_win_rate"], .5)
        self.assertAlmostEqual(report["breakeven"], .56715)
        self.assertAlmostEqual(report["edge"], -.06715)
        self.assertAlmostEqual(report["units"], -.1343)
        V.verify_summary(report, rows)
        broken = dict(report, edge=report["win_rate"] - report["breakeven"])
        with self.assertRaisesRegex(ValueError, "paper.edge"):
            V.verify_summary(broken, rows)

    def test_both_lanes_use_their_own_original_price_and_disclose_unpriced(self):
        rows = self.rows()
        board = {"open_lane": paper_performance(rows),
                 "totals": {"paper_record": paper_performance(rows, "market_prob")}}
        V.verify_board(board, rows, rows)
        self.assertNotEqual(board["open_lane"]["edge"], board["totals"]["paper_record"]["edge"])
        broken = copy.deepcopy(board)
        broken["totals"]["paper_record"] = broken["open_lane"]
        with self.assertRaisesRegex(ValueError, "totals.paper_record.breakeven"):
            V.verify_board(broken, rows, rows)
        broken = copy.deepcopy(board)
        broken["open_lane"]["unpriced_n"] = 0
        with self.assertRaisesRegex(ValueError, "open_lane.unpriced_n"):
            V.verify_board(broken, rows, rows)

    def test_missing_invalid_scalar_and_voided_records_do_not_get_invented_prices(self):
        rows = [{"won": True, "ask": value} for value in [None, True, 0, 1, math.nan, "bad"]]
        rows += [{"won": None, "ask": .4}, {"won": "SCALAR", "ask": .4},
                 {"won": True, "ask": .4, "voided": True}]
        original = copy.deepcopy(rows)
        report = paper_performance(rows)
        V.verify_summary(report, rows)
        self.assertEqual(report["priced_n"], 0)
        self.assertEqual(report["unpriced_n"], 6)
        for field in ("edge", "units", "priced_win_rate", "breakeven"):
            self.assertIsNone(report[field])
        with self.assertRaisesRegex(ValueError, "paper.priced_win_rate"):
            V.verify_summary(dict(report, priced_win_rate=0), rows)
        self.assertEqual([r.get("won") for r in rows], [r.get("won") for r in original])
        self.assertIsNone(rows[0]["ask"])

    def test_duplicate_ids_are_refused_without_rewriting_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            path.write_text('{"id":1}\n{"id":1}\n', encoding="utf-8")
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "duplicate record IDs"):
                V.load_rows(path)
            self.assertEqual(path.read_bytes(), before)

    def test_actual_export_record_refresh_and_json_reload_keep_matched_cohorts(self):
        originals = {name: (ROOT / "data/processed" / name).read_bytes()
                     for name in ("ledger.jsonl", "open_ledger.jsonl", "totals_ledger.jsonl")}
        canonical = V.load_rows(ROOT / "data/processed/open_ledger.jsonl")
        templates = [r for r in canonical if isinstance(r.get("won"), bool) and not r.get("voided")][:4]
        rows = [dict(copy.deepcopy(row), **fixture) for row, fixture in zip(templates, self.rows())]
        captured = copy.deepcopy(rows)
        totals = V.load_rows(ROOT / "data/processed/totals_ledger.jsonl")
        with patch.object(O, "_load", return_value=rows), patch.object(T, "_load", return_value=totals):
            fresh = E.build([], E.StartTimes())
            V.verify_board(fresh, rows, totals)
            refreshed = E.refresh_settlements(fresh)
            self.assertEqual(refreshed["generated"], fresh["generated"])
            V.verify_board(refreshed, rows, totals)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "sports.json"
                path.write_text(json.dumps(refreshed, allow_nan=False), encoding="utf-8")
                reloaded = json.loads(path.read_text(encoding="utf-8"))
            V.verify_board(reloaded, rows, totals)
            self.assertEqual(reloaded["open_lane"]["wins"], 3)
            self.assertEqual(reloaded["open_lane"]["priced_wins"], 1)
            self.assertEqual(reloaded["open_lane"]["edge"], round(-.06715, 4))
        self.assertEqual(rows, captured)
        for name, before in originals.items():
            self.assertEqual((ROOT / "data/processed" / name).read_bytes(), before, name)


if __name__ == "__main__":
    unittest.main()
