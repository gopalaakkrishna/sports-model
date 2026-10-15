import copy
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paper_reporting import paper_performance
import open_lane as O
import totals_ledger as T


class PriceMatchedReportingTests(unittest.TestCase):
    def test_edge_and_returns_use_exact_same_price_matched_rows(self):
        rows = [{"won": True, "ask": .6}, {"won": False, "ask": .5},
                {"won": True, "ask": None}, {"won": True, "ask": None}]
        original = copy.deepcopy(rows)
        r = paper_performance(rows)
        self.assertEqual(rows, original)
        self.assertEqual(r["wins"], 3)
        self.assertEqual(r["priced_n"], 2)
        self.assertEqual(r["unpriced_n"], 2)
        self.assertEqual(r["priced_win_rate"], .5)
        self.assertAlmostEqual(r["edge"], r["priced_win_rate"] - r["breakeven"])
        self.assertAlmostEqual(r["edge"] * 2, r["units"])
        self.assertNotAlmostEqual(r["edge"], .75 - r["breakeven"])

    def test_missing_invalid_and_void_prices_are_not_invented(self):
        rows = [{"won": True, "ask": p} for p in [None, 0, 1, -1, math.nan, "bad", True]]
        rows += [{"won": True, "ask": .4, "voided": True}, {"won": None, "ask": .4}, {"won": "SCALAR", "ask": .4}]
        r = paper_performance(rows)
        self.assertEqual(r["priced_n"], 0)
        for key in ["edge", "units", "priced_win_rate", "breakeven"]:
            self.assertIsNone(r[key])

    def test_totals_no_extra_vig_or_different_return_denominator(self):
        r = paper_performance([{"won": True, "market_prob": .6}, {"won": False, "market_prob": .5}], "market_prob")
        self.assertAlmostEqual(r["breakeven"], .55 + .07 * (.6*.4+.5*.5)/2)
        self.assertAlmostEqual(r["units"], r["edge"] * r["priced_n"])

    def test_actual_totals_and_open_cli_read_only_reports_share_the_same_math(self):
        rows = [{"won": True, "ask": .6, "market_prob": .6, "market": "ML"},
                {"won": False, "ask": .5, "market_prob": .5, "market": "ML"}]
        with patch.object(O, "_load", return_value=rows), patch.object(O.PS, "health", return_value={"pending": 0}):
            self.assertAlmostEqual(O.report()["edge"], paper_performance(rows)["edge"])
        with patch.object(T, "_load", return_value=rows), patch.object(T.PS, "health", return_value={"pending": 0}):
            self.assertAlmostEqual(T.report()["edge"], paper_performance(rows, "market_prob")["edge"])

    def test_actual_cli_reports_missing_prices_as_unavailable_without_mutating_rows(self):
        rows = [{"won": True, "ask": None, "market_prob": "bad", "market": "ML"},
                {"won": False, "ask": None, "market_prob": None, "market": "ML"}]
        original = copy.deepcopy(rows)
        for module in [O, T]:
            with patch.object(module, "_load", return_value=rows), patch.object(module.PS, "health", return_value={"pending": 0}):
                r = module.report()
                self.assertEqual(r["unpriced_n"], 2)
                self.assertIsNone(r["edge"])
        self.assertEqual(rows, original)
