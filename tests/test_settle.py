import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import settle as S
from nfl_teams import nfl_team
from supabase_sync import merge


class SettlementTests(unittest.TestCase):
    def setUp(self):
        S._K_CACHE.clear()
        S._EVIDENCE.clear()
        S._SCALAR_RESULTS.clear()
        S._PARSE_FAILURES.clear()

    def nfl_api(self, subtitle="SF vs LAR (Sep 10)", winner="Los Angeles R", result="yes"):
        event = "KXNFLGAME-26SEP10SFLAR"
        def get(url, **kwargs):
            body = ({"events": [{"event_ticker": event, "title": "SF 49ers vs LA Rams",
                                "sub_title": subtitle}]} if url.endswith("/events") else
                    {"markets": [{"event_ticker": event, "ticker": event + "-LAR",
                                  "yes_sub_title": winner, "result": result}]})
            return Mock(json=lambda: body, raise_for_status=lambda: None)
        return patch.object(S.requests, "get", side_effect=get)

    def test_nfl_abbreviation_title_and_city_leg(self):
        row = {"event": "SF @ LA (2026-09-10)", "pick": "HOME"}
        with self.nfl_api():
            result = S.settle_via_kalshi(row, "nfl")
        self.assertEqual(result[0], "HOME")
        self.assertEqual(row["settlement_evidence"]["ticker"], "KXNFLGAME-26SEP10SFLAR-LAR")

    def test_away_win(self):
        with self.nfl_api(winner="San Francisco"):
            self.assertEqual(S.settle_via_kalshi({"event": "SF @ LA (2026-09-10)"}, "nfl")[0], "AWAY")

    def test_legacy_code_and_new_york_mascots(self):
        self.assertEqual(nfl_team("TEN Titans"), "TEN")
        self.assertEqual(nfl_team("NY Giants"), "NYG")
        self.assertEqual(nfl_team("NY Jets"), "NYJ")

    def test_rate_limit_retries_without_losing_results(self):
        throttled = Mock(status_code=429)
        ok = Mock(status_code=200)
        with patch.object(S.requests, "get", side_effect=[throttled, ok]) as get, patch.object(S.time, "sleep"):
            self.assertIs(S.kalshi_get("https://example.test"), ok)
            self.assertEqual(get.call_count, 2)

    def test_known_soccer_league_does_not_scan_every_league(self):
        with patch.object(S, "_kalshi_soccer_results", return_value={}) as results:
            S.settle_soccer_via_kalshi({"event": "A v B (2026-09-05)", "league": "USA MLS", "pick": "HOME"})
        results.assert_called_once_with(["KXMLSGAME"])

    def test_exact_date_not_previous_day(self):
        with self.nfl_api():
            self.assertIsNone(S.settle_via_kalshi({"event": "SF @ LA (2026-09-11)"}, "nfl"))

    def test_same_city_is_not_same_franchise(self):
        with self.nfl_api():
            self.assertIsNone(S.settle_via_kalshi({"event": "SF @ LAC (2026-09-10)"}, "nfl"))
        self.assertEqual(nfl_team("New York G"), "NYG")
        self.assertEqual(nfl_team("New York J"), "NYJ")
        self.assertIsNone(nfl_team("New York"))
        self.assertIsNone(nfl_team("Los Angeles"))
        self.assertEqual(nfl_team("JAC"), "JAX")

    def test_unknown_winner_not_guessed(self):
        with self.nfl_api(winner="Los Angeles"):
            self.assertIsNone(S.settle_via_kalshi({"event": "SF @ LA (2026-09-10)"}, "nfl"))
        self.assertTrue(S._PARSE_FAILURES)

    def test_unsettled_not_scored(self):
        with self.nfl_api(result=""):
            self.assertIsNone(S.settle_via_kalshi({"event": "SF @ LA (2026-09-10)"}, "nfl"))

    def test_missing_subtitle_fails_visibly(self):
        with self.nfl_api(subtitle=""):
            self.assertIsNone(S.settle_via_kalshi({"event": "SF @ LA (2026-09-10)"}, "nfl"))
        self.assertTrue(S._PARSE_FAILURES)

    def test_request_failure_not_healthy_empty_record(self):
        with patch.object(S.requests, "get", side_effect=S.requests.Timeout("timeout")):
            self.assertEqual(S._kalshi_results("KXNFLGAME"), {})
        self.assertTrue(S._PARSE_FAILURES)

    def test_scalar_payout_excluded_not_loss_or_refund(self):
        event = "KXMLSGAME-26SEP05CINDCU"
        def get(url, **kwargs):
            body = ({"events": [{"event_ticker": event, "title": "Cincinnati vs DC United"}]}
                    if url.endswith("/events") else {"markets": [{
                        "event_ticker": event, "ticker": event + "-CIN", "yes_sub_title": "Cincinnati",
                        "result": "scalar", "status": "finalized", "settlement_value_dollars": "0.5900"}]})
            return Mock(json=lambda: body, raise_for_status=lambda: None)
        row = {"id": 116, "event": "FC Cincinnati v DC United (2026-09-05)", "pick": "HOME", "model_prob": .65}
        original = copy.deepcopy(row)
        with patch.object(S.requests, "get", side_effect=get), patch.object(S, "_KALSHI_SOCCER_SERIES", ["KXMLSGAME"]):
            result = S.settle_soccer_via_kalshi(row)
        self.assertEqual(result[0], "SCALAR")
        S.apply_settlement(row, *result)
        self.assertIsNone(row["won"])
        self.assertTrue(row["voided"])
        self.assertEqual(row["settlement_evidence"]["payout_dollars"], .59)
        for key, val in original.items():
            self.assertEqual(row[key], val)
        self.assertEqual(merge([row], [original])[0][0], row)
        self.assertEqual(merge([original], [row])[0][0], row)

    def test_binary_resolution_preserves_pick_and_price(self):
        row = {"event": "SF @ LA (2026-09-10)", "pick": "HOME", "model_prob": .65,
               "market_prob": .64, "locked_at": "original", "odds": 1.5, "stake_units": 1}
        original = copy.deepcopy(row)
        S.apply_settlement(row, "AWAY", "verified")
        self.assertFalse(row["won"])
        self.assertEqual(row["pnl_units"], -1)
        for key, val in original.items():
            self.assertEqual(row[key], val)

    def test_dry_run_reports_matches_without_writing(self):
        row = {"id": 1, "event": "SF @ LA (2026-09-10)", "sport": "nfl", "pick": "HOME"}
        with patch.object(S, "load", return_value=[row]), patch.object(S, "save") as save, \
             patch.object(sys, "argv", ["settle.py", "--dry-run"]), self.nfl_api():
            self.assertEqual(S.main(), 0)
        save.assert_not_called()
        self.assertNotIn("outcome", row)

    def test_wnba_city_matching_unchanged(self):
        S._K_CACHE["KXWNBAGAME"] = {("2026-09-10", frozenset({"las vegas", "seattle"})): "seattle"}
        self.assertEqual(S.settle_via_kalshi({"event": "Seattle Storm @ Las Vegas Aces (2026-09-10)"}, "wnba")[0], "AWAY")


if __name__ == "__main__":
    unittest.main()
