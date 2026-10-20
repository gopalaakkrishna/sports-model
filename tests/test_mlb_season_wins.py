import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import mlb_season_wins as S


class SeasonWindowTests(unittest.TestCase):
    def calendar(self, start='2026-03-25', end='2026-09-27', season='2026'):
        response = Mock()
        response.json.return_value = {'seasons': [{
            'seasonId': season, 'regularSeasonStartDate': start,
            'regularSeasonEndDate': end,
        }]}
        return response

    def test_offseason_and_preseason_are_noops_using_official_calendar(self):
        for day in [date(2026, 10, 20), date(2026, 1, 1), date(2027, 1, 1)]:
            with self.subTest(day=day), patch.object(S.requests, 'get', return_value=self.calendar()) as get:
                self.assertIsNone(S.regular_season_window(2026, day))
                self.assertEqual(get.call_count, 1)
                self.assertEqual(get.call_args.args[0], S.STATS + '/seasons/2026')

    def test_in_season_and_both_boundaries_use_real_end_not_october_fifth(self):
        for day in [date(2026, 3, 25), date(2026, 6, 1), date(2026, 9, 27)]:
            with self.subTest(day=day), patch.object(S.requests, 'get', return_value=self.calendar()):
                self.assertEqual(S.regular_season_window(2026, day), (day.isoformat(), '2026-09-27'))

    def test_later_official_regular_season_is_not_silently_cut_off(self):
        with patch.object(S.requests, 'get', return_value=self.calendar(end='2026-10-08')):
            self.assertEqual(S.regular_season_window(2026, date(2026, 10, 6)),
                             ('2026-10-06', '2026-10-08'))

    def test_missing_wrong_ambiguous_or_invalid_calendar_remains_failure(self):
        responses = [self.calendar(start='bad'), self.calendar(end='2026-03-01'),
                     self.calendar(season='2025'), self.calendar(end='2027-09-27')]
        empty = Mock()
        empty.json.return_value = {'seasons': []}
        duplicate = Mock()
        duplicate.json.return_value = {'seasons': self.calendar().json()['seasons'] * 2}
        missing = Mock()
        missing.json.return_value = {'seasons': [{'seasonId': '2026'}]}
        for response in responses + [empty, duplicate, missing]:
            with self.subTest(response=response), patch.object(S.requests, 'get', return_value=response):
                with self.assertRaises(ValueError):
                    S.regular_season_window(2026, date(2026, 6, 1))

    def test_calendar_http_failure_is_not_reported_as_offseason(self):
        response = self.calendar()
        response.raise_for_status.side_effect = requests.HTTPError('upstream unavailable')
        with patch.object(S.requests, 'get', return_value=response), self.assertRaises(requests.HTTPError):
            S.regular_season_window(2026, date(2026, 10, 20))

    def test_reversed_range_never_requests_schedule(self):
        with patch.object(S.requests, 'get') as get:
            self.assertEqual(S.remaining('2026-10-20', '2026-10-05'), [])
            get.assert_not_called()

    def test_last_day_requests_regular_games_and_excludes_postseason_and_final(self):
        game = {'gameType': 'R', 'status': {'detailedState': 'Scheduled'},
                'teams': {'home': {'team': {'name': 'Home', 'id': 1}},
                          'away': {'team': {'name': 'Away', 'id': 2}}}}
        response = Mock()
        response.json.return_value = {'dates': [{'games': [game,
            {**game, 'gameType': 'D'}, {**game, 'status': {'detailedState': 'Final'}}]}]}
        with patch.object(S.requests, 'get', return_value=response) as get:
            self.assertEqual(len(S.remaining('2026-09-27', '2026-09-27')), 1)
            self.assertEqual(get.call_args.kwargs['params']['gameTypes'], 'R')

    def test_offseason_main_never_fits_prices_logs_or_touches_archives(self):
        with patch.object(sys, 'argv', ['season-wins', '--season', '2026', '--log']), \
             patch.object(S, 'regular_season_window', return_value=None), \
             patch.object(S, 'remaining') as remaining, patch.object(S, 'standings') as standings, \
             patch.object(S.MM, 'fit') as fit, patch.object(S.pd, 'read_parquet') as parquet, \
             patch.object(S, 'kalshi_wins_markets') as markets, patch.object(S.pd.DataFrame, 'to_csv') as csv:
            self.assertEqual(S.main(), 0)
            for method in [remaining, standings, fit, parquet, markets, csv]:
                method.assert_not_called()

    def test_in_season_empty_schedule_never_generates_a_projection(self):
        with patch.object(sys, 'argv', ['season-wins']), \
             patch.object(S, 'regular_season_window', return_value=('2026-09-27', '2026-09-27')), \
             patch.object(S, 'remaining', return_value=[]), patch.object(S.MM, 'fit') as fit:
            self.assertEqual(S.main(), 0)
            fit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
