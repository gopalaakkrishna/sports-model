import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import mlb_predict as M
import guards as G


class EventPairTests(unittest.TestCase):
    def pairs(self, events):
        response = Mock(status_code=200)
        response.json.return_value = {'events': events}
        with patch.object(M.requests, 'get', return_value=response):
            return M._mlb_event_pairs()

    def test_postseason_prefix_preserves_teams_and_each_game_identity(self):
        events = [
            {'event_ticker': 'KXMLBGAME-26OCT151840TBACLE',
             'title': 'Game 3: Tampa Bay vs Cleveland'},
            {'event_ticker': 'KXMLBGAME-26OCT161840TBACLE',
             'title': 'Game 4: Tampa Bay vs Cleveland'},
            {'event_ticker': 'KXMLBGAME-26OCT152000MILLAD',
             'title': 'Game 4: Milwaukee vs Los Angeles D'},
        ]
        pairs = self.pairs(events)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[events[0]['event_ticker']], ('Tampa Bay', 'Cleveland'))
        self.assertEqual(pairs[events[2]['event_ticker']], ('Milwaukee', 'Los Angeles D'))
        self.assertEqual(M._ticker_date(events[0]['event_ticker']), '2026-10-15')
        self.assertEqual(M._ticker_date(events[1]['event_ticker']), '2026-10-16')

    def test_regular_title_and_optional_descriptor_still_parse(self):
        self.assertEqual(self.pairs([
            {'event_ticker': 'regular', 'title': 'Milwaukee vs Cincinnati'},
            {'event_ticker': 'suffix', 'title': 'Tampa Bay vs. Cleveland: Winner'},
        ]), {'regular': ('Milwaukee', 'Cincinnati'), 'suffix': ('Tampa Bay', 'Cleveland')})

    def test_unknown_nonempty_format_still_fails_loudly(self):
        with self.assertRaises(G.ParseYieldedNothing):
            self.pairs([{'event_ticker': 'bad', 'title': 'League championship'}])
        self.assertEqual(self.pairs([]), {})


if __name__ == '__main__':
    unittest.main()
