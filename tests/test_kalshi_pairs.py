import re
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
import kalshi_pairs as KP
import guards as G


def reply(events, cursor=None):
    r = Mock(status_code=200)
    r.json.return_value = {'events': events, 'cursor': cursor}
    return r


class SplitPairTests(unittest.TestCase):
    def test_event_and_old_market_titles_pair(self):
        self.assertEqual(KP.split_pair('Game 3: Milwaukee vs Tampa Bay'), ('Milwaukee', 'Tampa Bay'))
        self.assertEqual(KP.split_pair('Manchester United vs Aston Villa'), ('Manchester United', 'Aston Villa'))
        self.assertEqual(KP.split_pair('Tampa Bay vs. Cleveland: Winner'), ('Tampa Bay', 'Cleveland'))
        self.assertEqual(KP.split_pair('Milwaukee vs Cincinnati Winner?'), ('Milwaukee', 'Cincinnati'))
        self.assertEqual(KP.split_pair('Boston vs New York Y (Game 2)'), ('Boston', 'New York Y'))

    def test_per_outcome_market_titles_are_not_pairs(self):
        for t in ('Tampa Bay wins', 'Tie is the result', '', None):
            self.assertIsNone(KP.split_pair(t))

    def test_pair_for_prefers_events_over_a_per_outcome_market_title(self):
        pairs = {'KXMLBGAME-26OCT282000MILTB': ('Milwaukee', 'Tampa Bay')}
        self.assertEqual(KP.pair_for('KXMLBGAME-26OCT282000MILTB', pairs, 'Tampa Bay wins'), ('Milwaukee', 'Tampa Bay'))
        self.assertIsNone(KP.pair_for('missing', pairs, 'Tampa Bay wins'))


class EventPairsTests(unittest.TestCase):
    def test_reads_every_page(self):
        pages = [reply([{'event_ticker': 'a', 'title': 'Game 1: A vs B'}], cursor='next'),
                 reply([{'event_ticker': 'b', 'title': 'C vs D'}])]
        with patch.object(KP.requests, 'get', side_effect=pages) as get, patch.object(KP.time, 'sleep'):
            self.assertEqual(KP.event_pairs('KXMLBGAME', 'settled'), {'a': ('A', 'B'), 'b': ('C', 'D')})
        self.assertEqual(get.call_args_list[1].kwargs['params']['cursor'], 'next')

    def test_a_new_format_fails_loudly_and_an_empty_series_stays_quiet(self):
        with patch.object(KP.requests, 'get', return_value=reply([{'event_ticker': 'a', 'title': 'Tampa Bay wins'}])):
            with self.assertRaises(G.ParseYieldedNothing) as caught:
                KP.event_pairs('KXMLBGAME', 'settled')
        self.assertIn('Tampa Bay wins', str(caught.exception))
        with patch.object(KP.requests, 'get', return_value=reply([])):
            self.assertEqual(KP.event_pairs('KXMLBGAME'), {})


class NoTitleSplittingTests(unittest.TestCase):
    """The fourth repeat of this bug came from a file the earlier fixes missed.
    Any new split of a title on " vs " outside kalshi_pairs.py fails here."""

    PATTERNS = [re.compile(r'''\.split\(\s*["']\s*vs\.?\s*["']'''),
                re.compile(r'''["']\s*vs\s*["']\s+(?:not\s+)?in\s''')]

    def test_only_kalshi_pairs_splits_on_vs(self):
        offenders = []
        for path in sorted(SRC.glob('*.py')):
            if path.name == 'kalshi_pairs.py':
                continue
            for n, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                code = line.split('#', 1)[0]
                if any(p.search(code) for p in self.PATTERNS):
                    offenders.append(f'{path.name}:{n}: {line.strip()}')
        self.assertEqual(offenders, [], 'Use kalshi_pairs.split_pair / event_pairs instead:\n' + '\n'.join(offenders))


if __name__ == '__main__':
    unittest.main()
