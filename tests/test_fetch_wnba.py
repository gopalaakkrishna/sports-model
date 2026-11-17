import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import fetch_nfl_wnba as F

PAGE = """<table><thead><tr><th>Date</th><th>Visitor/Neutral</th><th>PTS</th>
<th>Home/Neutral</th><th>PTS</th></tr></thead><tbody>
<tr><td>Sat, Aug 16, 2026</td><td>Atlanta Dream</td><td>81</td><td>Golden State Valkyries</td><td>77</td></tr>
<tr><td>Fri, Oct 30, 2026</td><td>Golden State Valkyries</td><td>70</td><td>Atlanta Dream</td><td>75</td></tr>
</tbody></table>"""


def page(status=200, text=PAGE):
    return Mock(status_code=status, text=text)


class FetchWnbaTests(unittest.TestCase):
    def test_the_parser_dependency_is_installed(self):
        # Without lxml every season raised ImportError for three months.
        import lxml  # noqa: F401

    def test_a_failed_season_keeps_the_file_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as d, patch.object(F, 'RAW', Path(d)), \
             patch.object(F.time, 'sleep'), patch.object(F.requests, 'get', return_value=page()), \
             patch.object(F.pd, 'read_html', side_effect=ImportError('lxml not found')):
            self.assertIsNone(F.fetch_wnba(2025, 2026))
            self.assertFalse((Path(d) / 'wnba_games.parquet').exists())

    def test_every_season_parses_and_a_missing_current_page_is_fine(self):
        replies = [page(), page(status=404)]
        with tempfile.TemporaryDirectory() as d, patch.object(F, 'RAW', Path(d)), \
             patch.object(F.time, 'sleep'), patch.object(F.requests, 'get', side_effect=replies):
            df = F.fetch_wnba(2026, 2027)
        self.assertEqual(len(df), 2)
        self.assertEqual(df['played'].sum(), 2)
        self.assertEqual(str(df['date'].max().date()), '2026-10-30')

    def test_an_old_season_returning_an_error_is_a_failure(self):
        with tempfile.TemporaryDirectory() as d, patch.object(F, 'RAW', Path(d)), \
             patch.object(F.time, 'sleep'), patch.object(F.requests, 'get', side_effect=[page(status=429), page()]):
            self.assertIsNone(F.fetch_wnba(2025, 2026))


if __name__ == '__main__':
    unittest.main()
