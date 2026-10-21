import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import auto_update as A
import refresh_all as R


class SourceReceiptTests(unittest.TestCase):
    def refresh(self, failed=None):
        output = io.StringIO()
        def result(name):
            return name, name != failed, 'fixture source receipt'
        with patch.object(sys, 'argv', ['refresh-all']), patch.object(R, 'run_one', side_effect=result), \
             patch.object(R, 'inventory') as inventory, contextlib.redirect_stdout(output):
            status = R.main()
        inventory.assert_called_once()
        return status, output.getvalue()

    def test_all_seven_sources_must_succeed(self):
        status, output = self.refresh()
        self.assertEqual(status, 0)
        self.assertIn('7/7 succeeded', output)
        self.assertEqual(output.count('[OK ]'), 7)

    def test_any_failed_source_is_nonzero_and_other_sources_still_complete(self):
        for name in R.SOURCES:
            with self.subTest(source=name):
                status, output = self.refresh(name)
                self.assertEqual(status, 1)
                self.assertIn('6/7 succeeded', output)
                self.assertIn('FAILED: ' + name, output)

    def test_inventory_is_read_only_and_unknown_subset_is_not_success(self):
        for argv, expected in [(['--inventory'], 0), (['--only', 'not-a-source'], 1)]:
            with self.subTest(argv=argv), patch.object(sys, 'argv', ['refresh-all', *argv]), \
                 patch.object(R, 'inventory'), patch.object(R, 'run_one') as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(R.main(), expected)
                run.assert_not_called()

    def test_parent_retains_source_receipts_without_dumping_other_output(self):
        for code in [0, 1]:
            result = SimpleNamespace(returncode=code,
                stdout='unrelated output\n  [OK ] mlb 1s\n  [FAIL] nfl 1s\n total 2s 6/7 succeeded\n',
                stderr='source failure' if code else '')
            with patch.object(A.subprocess, 'run', return_value=result), patch.object(A, 'log') as log:
                self.assertEqual(A.run('refresh_all.py'), code == 0)
                text = '\n'.join(call.args[0] for call in log.call_args_list)
                self.assertIn('[OK ] mlb', text)
                self.assertIn('[FAIL] nfl', text)
                self.assertNotIn('unrelated output', text)

    def test_failed_source_refresh_stays_degraded_in_full_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'sports.json'
            target.write_text(json.dumps({'generated': '2026-10-21T00:00:00Z', 'open': [], 'settled': []}))
            args = SimpleNamespace(mode='full', dry_run=True, no_publish=True)
            with patch.object(A, 'PAYLOAD', target), patch.object(A, 'SELF_BOARD', target), \
                 patch.object(A, 'run', side_effect=lambda script, *a, **kw: script != 'refresh_all.py'), \
                 patch.object(A, 'log'):
                self.assertEqual(A._run(args), 1)
            refresh = json.loads(target.read_text())['refresh']
            self.assertEqual(refresh['status'], 'degraded')
            self.assertEqual(refresh['failed_steps'], ['refresh_all.py'])

    def test_full_cache_restore_does_not_replace_or_skip_full_chain(self):
        workflow = (Path(__file__).resolve().parents[1] / '.github/workflows/full.yml').read_text()
        self.assertLess(workflow.index('Restore historical download cache'), workflow.index('Run the full chain'))
        self.assertIn('python src/auto_update.py --mode full', workflow)
        self.assertIn('timeout-minutes: 30', workflow)


if __name__ == '__main__':
    unittest.main()
