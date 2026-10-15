import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import auto_update as A


class RefreshTests(unittest.TestCase):
    def test_failed_publish_raises_instead_of_silently_succeeding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sports.json').write_text('{}')
            for failing in range(4):
                responses = [(0, 'changed'), (0, ''), (0, ''), (0, '')]
                responses[failing] = (1, 'simulated failure')
                with patch.object(A, 'git', side_effect=responses):
                    with self.assertRaises(A.PublishError):
                        A._commit_and_push(root, ['sports.json'], 'test', False, 'sports')

    def test_no_change_and_dry_run_do_not_make_a_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sports.json').write_text('{}')
            for result, dry in [((0, ''), False), ((0, 'changed'), True)]:
                with patch.object(A, 'git', return_value=result) as git:
                    self.assertFalse(A._commit_and_push(root, ['sports.json'], 'test', dry, 'sports'))
                    self.assertEqual(git.call_count, 1)

    def test_real_chain_stamps_honest_health_preserves_evidence_and_reports_source_failure(self):
        for failed in [None, 'settle.py', 'mlb_predict.py']:
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'sports.json'
                original = {'generated': '2026-10-15T18:00:00Z', 'open': [{'id': 7}],
                            'settled': [{'id': 1, 'outcome': 'HOME'}]}
                target.write_text(json.dumps(original))
                args = argparse.Namespace(mode='light', dry_run=True, no_publish=True)
                with patch.object(A, 'PAYLOAD', target), patch.object(A, 'SELF_BOARD', target), \
                     patch.object(A, 'run', side_effect=lambda script, *args, **kw: script != failed), \
                     patch.object(A, 'log'):
                    result = A._run(args)
                saved = json.loads(target.read_text())
                self.assertEqual(saved['generated'], original['generated'])
                self.assertEqual(saved['open'], original['open'])
                self.assertEqual(saved['settled'], original['settled'])
                self.assertEqual(saved['refresh']['status'], 'degraded' if failed else 'ok')
                self.assertEqual(saved['refresh']['failed_steps'], [failed] if failed else [])
                self.assertEqual(result, 1 if failed else 0)


if __name__ == '__main__':
    unittest.main()
