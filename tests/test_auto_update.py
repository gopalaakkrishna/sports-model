import argparse
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import auto_update as A


class RefreshTests(unittest.TestCase):
    def test_regenerated_report_must_verify_before_any_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'sports.json'
            target.write_text(json.dumps({'generated':'2026-10-15T18:00:00Z'}))
            args = argparse.Namespace(mode='light', dry_run=False, no_publish=False)
            with patch.object(A, 'PAYLOAD', target), patch.object(A, 'SELF_BOARD', target), \
                 patch.object(A, 'run', side_effect=lambda script, *args, **kw: script != 'verify_paper_reports.py') as run, \
                 patch.object(A, 'publish_self_state') as publish_state, patch.object(A, 'publish') as publish, \
                 patch.object(A, 'log'):
                self.assertEqual(A._run(args), 1)
                publish_state.assert_not_called()
                publish.assert_not_called()
                self.assertIn('verify_paper_reports.py', [call.args[0] for call in run.call_args_list])

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
        for failed in [None, 'settle.py', 'mlb_predict.py', 'open_lane.py', 'totals_ledger.py']:
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
                self.assertEqual(saved['refresh']['pipeline_version'], '14.7.7')
                # A first failure is disclosed on the board but does not fail
                # the job (no email); see test_sustained_failure_alerts_once.
                self.assertEqual(result, 0)
                self.assertEqual(list(saved['refresh']['incidents']), [failed] if failed else [])
                self.assertEqual(saved['refresh']['alerted_steps'], [])

    def _chain(self, target, failed, now, mode='light'):
        args = argparse.Namespace(mode=mode, dry_run=True, no_publish=True)
        clock = unittest.mock.Mock(wraps=A.datetime)
        clock.now = lambda tz=None: now
        with patch.object(A, 'PAYLOAD', target), patch.object(A, 'SELF_BOARD', target), \
             patch.object(A, 'datetime', clock), \
             patch.object(A, 'run', side_effect=lambda script, *a, **kw: script not in failed), \
             patch.object(A, 'log'):
            return A._run(args), json.loads(target.read_text())['refresh']

    def test_sustained_failure_alerts_once_then_quiets_then_realerts(self):
        # The 2026-10-22 incident: open_lane.py failed on every 5-minute run
        # for 45 minutes and each run emailed. Now: silent under an hour, one
        # alert at the hour, quiet for six hours, then one reminder.
        start = datetime(2026, 10, 22, 7, 11, tzinfo=timezone.utc)
        minutes = lambda m: start + timedelta(minutes=m)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'sports.json'
            target.write_text(json.dumps({'generated': '2026-10-22T07:00:00Z'}))
            results = {}
            for m in [0, 5, 45, 59, 60, 65, 120, 359, 420, 425]:
                code, refresh = self._chain(target, {'open_lane.py'}, minutes(m))
                results[m] = code
                self.assertEqual(refresh['status'], 'degraded')
                self.assertEqual(refresh['incidents']['open_lane.py']['since'], start.isoformat())
            self.assertEqual(results, {0: 0, 5: 0, 45: 0, 59: 0, 60: 1, 65: 0, 120: 0, 359: 0, 420: 1, 425: 0})
            # Recovery closes the incident; a new failure starts a new clock.
            code, refresh = self._chain(target, set(), minutes(430))
            self.assertEqual((code, refresh['incidents'], refresh['status']), (0, {}, 'ok'))
            code, refresh = self._chain(target, {'open_lane.py'}, minutes(500))
            self.assertEqual(code, 0)
            self.assertEqual(refresh['incidents']['open_lane.py']['since'], minutes(500).isoformat())

    def test_full_refresh_source_failure_still_fails_immediately(self):
        # The scheduler's bounded full-refresh retry only fires on a failed job.
        now = datetime(2026, 10, 22, 11, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'sports.json'
            target.write_text(json.dumps({'generated': '2026-10-22T11:00:00Z'}))
            code, refresh = self._chain(target, {'refresh_all.py'}, now, mode='full')
            self.assertEqual((code, refresh['alerted_steps']), (1, ['refresh_all.py']))

    def test_light_run_keeps_but_never_alerts_on_full_only_incident(self):
        prior = {'refresh_all.py': {'since': '2026-10-21T11:30:00+00:00', 'alerted': None}}
        now = datetime(2026, 10, 22, 11, 30, tzinfo=timezone.utc)
        incidents, alert = A.track_incidents(prior, {'settle.py', 'open_lane.py'}, set(), now)
        self.assertEqual((incidents, alert), (prior, []))

    def test_unreadable_prior_incident_state_starts_a_fresh_clock(self):
        now = datetime(2026, 10, 22, 8, 0, tzinfo=timezone.utc)
        for prior in [{'open_lane.py': {'since': 'garbage'}}, {'open_lane.py': {}}]:
            incidents, alert = A.track_incidents(prior, {'open_lane.py'}, {'open_lane.py'}, now)
            self.assertEqual((incidents['open_lane.py']['since'], alert), (now.isoformat(), []))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'sports.json'
            for text in ['not json', '[]', '{"refresh": 5}', '{"refresh": {"incidents": [1]}}']:
                target.write_text(text)
                self.assertEqual(A.prior_incidents(target), {})


if __name__ == '__main__':
    unittest.main()
