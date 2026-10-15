"""Archived date identity, not doubleheader order, selects the original game."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import open_lane as OL
import paper_settlement as PS
import settle as S
AUDIT=json.loads((ROOT/'data/reconciliations/open-1418-2026-10-15.json').read_text(encoding='utf8'))


class ArchivedPaperIdentityTests(unittest.TestCase):
    def test_unique_original_date_identity_not_adjacent_game_or_play_order(self):
        original=AUDIT['original']
        history=AUDIT['original_schedule_snapshot']['orioles_at_yankees_september']
        matches=[r for r in history if r['date'][:10]==original['start'] and r['home_team']=='New York Yankees' and r['away_team']=='Baltimore Orioles']
        self.assertEqual(len(matches),1)
        self.assertEqual(matches[0]['gamePk'],823489)
        self.assertEqual(matches[0]['status'],'Scheduled')
        self.assertIsNone(matches[0]['home_runs'])
        self.assertEqual({r['date'][:10]:r['gamePk'] for r in history},{'2026-09-25':823491,'2026-09-26':823489,'2026-09-27':823490})
        self.assertTrue(AUDIT['archive_provenance']['public_archive_identical'])
        self.assertEqual(AUDIT['archive_provenance']['archive_hash'],AUDIT['original_schedule_snapshot']['sha256'])

    def test_current_result_must_match_archived_stable_id(self):
        original=copy.deepcopy(AUDIT['original'])
        query=dict(original,game_id=823489)
        with patch.object(S.requests,'get',return_value=Mock(json=lambda:AUDIT['official_schedule'],raise_for_status=lambda:None)) as get:
            result=S.settle_mlb(query)
        self.assertEqual(get.call_args.kwargs['params']['gamePks'],823489)
        self.assertEqual(result[0],'HOME')
        wrong=copy.deepcopy(AUDIT['official_schedule'])
        wrong['dates'][0]['games'][0]['gamePk']=823491
        with patch.object(S.requests,'get',return_value=Mock(json=lambda:wrong,raise_for_status=lambda:None)):
            self.assertIsNone(S.settle_mlb(dict(original,game_id=823489)))

    def test_reconciled_record_keeps_original_capture_and_missing_price(self):
        row=next(r for r in PS.load(ROOT/'data/processed/open_ledger.jsonl') if r['id']==1418)
        for k,v in AUDIT['original'].items():
            if k not in {'outcome','won','settled_at'}:self.assertEqual(row[k],v,k)
        self.assertEqual(row['outcome'],'HOME')
        self.assertIs(row['won'],True)
        self.assertEqual(row['settlement_evidence']['game_id'],823489)
        self.assertEqual(row['settlement_reconciliation']['archive_sha256'],AUDIT['archive_provenance']['archive_hash'])
        self.assertNotIn('game_id',row) # Recovered later; not claimed captured originally.
        self.assertIsNone(row['ask'])
        self.assertNotIn('pnl_units',row)

    def test_replay_is_idempotent_and_never_fetches_a_replacement_result(self):
        row=next(r for r in PS.load(ROOT/'data/processed/open_ledger.jsonl') if r['id']==1418)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'paper.jsonl'; PS.atomic_save(path,[row]); before=path.read_bytes()
            with patch.object(OL,'LEDGER',path),patch.object(S.requests,'get') as get:
                self.assertEqual(OL.settle(),0)
                self.assertEqual(path.read_bytes(),before)
                get.assert_not_called()
        board=json.loads((ROOT/'public/sports.json').read_text(encoding='utf8'))
        self.assertNotIn(1418,[r['id'] for r in board['paper_settlement']['open']['attention']])


if __name__=='__main__':unittest.main()
