"""Verified one-record reconciliation; unverified doubleheader stays pending."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import open_lane as OL
import paper_settlement as PS
import settle as S
import export_tara as E
import totals_ledger as TL

AUDIT = json.loads((ROOT / 'data/reconciliations/open-1419-2026-10-15.json').read_text(encoding='utf8'))


class VerifiedPaperReconciliationTests(unittest.TestCase):
    def test_exact_verified_identity_resolves_loss_without_rewriting_capture(self):
        original = copy.deepcopy(AUDIT['original'])
        query = dict(original, game_id=AUDIT['verified_resolution']['game_id'])
        response = Mock(json=lambda:AUDIT['official_schedule'], raise_for_status=lambda:None)
        with patch.object(S.requests, 'get', return_value=response) as get:
            result = S.settle_mlb(query)
        self.assertEqual(get.call_args.kwargs['params']['gamePks'], 824705)
        self.assertEqual(result[0], 'AWAY')
        row=copy.deepcopy(original)
        row['settlement_evidence']=query['settlement_evidence']
        S.apply_settlement(row, *result)
        self.assertIs(row['won'], False)
        for key, value in original.items():
            if key not in {'outcome','won','settled_at'}:
                self.assertEqual(row[key], value, key)
        self.assertNotIn('game_id',row) # Verified later, never passed off as original capture.
        self.assertIsNone(row['ask'])
        self.assertNotIn('pnl_units',row)

    def test_saved_date_alone_cannot_select_a_makeup(self):
        row=copy.deepcopy(AUDIT['original'])
        with patch.object(S.requests,'get',return_value=Mock(json=lambda:{'dates':[]},raise_for_status=lambda:None)):
            self.assertIsNone(S.settle_mlb(row))
        self.assertTrue(PS.pending(row))
        self.assertEqual(row['settlement_status'],'fixture-unmatched')

    def test_canonical_reconciliation_preserves_original_and_saved_clock(self):
        current=PS.load(ROOT/'data/processed/open_ledger.jsonl')
        row=next(r for r in current if r['id']==1419)
        for key, value in AUDIT['original'].items():
            if key not in {'outcome','won','settled_at'}:
                self.assertEqual(row[key],value,key)
        self.assertEqual(row['outcome'],'AWAY')
        self.assertIs(row['won'],False)
        self.assertEqual(row['settlement_evidence']['game_id'],824705)
        self.assertEqual(row['settlement_reconciliation']['identity_sources'], AUDIT['reconciliation']['identity_sources'])
        # Independently retained capture-time board corroborates the saved clock.
        parent=json.loads(subprocess.check_output(['git','show',f"{AUDIT['original_capture_ref']}:public/sports.json"],cwd=ROOT,text=True))
        saved=[r for r in parent['upcoming'] if r['match']=='Chicago Cubs @ Boston Red Sox' and r['start'].startswith('2026-09-26')]
        self.assertEqual(len(saved),1)
        self.assertEqual(saved[0]['start'],AUDIT['reconciliation']['historical_board_start_utc'])

    def test_repeat_settlement_preserves_resolution_and_does_not_guess_other_record(self):
        rows=PS.load(ROOT/'data/processed/open_ledger.jsonl')
        unverified=copy.deepcopy(AUDIT['original'])
        unverified.update(id=1418,event='Baltimore Orioles @ New York Yankees (2026-09-26)',model_prob=0.5779358322962183)
        PS.block(unverified,'fixture-unmatched','No unique official MLB fixture matches the saved identity; reconciliation required',{'source':'mlb','saved_date':'2026-09-26','game_id':None})
        pair=[unverified,copy.deepcopy(next(r for r in rows if r['id']==1419))]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'paper.jsonl'; totals=Path(directory)/'totals.jsonl'
            PS.atomic_save(path,pair); PS.atomic_save(totals,[])
            before=path.read_bytes()
            with patch.object(OL,'LEDGER',path),patch.object(TL,'LEDGER',totals),patch.object(S.requests,'get',return_value=Mock(json=lambda:{'dates':[]},raise_for_status=lambda:None)):
                self.assertEqual(OL.settle(),0)
                self.assertEqual(path.read_bytes(),before)
                health=E.paper_settlement_state()['open']
            self.assertEqual(health['binary_settled'],1)
            self.assertEqual(health['past_date_pending'],1)
            self.assertEqual(health['blockers'],{'fixture-unmatched':1})

    def test_published_paper_summary_matches_canonical_not_headline(self):
        rows=PS.load(ROOT/'data/processed/open_ledger.jsonl')
        board=json.loads((ROOT/'public/sports.json').read_text(encoding='utf8'))
        actual=PS.health(rows)
        published=board['paper_settlement']['open']
        for field in ('recorded','binary_settled','non_binary_resolved','cancelled_resolved','pending'):
            self.assertEqual(published[field],actual[field],field)
        binary=[r for r in rows if r.get('won') is not None and not r.get('voided')]
        self.assertEqual(board['open_lane']['n_settled'],len(binary))
        self.assertEqual(board['open_lane']['wins'],sum(r['won'] for r in binary))
        self.assertEqual(board['open_lane']['losses'],sum(not r['won'] for r in binary))
        exact_be=sum(r['ask']+.07*r['ask']*(1-r['ask']) for r in binary if r.get('ask') is not None)/sum(r.get('ask') is not None for r in binary)
        exact_rate=sum(r['won'] for r in binary)/len(binary)
        self.assertEqual(board['open_lane']['win_rate'],round(exact_rate,4))
        self.assertEqual(board['open_lane']['edge'],round(exact_rate-exact_be,4))
        for summary in board['open_lane']['by_kind']:
            group=[r for r in binary if f"{r.get('market')}·{r.get('kind')}"==summary['kind']]
            self.assertTrue(group)
            self.assertEqual(summary['n'],len(group))
            self.assertEqual(summary['wins'],sum(r['won'] for r in group))
            self.assertEqual(summary['win_rate'],round(sum(r['won'] for r in group)/len(group),4))
        headline=PS.load(ROOT/'data/processed/ledger.jsonl')
        headline_done=[r for r in headline if r.get('won') is not None and not r.get('voided')]
        self.assertEqual(board['record']['wins'],sum(r['won'] for r in headline_done))
        self.assertEqual(board['record']['losses'],sum(not r['won'] for r in headline_done))


if __name__=='__main__':
    unittest.main()
