import copy
import json
import sys
import tempfile
import unittest
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import settle as S
import open_lane as OL
import totals_ledger as TL
import paper_settlement as PS
import export_tara as E

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/paper_legacy_2026_10_15.json').read_text())


class PaperSettlementTests(unittest.TestCase):
    def setUp(self):
        for cache in (S._K_CACHE, S._SCALAR_RESULTS, S._EVIDENCE, S._MARKET_CACHE):
            cache.clear()
        S._PARSE_FAILURES.clear()

    def api(self, url, **kw):
        params = kw.get('params') or {}
        body = {}
        if '/schedule' in url:
            if params.get('gamePks'):
                return Mock(status_code=200, json=lambda: FIXTURE['rescheduled_mlb'], raise_for_status=lambda:None)
            date = params.get('startDate')
            games = []
            for item in FIXTURE['mlb']:
                if item['event'].endswith(f'({date})'):
                    for g in item['exactMatches']:
                        games.append({'gamePk': g['gamePk'], 'gameDate': g['gameDate'],
                                      'status': g['status'], 'rescheduleDate': g.get('rescheduleDate'),
                                      'teams': {'home': {'team': {'name': g['home']}},
                                                'away': {'team': {'name': g['away']}}}})
            body = {'dates': [{'games': games}]}
        elif url.endswith('/events'):
            group = next(g for g in FIXTURE['events'] if g['series'] == params['series_ticker'])
            body = {'events': group['matches']}
        elif url.endswith('/markets'):
            group = next(g for g in FIXTURE['events'] if g['series'] == params['series_ticker'])
            body = {'markets': [m for e in group['matches'] for m in e['markets']]}
        else:
            market = next(x['market'] for x in FIXTURE['totals'] if url.endswith('/' + x['market']['ticker']))
            body = {'market': market}
        return Mock(status_code=200, json=lambda: body, raise_for_status=lambda: None)

    def test_real_23_row_capture_through_settle_save_reload_export_and_repeat(self):
        original = copy.deepcopy(FIXTURE['rows'])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'open.jsonl'
            totals = Path(d) / 'totals.jsonl'
            PS.atomic_save(path, original)
            PS.atomic_save(totals, [])
            with patch.object(OL, 'LEDGER', path), patch.object(TL, 'LEDGER', totals), \
                 patch.object(S.requests, 'get', side_effect=self.api), patch.object(S.D, 'load_history') as history:
                self.assertEqual(OL.settle(), 21)
                rows = OL._load()
                self.assertEqual(sum(r.get('outcome') == 'SCALAR' for r in rows), 19)
                self.assertEqual(sum(r.get('outcome') == 'CANCELLED' for r in rows), 1)
                self.assertEqual({r['id'] for r in rows if PS.pending(r)}, {1418, 1419})
                self.assertTrue(next(r for r in rows if r['id']==1304)['won'])
                self.assertTrue(all(r.get('won') is None for r in rows if r['id']!=1304))
                history.assert_not_called()  # no archive result may override a final scalar payout
                for before, after in zip(original, rows):
                    for k, v in before.items():
                        if k not in {'outcome', 'won', 'settled_at', 'voided'}:
                            self.assertEqual(after[k], v, (before['id'], k))
                saved = path.read_bytes()
                self.assertEqual(OL.settle(), 0)
                self.assertEqual(path.read_bytes(), saved)
                summary = E.paper_settlement_state()['open']
                self.assertEqual(summary['binary_settled'], 1)
                self.assertEqual(summary['non_binary_resolved'], 19)
                self.assertEqual(summary['cancelled_resolved'], 1)
                self.assertEqual(summary['past_date_pending'], 2)
                self.assertEqual(summary['blockers'], {'fixture-unmatched': 2})
                self.assertEqual(summary['status'], 'attention-required')

    def test_under_scalar_receives_no_side_payout_not_yes_payout(self):
        m = {'ticker': 'T', 'status': 'finalized', 'result': 'scalar', 'settlement_value_dollars': '.61'}
        result = PS.market_result(m, 'OVER', 'UNDER', selected_no=True)
        self.assertEqual(result[0], 'SCALAR')
        self.assertEqual(result[2]['payout_dollars'], .39)

    def test_bad_scalar_and_open_markets_are_not_resolved(self):
        for p in ('nan', 'inf', '-.1', '1.1', None):
            self.assertIsNone(PS.market_result({'status': 'finalized', 'result': 'scalar', 'settlement_value_dollars': p}, 'H', 'A'))
        self.assertIsNone(PS.market_result({'status': 'active', 'result': 'yes'}, 'H', 'A'))

    def test_wrong_exact_ticker_or_event_is_not_accepted(self):
        for market in ({'ticker': 'OTHER'}, {'ticker': 'T', 'event_ticker': 'WRONG'}):
            S._MARKET_CACHE.clear()
            row = {'event_ticker': 'EXPECTED'}
            with patch.object(S.requests, 'get', return_value=Mock(status_code=200, json=lambda: {'market': market}, raise_for_status=lambda: None)):
                self.assertIsNone(S.settle_exact_market(row, 'T', 'HOME', 'AWAY'))
            self.assertEqual(row['settlement_status'], 'market-identity-mismatch')

    def test_scalar_date_boundary_is_unique_not_unrestricted_fuzzy_match(self):
        pair = frozenset({'cincinnati', 'dc united'})
        S._K_CACHE['soccer:KXMLSGAME'] = {}
        for day in ('2026-09-05', '2026-09-06'):
            S._SCALAR_RESULTS[(day, pair)] = {'cincinnati': {'payout_dollars': .59, 'event_ticker': 'KXMLSGAME-' + day}}
        row = {'event': 'FC Cincinnati v DC United (2026-09-06)', 'league': 'USA MLS', 'pick': 'HOME'}
        self.assertIsNone(S.settle_soccer_via_kalshi(row))
        self.assertEqual(row['settlement_status'], 'fixture-ambiguous')
        row['event_ticker'] = 'KXMLSGAME-2026-09-05'
        self.assertEqual(S.settle_soccer_via_kalshi(row)[0], 'SCALAR')

    def test_totals_lane_scalar_binary_and_retry_do_not_touch_captures(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'totals.jsonl'
            rows = [{'id': 'T', 'start': '2026-09-05T00:00:00Z', 'market_prob': .6, 'outcome': None, 'won': None, 'locked_at': 'original'}]
            PS.atomic_save(path, rows)
            m = {'ticker': 'T', 'status': 'finalized', 'result': 'scalar', 'settlement_value_dollars': '.61'}
            with patch.object(TL, 'LEDGER', path), patch.object(S.requests, 'get', return_value=Mock(status_code=200,json=lambda:{'market':m},raise_for_status=lambda:None)):
                self.assertEqual(TL.settle(), 1)
                first = path.read_bytes()
                self.assertEqual(TL.settle(), 0)
                self.assertEqual(path.read_bytes(), first)
                self.assertEqual(TL.report()['n_settled'], 0)
                self.assertEqual(TL._load()[0]['locked_at'], 'original')

    def test_future_records_are_not_queried_or_false_backlog(self):
        row = {'id': 1, 'start': '2099-01-01', 'event': 'A v B (2099-01-01)', 'won': None, 'outcome': None}
        with patch.object(OL, '_load', return_value=[row]), patch.object(OL, '_save') as save, patch.object(S.requests, 'get') as get:
            self.assertEqual(OL.settle(), 0)
        get.assert_not_called()
        save.assert_not_called()
        h = PS.health([row], datetime(2026, 10, 15, tzinfo=timezone.utc))
        self.assertEqual(h['future_pending'], 1)
        self.assertEqual(h['past_date_pending'], 0)

    def test_transport_failure_is_persisted_and_not_ci_success(self):
        row = {'id': 1, 'start': '2026-09-01', 'event': 'A v B (2026-09-01)', 'market': 'TOTAL', 'pick': 'OVER', 'ticker': 'T'}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'paper.jsonl'
            PS.atomic_save(path, [row])
            with patch.object(OL, 'LEDGER', path), patch.object(S.requests, 'get', side_effect=S.requests.Timeout('offline')), patch.object(sys, 'argv', ['open_lane.py', '--settle']):
                self.assertEqual(OL.main(), 1)
                self.assertEqual(OL._load()[0]['settlement_status'], 'source-error')

    def test_doubleheader_is_not_resolved_using_first_matching_game(self):
        g = {'gamePk': 1, 'status': {'detailedState': 'Final'}, 'teams': {'home': {'team': {'name': 'Home'}, 'score': 2}, 'away': {'team': {'name': 'Away'}, 'score': 1}}}
        g2 = copy.deepcopy(g); g2['gamePk'] = 2
        body = {'dates': [{'games': [g, g2]}]}
        row = {'event': 'Away @ Home (2026-09-01)'}
        with patch.object(S.requests, 'get', return_value=Mock(json=lambda:body,raise_for_status=lambda:None)):
            self.assertIsNone(S.settle_mlb(row))
            row['game_id'] = 2
            self.assertEqual(S.settle_mlb(row)[0], 'HOME')
            self.assertEqual(row['settlement_evidence']['game_id'], 2)

    def test_atomic_write_failure_preserves_previous_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'paper.jsonl'; PS.atomic_save(p, [{'id': 1}])
            before = p.read_bytes()
            with patch.object(PS.os, 'replace', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError): PS.atomic_save(p, [{'id': 2}])
            self.assertEqual(p.read_bytes(), before)
            self.assertEqual(list(Path(d).glob('*.tmp')), [])

    def test_corruption_or_duplicate_ids_never_silently_drop_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rows.jsonl'; PS.atomic_save(p,[{'id':1}]); before=p.read_bytes()
            with self.assertRaises(ValueError): PS.atomic_save(p,[{'id':1},{'id':'1'}])
            self.assertEqual(p.read_bytes(),before)
            with patch.object(Path,'read_text',return_value='{"id":1}\nBROKEN\n'):
                with self.assertRaises(ValueError): PS.load(p)

    def test_soccer_selected_leg_no_does_not_invent_other_team_as_winner(self):
        row={'id':1,'event':'A v B (2026-09-01)','start':'2026-09-01','market':'1X2','pick':'HOME','ticker':'T'}
        m={'ticker':'T','result':'no','status':'finalized'}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'paper.jsonl';PS.atomic_save(p,[row])
            with patch.object(OL,'LEDGER',p),patch.object(S.requests,'get',return_value=Mock(status_code=200,json=lambda:{'market':m},raise_for_status=lambda:None)):
                OL.settle()
                saved=OL._load()[0]
                self.assertEqual(saved['outcome'],'NOT_HOME');self.assertFalse(saved['won'])

    def test_new_captures_retain_market_event_and_mlb_game_identity(self):
        soccer=pd.DataFrame([{'event':'KXMLSGAME-26OCT20AB','ticker':'KXMLSGAME-26OCT20AB-A',
            'series':'KXMLSGAME','match':'A v B','when':'2026-10-21T01:00','leg':'HOME','model':.7,'ask':.65}])
        with patch.object(OL,'_latest',return_value=Path('report.csv')),patch.object(OL.pd,'read_csv',return_value=soccer):
            rows=OL._rows_from_soccer_1x2()
        self.assertEqual(rows[0]['ticker'],'KXMLSGAME-26OCT20AB-A')
        self.assertEqual(rows[0]['event_ticker'],'KXMLSGAME-26OCT20AB')
        baseball=pd.DataFrame([{'date':'2026-10-20','match':'Away @ Home','model':.7,'side':'HOME',
            'game_id':123,'start_utc':'2026-10-21T01:00:00Z','ticker':'KXMLBGAME-26OCT20AH-H','event_ticker':'KXMLBGAME-26OCT20AH'}])
        with patch.object(OL,'_latest',return_value=Path('report.csv')),patch.object(OL.pd,'read_csv',return_value=baseball):
            rows=OL._rows_from_two_sided('mlb_predictions','baseball','MLB')
        self.assertEqual(rows[0]['game_id'],123);self.assertEqual(rows[0]['start'],'2026-10-21T01:00:00Z')
        self.assertEqual(rows[0]['model_prob'],.7)

    def test_settlement_only_export_updates_paper_health_without_repricing(self):
        old={'generated':'2026-10-01T10:00:00Z','board':[], 'open':[], 'settled':[],
             'totals':{'picks':[{'ask':.61}],'paper_record':{'n':1}},'open_lane':{'n_settled':1}}
        fresh={'generated':'2026-10-15T20:00:00Z','open':[],'settled':[],'disclosures':[],
               'totals':{'picks':[{'ask':.99}],'paper_record':{'n':2}},'open_lane':{'n_settled':2},
               'paper_settlement':{'version':'14.7.5','open':{'past_date_pending':2}}}
        with patch.object(E,'build',return_value=fresh): result=E.refresh_settlements(old)
        self.assertEqual(result['generated'],old['generated'])
        self.assertEqual(result['totals']['picks'],old['totals']['picks'])
        self.assertEqual(result['totals']['paper_record'],{'n':2})
        self.assertEqual(result['open_lane'],fresh['open_lane'])
        self.assertEqual(result['paper_settlement'],fresh['paper_settlement'])

    def test_postponed_without_verified_makeup_remains_pending(self):
        g={'gamePk':123,'gameDate':'2026-09-01T22:00:00Z','status':{'detailedState':'Postponed','reason':'Rain'},
           'teams':{'home':{'team':{'name':'Home'}},'away':{'team':{'name':'Away'}}}}
        row={'event':'Away @ Home (2026-09-01)'}
        with patch.object(S.requests,'get',return_value=Mock(json=lambda:{'dates':[{'games':[g]}]},raise_for_status=lambda:None)):
            self.assertIsNone(S.settle_mlb(row))
        self.assertEqual(row['settlement_status'],'postponed')
        self.assertNotIn('outcome',row)

    def test_doubleheader_capture_has_two_ids_and_repeat_lock_is_idempotent(self):
        games=[{'event':'A @ B (2026-10-20)','market':'ML','kind':'TOP','pick':'HOME','sport':'baseball','game_id':n} for n in (123,124)]
        with tempfile.TemporaryDirectory() as d,patch.object(OL,'_rows_from_soccer_1x2',return_value=[]),patch.object(OL,'_rows_from_totals',return_value=[]),patch.object(OL,'_rows_from_two_sided',side_effect=lambda prefix,*a:copy.deepcopy(games) if prefix=='mlb_predictions' else []):
            with patch.object(OL,'LEDGER',Path(d)/'paper.jsonl'):
                self.assertEqual(OL.lock(),2);self.assertEqual(OL.lock(),0)
                self.assertEqual([r['game_id'] for r in OL._load()],[123,124])
                for r in games:r.pop('game_id')
                self.assertEqual(OL.lock(),0)  # temporary missing identity cannot duplicate a captured game


if __name__ == '__main__':
    unittest.main()
