"""Locked-prediction ledger: record, settle, and score.

Design rules, because a scoreboard that flatters you is worse than none:

* A prediction is LOCKED with a timestamp and the market price *at that moment*.
  Without the contemporaneous price you cannot tell later whether you beat the
  market or just agreed with it.
* Nothing may be edited after locking. Settling only fills in the result.
* Scoring is by log loss and Brier against the market's log loss on the same
  set — not by win rate. Win rate is nearly meaningless: picking heavy
  favourites gives a great win rate and can still lose money, and a 60% call
  that loses is not a bad call.
* Results are broken out by sport, market type and confidence band, so a good
  soccer record cannot hide a bad basketball one.

    python ledger.py lock --sport soccer --event "Inter Miami v Atl. San Luis" \
        --market 1X2 --pick HOME --model-prob 0.69 --market-prob 0.68 --odds 1.47
    python ledger.py settle --id 3 --outcome HOME
    python ledger.py report
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "processed" / "ledger.jsonl"
EPS = 1e-15

FIELDS = [
    "id", "locked_at", "sport", "league", "event", "market", "pick",
    "model_prob", "market_prob", "odds", "venue", "stake_units", "notes",
    "settled_at", "outcome", "won", "pnl_units",
]


def _load() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def _save(rows: list[dict]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _fixture_key(event: str) -> tuple:
    """The two teams AND the date. Teams alone are not a fixture.

    This used to drop the date, because "Toronto Tempo @ Portland Fire" tips at
    01:00 ET and two DIFFERENT code paths were computing two different date
    labels for that ONE game (venue-local vs ET) — same game, logged twice,
    looking like two different rows. Stripping the date "fixed" that by
    treating the two labels as equal.

    It also broke something worse: teams play multi-game series. Mets @ Braves
    on the 10th and Mets @ Braves on the 11th are two DIFFERENT games, and
    dropping the date made the second one look like a duplicate of the first —
    auto-lock would silently refuse to record a genuinely new pick.

    The venue/ET ambiguity that motivated this is now fixed at the source:
    auto_lock always derives its event date from the same `ev_date`
    computation, so the same physical game gets the same label every time.
    With that fixed upstream, the date belongs back in the key — it is part of
    what makes a fixture a fixture.
    """
    ev = re.sub(r"\s*\((\d{4}-\d{2}-\d{2})\)\s*$", "", str(event)).strip()
    m = re.search(r"\((\d{4}-\d{2}-\d{2})\)\s*$", str(event))
    date = m.group(1) if m else None
    for sep in (" @ ", " vs ", " v "):
        if sep in ev:
            x, y = ev.split(sep, 1)
            return (frozenset({x.strip().lower(), y.strip().lower()}), date)
    return (frozenset({ev.lower()}), date)


def cmd_lock(a) -> None:
    rows = _load()
    # A voided prediction must not block its own replacement.
    key = (_fixture_key(a.event), a.market, a.pick)
    if any((_fixture_key(r["event"]), r["market"], r["pick"]) == key
           and r.get("outcome") is None and not r.get("voided") for r in rows):
        print("an unsettled identical prediction already exists — not duplicating")
        return
    # A backfill is a lock written after the fact. It is allowed — a call that
    # was genuinely made and then never written down is missing from the record,
    # which is its own kind of dishonesty — but it is marked, permanently and
    # visibly, because it is weaker evidence than a true pre-game lock. The
    # price has to come from a contemporaneous artefact, not from memory.
    backfill = getattr(a, "backfill_at", None)
    if backfill and not getattr(a, "backfill_reason", None):
        print("--backfill-at requires --backfill-reason "
              "(where the contemporaneous price came from)")
        return
    rec = {
        "id": max([r["id"] for r in rows], default=0) + 1,
        "locked_at": backfill or
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sport": a.sport, "league": a.league, "event": a.event,
        "market": a.market, "pick": a.pick,
        "model_prob": a.model_prob, "market_prob": a.market_prob,
        "odds": a.odds, "venue": a.venue, "stake_units": a.stake_units,
        "notes": a.notes, "settled_at": None, "outcome": None,
        "won": None, "pnl_units": None,
    }
    if backfill:
        rec["backfilled"] = True
        rec["backfill_reason"] = a.backfill_reason
    rows.append(rec)
    _save(rows)
    edge = (a.model_prob - a.market_prob) if a.market_prob is not None else None
    print(f"locked #{rec['id']}: {a.event} | {a.market} | {a.pick} "
          f"@ model {a.model_prob:.1%}"
          + (f", market {a.market_prob:.1%}, edge {edge:+.1%}" if edge is not None else ""))


def cmd_settle(a) -> None:
    rows = _load()
    for r in rows:
        if r["id"] != a.id:
            continue
        if r["outcome"] is not None:
            print(f"#{a.id} already settled as {r['outcome']}")
            return
        r["outcome"] = a.outcome
        r["settled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        r["won"] = bool(a.outcome == r["pick"])
        if r["odds"] and r["stake_units"]:
            r["pnl_units"] = (r["stake_units"] * (r["odds"] - 1)) if r["won"] \
                else -r["stake_units"]
        _save(rows)
        print(f"settled #{a.id}: {r['event']} -> {a.outcome} "
              f"({'WON' if r['won'] else 'lost'})")
        return
    print(f"no prediction with id {a.id}")


def cmd_void(a) -> None:
    """Void a prediction made in error, BEFORE the event settles.

    Predictions are immutable once locked, so a mistake cannot simply be edited
    away. Voiding keeps the original row visible with a stated reason and
    excludes it from scoring. Anything already settled can never be voided —
    that would let losers quietly disappear from the record.
    """
    rows = _load()
    for r in rows:
        if r["id"] != a.id:
            continue
        if r.get("outcome") is not None:
            print(f"#{a.id} is already settled — refusing to void. "
                  "Settled predictions stay in the record.")
            return
        if r.get("voided"):
            print(f"#{a.id} already voided")
            return
        r["voided"] = True
        r["void_reason"] = a.reason
        r["voided_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _save(rows)
        print(f"voided #{a.id}: {r['event']} — {a.reason}")
        return
    print(f"no prediction with id {a.id}")


def cmd_unsettle(a) -> None:
    """Reverse a settlement that was recorded from bad data.

    Settling is meant to be one-way so losses cannot quietly vanish. But a
    settlement entered from a WRONG result is not a loss to be hidden — it is
    corrupt data, and leaving it in poisons every number downstream. So this
    exists, requires a reason, and records the reversal permanently in the row's
    history rather than erasing it.
    """
    rows = _load()
    for r in rows:
        if r["id"] != a.id:
            continue
        if r.get("outcome") is None:
            print(f"#{a.id} is not settled")
            return
        hist = r.setdefault("correction_history", [])
        hist.append({
            "reverted_outcome": r["outcome"],
            "reverted_won": r.get("won"),
            "reason": a.reason,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        r["outcome"] = None
        r["settled_at"] = None
        r["won"] = None
        r["pnl_units"] = None
        _save(rows)
        print(f"unsettled #{a.id}: {r['event']}")
        print(f"  reason: {a.reason}")
        print(f"  (previous settlement kept in correction_history)")
        return
    print(f"no prediction with id {a.id}")


def cmd_report(a) -> None:
    all_rows = _load()
    corrected = [r for r in all_rows if r.get("correction_history")]
    if corrected:
        print(f"  ({len(corrected)} row(s) have corrected settlements)")
        for r in corrected:
            for h in r["correction_history"]:
                print(f"    #{r['id']} was '{h['reverted_outcome']}' — {h['reason']}")
    voided = [r for r in all_rows if r.get("voided")]
    if voided:
        print(f"  ({len(voided)} voided prediction(s), excluded from scoring)")
        for r in voided:
            print(f"    #{r['id']} {r['event']} — {r.get('void_reason', '')}")
    rows = [r for r in all_rows
            if r["outcome"] is not None and not r.get("voided")]
    if not rows:
        print("no settled predictions yet")
        pend = [r for r in _load() if r["outcome"] is None]
        if pend:
            print(f"({len(pend)} still open)")
        return
    df = pd.DataFrame(rows)

    def scores(g):
        p = g["model_prob"].clip(EPS, 1 - EPS)
        y = g["won"].astype(int)
        ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        br = float(((p - y) ** 2).mean())
        out = {"n": len(g), "win_rate": float(y.mean()),
               "model_logloss": ll, "brier": br}
        mk = g["market_prob"].dropna()
        if len(mk) == len(g):
            q = g["market_prob"].clip(EPS, 1 - EPS)
            out["market_logloss"] = float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())
            out["vs_market"] = out["model_logloss"] - out["market_logloss"]
        if g["pnl_units"].notna().any():
            staked = g.loc[g["pnl_units"].notna(), "stake_units"].sum()
            pnl = g["pnl_units"].sum()
            out["staked"] = float(staked)
            out["pnl"] = float(pnl)
            out["roi"] = float(pnl / staked) if staked else float("nan")
        return out

    def show(title, grouped):
        print(f"\n  {title}")
        hdr = f"    {'group':<22}{'n':>5}{'win%':>7}{'logloss':>10}{'vs mkt':>9}{'ROI':>9}"
        print(hdr)
        for k, s in grouped:
            vs = f"{s.get('vs_market'):+.4f}" if "vs_market" in s else "  n/a"
            roi = f"{s.get('roi'):+.1%}" if "roi" in s and s["roi"] == s["roi"] else "  n/a"
            print(f"    {str(k):<22}{s['n']:>5}{s['win_rate']:>7.1%}"
                  f"{s['model_logloss']:>10.4f}{vs:>9}{roi:>9}")

    overall = scores(df)
    print("  OVERALL")
    print(f"    settled predictions   {overall['n']}")
    print(f"    win rate              {overall['win_rate']:.1%}")
    print(f"    model log loss        {overall['model_logloss']:.4f}")
    if "market_logloss" in overall:
        print(f"    market log loss       {overall['market_logloss']:.4f}")
        gap = overall["vs_market"]
        print(f"    vs market             {gap:+.4f}"
              f"   ({'BEATING market' if gap < 0 else 'losing to market'})")
    if "roi" in overall:
        print(f"    staked / P&L / ROI    {overall['staked']:.1f}u / "
              f"{overall['pnl']:+.2f}u / {overall['roi']:+.1%}")

    for col, title in [("sport", "BY SPORT"), ("market", "BY MARKET TYPE"),
                       ("league", "BY LEAGUE"), ("venue", "BY VENUE")]:
        if col in df and df[col].notna().any():
            show(title, [(k, scores(g)) for k, g in df.groupby(col)])

    band = pd.cut(df["model_prob"], [0, .4, .55, .7, .85, 1.0],
                  labels=["<40%", "40-55%", "55-70%", "70-85%", ">85%"])
    show("BY CONFIDENCE BAND", [(k, scores(g)) for k, g in df.groupby(band, observed=True)])

    n = overall["n"]
    if n < 100:
        print(f"\n  NOTE: {n} settled predictions is far too few to conclude anything.")
        print("  Distinguishing a real 2% edge from noise takes on the order of")
        print("  1,000+ bets. Treat everything above as bookkeeping, not evidence.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    lk = sub.add_parser("lock", help="record a prediction before kickoff")
    lk.add_argument("--sport", required=True)
    lk.add_argument("--league", default="")
    lk.add_argument("--event", required=True)
    lk.add_argument("--market", required=True, help="1X2, OU2.5, BTTS, ...")
    lk.add_argument("--pick", required=True, help="HOME/DRAW/AWAY/OVER/...")
    lk.add_argument("--model-prob", type=float, required=True, dest="model_prob")
    lk.add_argument("--market-prob", type=float, default=None, dest="market_prob")
    lk.add_argument("--odds", type=float, default=None)
    lk.add_argument("--venue", default="")
    lk.add_argument("--stake-units", type=float, default=None, dest="stake_units")
    lk.add_argument("--notes", default="")
    lk.add_argument("--backfill-at", dest="backfill_at", default=None,
                    help="ISO timestamp of when the call was actually made")
    lk.add_argument("--backfill-reason", dest="backfill_reason", default=None,
                    help="where the contemporaneous price came from")
    lk.set_defaults(func=cmd_lock)

    st = sub.add_parser("settle", help="fill in the result")
    st.add_argument("--id", type=int, required=True)
    st.add_argument("--outcome", required=True)
    st.set_defaults(func=cmd_settle)

    vd = sub.add_parser("void", help="void an unsettled prediction made in error")
    vd.add_argument("--id", type=int, required=True)
    vd.add_argument("--reason", required=True)
    vd.set_defaults(func=cmd_void)

    us = sub.add_parser("unsettle",
                        help="reverse a settlement entered from incorrect data")
    us.add_argument("--id", type=int, required=True)
    us.add_argument("--reason", required=True)
    us.set_defaults(func=cmd_unsettle)

    rp = sub.add_parser("report", help="scorecard")
    rp.set_defaults(func=cmd_report)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
