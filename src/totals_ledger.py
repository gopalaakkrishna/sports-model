"""Record and settle the totals paper lane, so it produces evidence.

WHY THIS EXISTS

The totals lane shipped picks to the board but wrote none of them down.
That is the same failure that lost the Toronto and Seattle calls: a pick is
surfaced, never recorded, and then does not exist when it loses. A lane that
only displays is not a paper lane, it is decoration — and the entire stated
justification for totals was to find out whether the aligned+floor filter
reproduces the winner board's record on a second market type. Without
settlement that question can never be answered.

So surfacing and recording are one action here too.

WHY THIS IS SIMPLER THAN THE WINNER LEDGER

The winner ledger has to match a fixture across three naming conventions and
guard against multi-game series, which is where several past bugs came from.
Totals picks carry the Kalshi ticker they were priced from, and that ticker
IS the market. It is unique, stable, and settles itself:

    KXMLSTOTAL-26AUG19CLBMTL-3  ->  result: "yes" | "no"

So the id is the ticker, settlement is one authoritative lookup, and no
name resolution is involved anywhere.

SEPARATE FILE ON PURPOSE

data/processed/totals_ledger.jsonl, not the main ledger. The headline record
is the thing whose honesty everything else rests on, and totals have no
measured edge over the closing line — merging them would import an unearned
claim into it. Kept apart, they can be scored on their own and compared.

    python totals_ledger.py --lock      # commit today's picks
    python totals_ledger.py --settle    # resolve finished ones via Kalshi
    python totals_ledger.py --report
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import paper_settlement as PS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

LEDGER = ROOT / "data" / "processed" / "totals_ledger.jsonl"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"

# Matches the winner board's actionable window. A pick priced three weeks out
# would be recorded against a price that has long since moved, and the price
# at lock time is the whole point of the row.
LOCK_HOURS = 36.0


def _load() -> list[dict]:
    return PS.load(LEDGER)


def _save(rows: list[dict]) -> None:
    PS.atomic_save(LEDGER, rows)


def _latest_report() -> Path | None:
    c = sorted(glob.glob(str(ROOT / "reports" / "totals_*.csv")))
    return Path(c[-1]) if c else None


def lock() -> int:
    """Commit today's totals picks that start inside the window."""
    rep = _latest_report()
    if rep is None:
        print("no totals report to lock from")
        return 0
    df = pd.read_csv(rep)
    if "pick" not in df.columns:
        print("report has no pick column")
        return 0
    df = df[df["pick"] == True]
    rows = _load()
    have = {r["id"] for r in rows}
    now = datetime.now(timezone.utc)
    n = 0
    for _, r in df.iterrows():
        tk = str(r.get("ticker") or "").strip()
        if not tk or tk in have:
            continue
        # Window test. A missing clock is treated as out-of-window rather than
        # in: locking a fixture we cannot time would record a price against a
        # game that might be weeks away.
        when = str(r.get("when") or "")
        try:
            start = datetime.fromisoformat(when.replace("Z", "+00:00"))
        except ValueError:
            continue
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if not (now - timedelta(hours=6) <= start <= now + timedelta(hours=LOCK_HOURS)):
            continue
        rows.append({
            "id": tk,
            "locked_at": now.isoformat(timespec="seconds"),
            "sport": "soccer",
            "league": str(r.get("league")),
            "event": str(r.get("match")),
            "market": f"OVER {r.get('line')} GOALS",
            "line": float(r.get("line")),
            "start": when,
            # The price actually payable, and the model's calibrated view of
            # it. Both are needed: the record is scored on outcomes, but
            # breakeven is scored on what the fill cost.
            "model_prob": float(r.get("model")),
            "market_prob": float(r.get("ask")),
            "bid": float(r.get("bid")),
            "ask": float(r.get("ask")),
            "lane": "paper",
            "notes": "totals aligned+floor; paper lane, not in headline record",
            "outcome": None, "won": None, "settled_at": None,
        })
        have.add(tk)
        n += 1
    if n:
        _save(rows)
    print(f"locked {n} new totals pick(s)  (ledger now {len(rows)})")
    return n


def settle() -> int:
    """Resolve open rows from Kalshi's own settlement."""
    rows = _load()
    import settle as S
    open_rows = [r for r in rows if PS.pending(r)]
    if not open_rows:
        print("nothing open to settle")
        return 0
    now = datetime.now(timezone.utc)
    n = 0
    before = json.dumps(rows, sort_keys=True)
    for r in open_rows:
        # Do not even ask until the game has had time to finish; an open
        # market returns no result and burns a request.
        try:
            start = datetime.fromisoformat(str(r["start"]).replace("Z", "+00:00"))
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if now - start < timedelta(hours=2.5):
                continue
        except (ValueError, KeyError):
            pass
        res = S.settle_exact_market(r, r['id'], "OVER", "UNDER")
        if res is None:
            continue
        outcome, detail = res
        # This lane always buys OVER; never change its original capture to
        # manufacture the pick field required by winner-ledger settlement.
        r["outcome"] = outcome
        r["won"] = None if outcome == "SCALAR" else outcome == "OVER"
        r["settled_at"] = now.isoformat(timespec="seconds")
        r["settlement_detail"] = detail
        r["settlement_status"] = "resolved"
        r["settlement_version"] = PS.VERSION
        r.pop("settlement_blocker", None)
        if outcome == "SCALAR":
            r.update(voided=True, void_reason=detail)
        n += 1
    if json.dumps(rows, sort_keys=True) != before:
        _save(rows)
    print(f"settled {n} totals pick(s)")
    return n


def report() -> dict:
    """Score the lane the same way the winner board is scored."""
    rows = _load()
    done = [r for r in rows if r.get("won") is not None and not r.get("voided")]
    out = {"n_total": len(rows), "n_settled": len(done),
           "wins": 0, "losses": 0}
    if not done:
        print(f"totals lane: {len(rows)} recorded, none settled yet")
        return out
    w = sum(1 for r in done if r["won"])
    n = len(done)
    px = [float(r["market_prob"]) for r in done]
    fee = sum(0.07 * p * (1 - p) for p in px) / n
    be = sum(px) / n + fee
    wr = w / n
    # Units are what a flat 1-contract stake would have returned, net of fee.
    units = sum((1.0 if r["won"] else 0.0) - float(r["market_prob"])
                - 0.07 * float(r["market_prob"]) * (1 - float(r["market_prob"]))
                for r in done)
    out.update({"wins": w, "losses": n - w, "win_rate": wr,
                "breakeven": be, "edge": wr - be, "units": units})
    print(f"totals lane (paper): {w}-{n - w} of {n} settled  "
          f"win rate {wr:.0%}  breakeven {be:.0%}  edge {wr - be:+.0%}  "
          f"units {units:+.2f}")
    print(f"  {PS.health(rows)['pending']} still open; non-binary/cancelled excluded from W/L")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if not (a.lock or a.settle or a.report):
        a.lock = a.settle = a.report = True
    if a.lock:
        lock()
    if a.settle:
        settle()
    if a.report:
        report()
    return 1 if PS.health(_load())["source_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
