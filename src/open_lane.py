"""The unfiltered lane: every market, every call, no floor and no filters.

WHAT THIS IS

The main record only admits a pick when the model AGREES with the market and
clears 60%. Both conditions were added for measured reasons, and both are
still right for that record. This lane removes them entirely and writes down
whatever the model actually thinks, on every market it can price:

    soccer   HOME / DRAW / AWAY        (draws included — the 60% floor made
                                        them structurally impossible, since a
                                        draw has never exceeded 54%)
    soccer   OVER / UNDER 2.5 goals
    MLB      HOME / AWAY
    WNBA     HOME / AWAY
    NFL      HOME / AWAY, OVER / UNDER
    cricket  HOME / AWAY

One game can therefore produce several calls — a winner and a total are
separate rows on the same fixture. That is the point.

WHY IT IS A SEPARATE FILE

data/processed/open_ledger.jsonl, never data/processed/ledger.jsonl. The main
record's entire value is that it has never flattered anyone: every time it
looked good, the readiness metric said "not yet" and was right. Pouring
unfiltered picks into it would destroy the one honest measurement in the
project to answer a different question. Two ledgers, two scores, no mixing.

WHAT TO EXPECT, STATED IN ADVANCE

So that the result cannot be rationalised afterwards. Everything measured so
far says this lane should roughly break even before fees and lose after them:

  * the model does not beat the market anywhere it has been tested, and adds
    weight 0.00 to the opening line
  * removing the ALIGNED filter admits exactly the disagreement band that
    backtested as model error (1.0373 vs the market's 0.9641)
  * removing the floor admits coin flips, where Kalshi's fee peaks
  * the UNDER side of totals measured 51% actual against 63% stated
  * MLB totals scored worse than a constant, so they are excluded even here

If it does better than that, the interesting question is which subset carried
it, which is why every row records its market type and its price.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

import leagues as LG

LEDGER = ROOT / "data" / "processed" / "open_ledger.jsonl"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"


def _load() -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _save(rows: list[dict]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")


def _latest(prefix: str):
    c = sorted(glob.glob(str(ROOT / "reports" / f"{prefix}_*.csv")))
    if not c:
        return None
    newest = Path(c[-1])
    # Same 3-day bound the board uses. A dead predictor must show up as a
    # missing section, not as confident month-old calls.
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})", newest.name)
    if m:
        age = (pd.Timestamp.now().normalize() - pd.Timestamp(m.group(1))).days
        if age > 3:
            print(f"  {prefix}: newest report is {age}d old — skipping")
            return None
    return newest


def _rows_from_soccer_1x2() -> list[dict]:
    p = _latest("kalshi_edge")
    if p is None:
        return []
    d = pd.read_csv(p)
    out = []
    for ev, g in d.groupby("event"):
        g = g.dropna(subset=["model"])
        if g.empty:
            continue
        s = g["ask"].sum()

        def row(r, kind):
            mkt = float(r["ask"]) / s if s and s == s else None
            return {
                "sport": "soccer",
                "league": LG.pretty(str(r.get("series"))),
                "event": f"{r['match']} ({str(r.get('when'))[:10]})",
                "market": "1X2", "kind": kind,
                "pick": str(r["leg"]).upper(),
                "model_prob": float(r["model"]),
                "market_prob": mkt,
                "ask": float(r["ask"]) if r["ask"] == r["ask"] else None,
                # Liquidity travels WITH the call, because without it this
                # lane lies. Illiquid fixtures quote ask 0.80 on all three
                # legs against a 0.06 bid; normalising that gives ~33% per
                # leg, and the model then appears to find a 50-point edge on
                # Bayern at home. Nothing is filtered out here — the user
                # asked for free rein — but every row carries the spread so
                # the report can separate real prices from dead books.
                "spread": float(r["spread"]) if r.get("spread") == r.get("spread") else None,
                "depth": float(r["depth"]) if r.get("depth") == r.get("depth") else None,
                "tradeable": bool(r["tradeable"]) if "tradeable" in r else None,
                "start": str(r.get("when")), "ticker": None,
            }

        # TOP — the single outcome the model thinks most likely. Note this can
        # never be a draw: a draw peaks near 30% and never wins an argmax over
        # three outcomes, which is the same blind spot the 60% floor had, just
        # relocated. That is exactly why VALUE exists alongside it.
        out.append(row(g.loc[g["model"].idxmax()], "TOP"))

        # VALUE — every leg the model prices above the market, so a fixture can
        # produce several calls and a draw can finally be one of them.
        for _, r in g.iterrows():
            if s and s == s and float(r["model"]) > float(r["ask"]) / s:
                out.append(row(r, "VALUE"))
    return out


def _rows_from_totals() -> list[dict]:
    p = _latest("totals")
    if p is None:
        return []
    d = pd.read_csv(p)
    out = []
    for _, r in d.iterrows():
        mp = float(r["model"])
        over = mp >= 0.5
        ask = float(r["ask"])
        out.append({
            "sport": "soccer",
            "league": LG.pretty(str(r.get("league"))),
            "event": f"{r['match']} ({str(r.get('when'))[:10]})",
            "market": "TOTAL", "kind": "TOP",
            "pick": "OVER" if over else "UNDER",
            "line": float(r.get("line", 2.5)),
            "model_prob": mp if over else 1.0 - mp,
            "market_prob": ask if over else 1.0 - ask,
            "ask": ask if over else round(1.0 - ask, 4),
            "start": str(r.get("when")),
            # Totals settle straight off their own Kalshi market, which is
            # exact and needs no name matching at all.
            "ticker": str(r.get("ticker")) if r.get("ticker") == r.get("ticker") else None,
        })
    return out


def _rows_from_two_sided(prefix: str, sport: str, league: str) -> list[dict]:
    """MLB / WNBA reports: one row per side, pick the model's favourite."""
    p = _latest(prefix)
    if p is None:
        return []
    d = pd.read_csv(p)
    if "match" not in d.columns or "model" not in d.columns:
        return []
    out = []
    for match, g in d.groupby("match"):
        g = g.dropna(subset=["model"])
        if g.empty:
            continue
        top = g.loc[g["model"].idxmax()]
        date = str(top.get("date") or top.get("start") or "")[:10]
        out.append({
            "sport": sport, "league": league,
            "event": f"{match} ({date})",
            "market": "ML", "kind": "TOP",
            "pick": str(top.get("side", "")).upper(),
            "model_prob": float(top["model"]),
            "market_prob": float(top["ask"]) if "ask" in top and top["ask"] == top["ask"] else None,
            "ask": float(top["ask"]) if "ask" in top and top["ask"] == top["ask"] else None,
            "start": date, "ticker": None,
        })
    return out


def lock() -> int:
    """Write down everything the model currently thinks. No filters."""
    rows = _load()
    have = {(r["event"], r["market"], r.get("kind"), r.get("pick"),
         r.get("line")) for r in rows}
    now = datetime.now(timezone.utc)
    new = (_rows_from_soccer_1x2() + _rows_from_totals()
           + _rows_from_two_sided("mlb_predictions", "baseball", "MLB")
           + _rows_from_two_sided("wnba_kalshi", "basketball", "WNBA"))
    n = 0
    nxt = max((r.get("id", 0) for r in rows), default=0) + 1
    for r in new:
        key = (r["event"], r["market"], r.get("kind"), r.get("pick"),
               r.get("line"))
        if key in have:
            continue
        r.update({"id": nxt, "locked_at": now.isoformat(timespec="seconds"),
                  "outcome": None, "won": None, "settled_at": None,
                  "voided": False,
                  "notes": "open lane: unfiltered, no floor, no aligned filter"})
        rows.append(r)
        have.add(key)
        nxt += 1
        n += 1
    if n:
        _save(rows)
    print(f"open lane: locked {n} new call(s)  (ledger now {len(rows)})")
    return n


def _settle_ticker(tk: str):
    try:
        r = requests.get(f"{KALSHI}/markets/{tk}", timeout=30)
        if r.status_code != 200:
            return None
        res = str((r.json().get("market") or {}).get("result") or "").lower()
        time.sleep(0.1)
        return res if res in ("yes", "no") else None
    except requests.RequestException:
        return None


def settle() -> int:
    """Resolve open calls. Totals settle by ticker; the rest reuse settle.py."""
    import settle as S
    rows = _load()
    opn = [r for r in rows if r.get("won") is None and not r.get("voided")]
    if not opn:
        print("open lane: nothing to settle")
        return 0
    now = datetime.now(timezone.utc)
    hist = None
    n = 0
    for r in opn:
        # Do not ask before the game can plausibly have finished.
        try:
            st = pd.Timestamp(str(r.get("start"))[:10])
            if (pd.Timestamp.now().normalize() - st).days < 0:
                continue
        except (ValueError, TypeError):
            pass

        res = None
        if r["market"] == "TOTAL" and r.get("ticker"):
            got = _settle_ticker(r["ticker"])
            if got is not None:
                # The ticker is the OVER market, so "yes" means the over hit.
                actual = "OVER" if got == "yes" else "UNDER"
                res = (actual, f"Kalshi settled: {actual}")
        else:
            sport = str(r.get("sport", "")).lower()
            rec = {"event": r["event"], "sport": sport, "pick": r["pick"]}
            if sport == "soccer":
                res = S.settle_soccer_via_kalshi(rec)
                if res is None:
                    if hist is None:
                        import data as D
                        hist = D.load_history()
                    res = S.settle_soccer(rec, hist)
            elif sport in ("baseball", "mlb"):
                res = S.settle_mlb(rec)
            elif sport in S._KALSHI_SERIES:
                res = S.settle_via_kalshi(rec, sport)

        if res is None:
            continue
        outcome, _ = res
        r["outcome"] = outcome
        r["won"] = bool(outcome == r["pick"])
        r["settled_at"] = now.isoformat(timespec="seconds")
        n += 1
    if n:
        _save(rows)
    print(f"open lane: settled {n}")
    return n


def report() -> dict:
    rows = _load()
    done = [r for r in rows if r.get("won") is not None and not r.get("voided")]
    out = {"n_total": len(rows), "n_settled": len(done)}
    if not done:
        print(f"open lane: {len(rows)} recorded, none settled yet")
        return out
    df = pd.DataFrame(done)
    w = int(df["won"].sum())
    n = len(df)
    px = pd.to_numeric(df.get("ask"), errors="coerce")
    have = px.notna()
    fee = (0.07 * px[have] * (1 - px[have])).mean() if have.any() else 0.0
    be = (px[have].mean() + fee) if have.any() else float("nan")
    units = ((df["won"].astype(float) - px) - 0.07 * px * (1 - px))[have].sum()
    out.update({"wins": w, "losses": n - w, "win_rate": w / n,
                "breakeven": be, "edge": w / n - be, "units": float(units)})
    print(f"\nOPEN LANE (unfiltered)  {w}-{n - w} of {n} settled")
    print(f"  win rate {w / n:.1%}   breakeven {be:.1%}   "
          f"edge {w / n - be:+.1%}   units {units:+.2f}")
    print(f"  {len(rows) - n} still open\n")
    print(f"  {'market':<10}{'n':>5}{'W-L':>10}{'win%':>8}{'breakeven':>11}{'edge':>8}{'units':>9}")
    for mk, g in df.groupby("market"):
        gw = int(g["won"].sum())
        gp = pd.to_numeric(g.get("ask"), errors="coerce")
        h = gp.notna()
        gbe = (gp[h].mean() + (0.07 * gp[h] * (1 - gp[h])).mean()) if h.any() else float("nan")
        gu = ((g["won"].astype(float) - gp) - 0.07 * gp * (1 - gp))[h].sum()
        print(f"  {mk:<10}{len(g):>5}{f'{gw}-{len(g) - gw}':>10}{gw / len(g):>8.1%}"
              f"{gbe:>11.1%}{gw / len(g) - gbe:>+8.1%}{gu:>9.2f}")
    for col, lab in (("sport", "sport"), ("pick", "side")):
        if col not in df:
            continue
        print(f"\n  by {lab}:")
        for v, g in df.groupby(col):
            gw = int(g["won"].sum())
            print(f"    {str(v):<14}{len(g):>5}  {gw}-{len(g) - gw}  {gw / len(g):>6.1%}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
