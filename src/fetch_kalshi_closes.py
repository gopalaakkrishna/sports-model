"""Collect Kalshi pre-game CLOSING lines for every sport we trade.

WHY THIS EXISTS

Every "can the model beat the market" conclusion in this project so far has
been measured against the wrong opponent. FINDINGS.md benchmarks soccer against
Pinnacle and market-average BOOKMAKER odds — 68k matches, blend weight 0.00,
conclusive. But we do not trade Pinnacle. We trade Kalshi, a far smaller
retail-driven market whose prices are not the same thing.

Only MLB closes were ever collected (fetch_kalshi_mlb.py), which is why the
soft-market question could never be answered for anything else. Measured
2026-08-10, Kalshi's median spread ranged from 1c (MLB, WNBA) to 29c (Liga MX)
— a 29x difference in how tightly a market is priced. That is exactly the axis
along which a soft price would live, and we have no closing data to test it on.

This collects that data uniformly.

WHAT "PRE-GAME" MEANS HERE, AND WHY IT IS NOT THE CLOSING LINE

Kalshi contracts trade THROUGH the event, so a settled market's last price
knows the score. Using it would be lookahead of the worst kind. The
candlesticks endpoint gives per-minute history, so the fix is to cut off
before the game starts — but Kalshi exposes no reliable start time:

  close_time                02:34:38Z   <- game actually ended
  settlement_ts             02:37:15Z
  occurrence_datetime       03:00:00Z   <- NOT the start
  expected_expiration_time  03:00:00Z   <- identical to occurrence

`occurrence_datetime` is the expected EXPIRATION. Anchoring to it produced a
dataset where every winning leg closed at 0.99 — i.e. post-game prices. The
first version of this file did exactly that, which is why the degeneracy check
at the bottom now exists.

The ticker's embedded HHMM (MLB, cricket) is venue-local, not UTC, and absent
entirely for WNBA, NFL and soccer. So there is no uniform start time to use.

Rather than guess one, this anchors to `close_time` — which lands within
minutes of the final whistle — and steps back by more than any plausible event
duration. That yields a price guaranteed to be pre-game.

This is deliberately NOT the closing line, and calling it one would be wrong.
It is the price some hours before the event. That is arguably the more useful
benchmark anyway: the pipeline runs every 15 minutes and commits picks up to
36h ahead, so a line hours out is much closer to what we could actually trade
than a whistle-time close we would never reach.

SHAPE: MLB is two-way; soccer is three-way (a draw is a real outcome). A wide
home/away schema cannot hold that, so rows are stored long — one per market
leg — and callers pivot as they need.

    python src/fetch_kalshi_closes.py                 # all series, incremental
    python src/fetch_kalshi_closes.py --series KXWNBAGAME
    python src/fetch_kalshi_closes.py --max-events 50  # a bounded first pass
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
K = "https://api.elections.kalshi.com/trade-api/v2"
OUT = ROOT / "data" / "raw" / "kalshi_closes.parquet"

# Everything we price. MLB is included so it eventually replaces the separate
# (and slightly mis-timed) kalshi_mlb_closes.parquet with a correct cut-off.
SERIES = [
    "KXMLBGAME", "KXWNBAGAME", "KXNFLGAME", "KXHUNDREDMATCH",
    "KXMLSGAME", "KXLIGAMXGAME", "KXLALIGAGAME", "KXBUNDESLIGA2GAME",
    "KXALLSVENSKANGAME", "KXELITESERIENGAME", "KXSCOTTISHPREMGAME",
    "KXJLEAGUEGAME", "KXEPLGAME", "KXSERIEAGAME", "KXLIGUE1GAME",
]

STATS = defaultdict(int)

# Hours before market close to sample the price. `close_time` sits within
# minutes of the final whistle, so this must exceed the longest plausible
# event — including extra innings, overtime and stoppages — or the sample
# lands mid-game and the price knows the score.
#
#   MLB      9 innings ~3h, extras have run past 6h
#   NFL      ~3h10 typical, overtime longer
#   cricket  The Hundred ~2h30
#   soccer   ~2h with stoppage; ~2h30 if extra time
#   WNBA     ~2h, overtime longer
#
# 8 hours clears all of them with room to spare. The cost of being generous is
# a slightly staler price; the cost of being tight is a contaminated dataset
# that silently invalidates every conclusion drawn from it.
LEAD_HOURS = 8.0

# A clean pre-game sample should be spread across the probability range. If a
# series comes back mostly pinned at the extremes, the cut-off is landing
# in-game for that series and the data must not be trusted.
DEGENERATE_FRAC = 0.60
EXTREME_LO, EXTREME_HI = 0.05, 0.95


def settled_markets(series: str, session: requests.Session) -> list[dict]:
    out, cursor = [], None
    while True:
        p = {"series_ticker": series, "status": "settled", "limit": 200}
        if cursor:
            p["cursor"] = cursor
        try:
            r = session.get(f"{K}/markets", params=p, timeout=60)
        except requests.RequestException:
            break
        if r.status_code != 200:
            break
        j = r.json()
        ms = j.get("markets", [])
        out.extend(ms)
        cursor = j.get("cursor")
        if not cursor or not ms:
            break
        time.sleep(0.1)
    return out


def pregame_close(series: str, ticker: str, open_iso: str, start: datetime,
                  session: requests.Session, max_retries: int = 6):
    """(price, volume) from the last candle strictly BEFORE `start`.

    A 429 must never be read as "no data" — doing so silently discarded 97% of
    games on the MLB collector's first run, and the fetch still looked like it
    succeeded. Throttling is retried with backoff and counted.
    """
    try:
        open_t = datetime.fromisoformat(str(open_iso).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        STATS["bad_open_time"] += 1
        return None

    cs, delay = None, 0.5
    for _ in range(max_retries):
        try:
            r = session.get(f"{K}/series/{series}/markets/{ticker}/candlesticks",
                            params={"start_ts": int(open_t.timestamp()),
                                    "end_ts": int(start.timestamp()),
                                    "period_interval": 60}, timeout=60)
        except requests.RequestException:
            time.sleep(delay); delay = min(delay * 2, 16.0); continue
        if r.status_code == 200:
            cs = r.json().get("candlesticks", [])
            STATS["ok"] += 1
            break
        if r.status_code == 429:
            STATS["rate_limited"] += 1
            time.sleep(delay); delay = min(delay * 2, 16.0); continue
        STATS[f"http_{r.status_code}"] += 1
        time.sleep(delay); delay = min(delay * 2, 16.0)
    if cs is None:
        STATS["gave_up"] += 1
        return None

    best = None
    for c in cs:
        if c.get("end_period_ts", 0) > start.timestamp():
            continue
        px = (c.get("price") or {}).get("close_dollars")
        if px in (None, ""):
            continue
        try:
            v = float(px)
        except (TypeError, ValueError):
            continue
        if not (0.0 < v < 1.0):
            continue
        best = (v, float(c.get("volume_fp") or 0.0))
    if best is None:
        STATS["no_priced_candle"] += 1
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default=None,
                    help="comma-separated subset (default: all)")
    ap.add_argument("--max-events", type=int, default=None,
                    help="stop after this many NEW events per series")
    ap.add_argument("--pace", type=float, default=0.25,
                    help="seconds between candlestick calls")
    args = ap.parse_args()

    series_list = ([s.strip() for s in args.series.split(",") if s.strip()]
                   if args.series else SERIES)

    have = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
    seen = set(have["market_ticker"]) if len(have) else set()
    print(f"existing rows: {len(have):,} across "
          f"{have['series'].nunique() if len(have) else 0} series")

    session = requests.Session()
    rows = []
    for series in series_list:
        ms = settled_markets(series, session)
        if not ms:
            print(f"  {series:<24} no settled markets")
            continue
        by_ev = defaultdict(list)
        for m in ms:
            by_ev[m.get("event_ticker")].append(m)

        new_events = [ev for ev, mk in by_ev.items()
                      if any(m.get("ticker") not in seen for m in mk)]
        if args.max_events:
            new_events = new_events[:args.max_events]
        print(f"  {series:<24} {len(ms):>5} markets  {len(by_ev):>4} events  "
              f"{len(new_events):>4} new")

        for i, ev in enumerate(new_events, 1):
            mk = by_ev[ev]
            # Anchor to close_time (≈ final whistle), not occurrence_datetime
            # (which is the expected EXPIRATION and lands after the result).
            ct = mk[0].get("close_time")
            if not ct:
                STATS["no_close_time"] += 1
                continue
            try:
                closed = datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
            except ValueError:
                STATS["bad_close_time"] += 1
                continue
            start = closed - timedelta(hours=LEAD_HOURS)
            title = str(mk[0].get("title", "")).strip()
            for m in mk:
                tick = m.get("ticker")
                if not tick or tick in seen:
                    continue
                pc = pregame_close(series, tick, m.get("open_time", ""),
                                   start, session)
                if pc is None:
                    continue
                rows.append({
                    "series": series,
                    "event_ticker": ev,
                    "market_ticker": tick,
                    "date": closed.date().isoformat(),
                    "sampled_utc": start.isoformat(),
                    "close_time_utc": closed.isoformat(),
                    "lead_hours": LEAD_HOURS,
                    "title": title,
                    "leg": str(m.get("yes_sub_title", "")).strip(),
                    "close": pc[0],
                    "volume": pc[1],
                    # 'yes' means this leg happened. Kalshi supplies the
                    # outcome directly, so no external results feed is needed.
                    "result": m.get("result"),
                })
                time.sleep(args.pace)
            if i % 20 == 0:
                print(f"    {i}/{len(new_events)} events  kept={len(rows)}  "
                      f"ok={STATS['ok']} 429={STATS['rate_limited']}", end="\r")
        print()

    if not rows:
        print("\nno new rows collected")
        return 0

    fresh = pd.DataFrame(rows)
    df = (pd.concat([have, fresh], ignore_index=True)
          if len(have) else fresh)
    df = df.drop_duplicates(subset=["market_ticker"], keep="last")
    df = df.sort_values(["series", "date"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"\nrequest stats: {dict(STATS)}")

    # INTEGRITY GATE. A pre-game sample must be spread across the probability
    # range. If it is pinned at the extremes, the cut-off landed in-game and
    # the price knows the result — which is precisely the bug the first version
    # of this file shipped (every winning leg at 0.99). Report it loudly per
    # series rather than let a contaminated benchmark reach an analysis.
    print(f"\n{'series':<24}{'legs':>7}{'events':>8}{'extreme':>9}  {'dates':<26}state")
    bad = []
    for s, g in df.groupby("series"):
        ext = ((g["close"] < EXTREME_LO) | (g["close"] > EXTREME_HI)).mean()
        state = "ok"
        if ext > DEGENERATE_FRAC:
            state = "SUSPECT — looks post-game"
            bad.append(s)
        print(f"  {s:<22}{len(g):>7}{g['event_ticker'].nunique():>8}"
              f"{ext:>8.0%}   {g['date'].min()} .. {g['date'].max()}  {state}")

    # A winner-vs-price sanity check: if the sample is genuinely pre-game the
    # winning legs should average well below 1.0. Near 1.0 means lookahead.
    w = df[df["result"] == "yes"]["close"]
    if len(w):
        print(f"\n  mean price on legs that WON: {w.mean():.3f}"
              f"   (near 1.00 would mean the sample is post-game)")

    if bad:
        print(f"\n  !! {len(bad)} series look contaminated: {', '.join(bad)}")
        print("     Do NOT benchmark against those until the cut-off is fixed.")

    print(f"\nadded {len(fresh):,} rows -> {len(df):,} total")
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
