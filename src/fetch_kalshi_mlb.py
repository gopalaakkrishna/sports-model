"""Build an MLB market benchmark from Kalshi settled game markets.

Kalshi contracts trade THROUGH the game, so `last_price` on a settled market is
contaminated by in-game information — using it as a benchmark would be lookahead
of the worst kind (the price knows the score). The candlesticks endpoint gives
per-minute history, so the last candle strictly before first pitch is a genuine
pre-game closing line.

First pitch comes from the ticker itself: KXMLBGAME-26AUG051510TBCOL-TB encodes
2026-08-05 15:10 UTC.

Output: one row per game with both sides' closing prices, the settled result,
and the volume behind the close.
"""

from __future__ import annotations

import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
K = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXMLBGAME"

MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}

KALSHI_TO_MLB = {
    "Los Angeles A": "Los Angeles Angels", "Los Angeles D": "Los Angeles Dodgers",
    "New York Y": "New York Yankees", "New York M": "New York Mets",
    "Chicago C": "Chicago Cubs", "Chicago W": "Chicago White Sox",
    "Chicago WS": "Chicago White Sox", "Chicago Cubs": "Chicago Cubs",
    "A's": "Athletics", "Athletics": "Athletics",
    "Boston": "Boston Red Sox", "Atlanta": "Atlanta Braves",
    "Detroit": "Detroit Tigers", "San Francisco": "San Francisco Giants",
    "Arizona": "Arizona Diamondbacks", "Houston": "Houston Astros",
    "San Diego": "San Diego Padres", "Tampa Bay": "Tampa Bay Rays",
    "Seattle": "Seattle Mariners", "Baltimore": "Baltimore Orioles",
    "Texas": "Texas Rangers", "Colorado": "Colorado Rockies",
    "St. Louis": "St. Louis Cardinals", "Miami": "Miami Marlins",
    "Milwaukee": "Milwaukee Brewers", "Minnesota": "Minnesota Twins",
    "Kansas City": "Kansas City Royals", "Cleveland": "Cleveland Guardians",
    "Cincinnati": "Cincinnati Reds", "Pittsburgh": "Pittsburgh Pirates",
    "Philadelphia": "Philadelphia Phillies", "Washington": "Washington Nationals",
    "Toronto": "Toronto Blue Jays",
}


def first_pitch_utc(ticker: str) -> datetime | None:
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})(\d{4})", ticker)
    if not m:
        return None
    yy, mon, dd, hhmm = m.groups()
    if mon not in MONTHS:
        return None
    try:
        return datetime(2000 + int(yy), MONTHS[mon], int(dd),
                        int(hhmm[:2]), int(hhmm[2:]), tzinfo=timezone.utc)
    except ValueError:
        return None


def all_settled() -> list[dict]:
    out, cursor = [], None
    while True:
        p = {"series_ticker": SERIES, "status": "settled", "limit": 200}
        if cursor:
            p["cursor"] = cursor
        r = requests.get(f"{K}/markets", params=p, timeout=60)
        if r.status_code != 200:
            break
        j = r.json()
        ms = j.get("markets", [])
        out.extend(ms)
        cursor = j.get("cursor")
        print(f"  fetched {len(out)} settled markets", end="\r")
        if not cursor or not ms:
            break
        time.sleep(0.1)
    print()
    return out


# Kalshi rate-limits the candlesticks endpoint. Treating a 429 as "no data"
# silently discarded 97% of games on the first run — the fetch looked like it
# succeeded and simply returned a tiny sample. Any non-200 must be retried, and
# throttling must be counted and reported, never swallowed.
STATS_COUNTER = {"ok": 0, "rate_limited": 0, "failed": 0, "no_price": 0}


def pregame_close(ticker: str, open_iso: str, fp: datetime,
                  session: requests.Session, max_retries: int = 6
                  ) -> tuple[float, float] | None:
    """(close price, volume in that candle) from the last candle before first pitch."""
    try:
        open_t = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
    except Exception:
        STATS_COUNTER["failed"] += 1
        return None

    cs = None
    delay = 0.5
    for attempt in range(max_retries):
        try:
            r = session.get(f"{K}/series/{SERIES}/markets/{ticker}/candlesticks",
                            params={"start_ts": int(open_t.timestamp()),
                                    "end_ts": int(fp.timestamp()),
                                    "period_interval": 60}, timeout=60)
        except requests.RequestException:
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code == 200:
            cs = r.json().get("candlesticks", [])
            STATS_COUNTER["ok"] += 1
            break
        if r.status_code == 429:
            STATS_COUNTER["rate_limited"] += 1
            time.sleep(delay)
            delay = min(delay * 2, 16.0)
            continue
        time.sleep(delay)
        delay *= 2
    if cs is None:
        STATS_COUNTER["failed"] += 1
        return None

    best = None
    for c in cs:
        ts = c.get("end_period_ts", 0)
        if ts > fp.timestamp():
            continue
        px = (c.get("price") or {}).get("close_dollars")
        if px in (None, ""):
            continue
        try:
            val = float(px)
        except ValueError:
            continue
        if val <= 0.0 or val >= 1.0:
            continue
        best = (val, float(c.get("volume_fp") or 0.0))
    if best is None:
        STATS_COUNTER["no_price"] += 1
    return best


def main():
    print("fetching settled Kalshi MLB markets...")
    ms = all_settled()
    by_ev = defaultdict(list)
    for m in ms:
        by_ev[m.get("event_ticker")].append(m)
    print(f"{len(ms)} markets across {len(by_ev)} games")

    rows, done, unmapped = [], 0, {}
    session = requests.Session()
    for ev, mk in by_ev.items():
        title = str(mk[0].get("title", "")).replace(" Winner?", "").strip()
        if " vs " not in title:
            continue
        a_raw, b_raw = [s.strip() for s in title.split(" vs ", 1)]
        away, home = KALSHI_TO_MLB.get(a_raw), KALSHI_TO_MLB.get(b_raw)
        fp = first_pitch_utc(mk[0].get("ticker", ""))
        if not away or not home or fp is None:
            for nm, ok in ((a_raw, away), (b_raw, home)):
                if not ok:
                    unmapped[nm] = unmapped.get(nm, 0) + 1
            continue

        legs = {}
        for m in mk:
            team = KALSHI_TO_MLB.get(str(m.get("yes_sub_title", "")).strip())
            if not team:
                continue
            pc = pregame_close(m["ticker"], m.get("open_time", ""), fp, session)
            if pc is None:
                continue
            legs[team] = {"close": pc[0], "vol": pc[1], "result": m.get("result")}
            time.sleep(0.25)   # pace against the rate limiter
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(by_ev)} games  kept={len(rows)}  "
                  f"ok={STATS_COUNTER['ok']} 429={STATS_COUNTER['rate_limited']} "
                  f"fail={STATS_COUNTER['failed']}", end="\r")
        if home not in legs or away not in legs:
            continue

        res_home = legs[home]["result"]
        if res_home not in ("yes", "no"):
            continue
        rows.append({
            "date": fp.date().isoformat(),
            "first_pitch_utc": fp.isoformat(),
            "home": home, "away": away,
            "close_home": legs[home]["close"], "close_away": legs[away]["close"],
            "vol_home": legs[home]["vol"], "vol_away": legs[away]["vol"],
            "home_win": 1 if res_home == "yes" else 0,
        })
    print()

    if not rows:
        print("no usable rows", file=sys.stderr)
        sys.exit(1)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["book_sum"] = df["close_home"] + df["close_away"]
    out = ROOT / "data" / "raw" / "kalshi_mlb_closes.parquet"
    df.to_parquet(out, index=False)

    print(f"\nrequest stats: ok={STATS_COUNTER['ok']} "
          f"rate_limited_retries={STATS_COUNTER['rate_limited']} "
          f"gave_up={STATS_COUNTER['failed']} no_priced_candle={STATS_COUNTER['no_price']}")
    if unmapped:
        print(f"  unmapped Kalshi team names: {unmapped}")
    kept_frac = len(df) / max(len(by_ev), 1)
    if kept_frac < 0.6:
        print(f"  !! kept only {kept_frac:.0%} of games — investigate before using")

    print(f"\nSaved {len(df):,} games -> {out}")
    print(f"  dates {df['date'].min()} .. {df['date'].max()}")
    print(f"  mean book sum (1.00 = no spread): {df['book_sum'].mean():.4f}")
    print(f"  home win rate: {df['home_win'].mean():.3%}")
    print(f"  mean pre-game close on home: {df['close_home'].mean():.3f}")
    print(f"  median closing-hour volume: {df[['vol_home','vol_away']].median().mean():,.0f}")


if __name__ == "__main__":
    main()
