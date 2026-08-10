"""Price WNBA games against Kalshi — the market benchmark the model lacked.

Basketball Reference carries no odds, so until now the WNBA model could be shown
to beat a base rate but not measured against a market at all. Kalshi quotes WNBA
games with real depth, which supplies the missing benchmark.

Ticker convention, verified against the rules text rather than assumed: in
KXWNBAGAME-26AUG08SEAPDX the rules read "the Seattle vs Portland game", and
Basketball Reference lists that fixture as Seattle AT Portland. So the FIRST
code is the away side and the second is the home side — the same
away-then-home ordering MLB uses, and the opposite of the soccer series. Reading
it the other way round would invert every prediction.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
import margin_model as MM

ROOT = Path(__file__).resolve().parents[1]
K = "https://api.elections.kalshi.com/trade-api/v2"
FEE_RATE = 0.07

# Kalshi short name -> Basketball Reference full name.
KALSHI_TO_BREF = {
    "Atlanta": "Atlanta Dream", "Chicago": "Chicago Sky",
    "Connecticut": "Connecticut Sun", "Dallas": "Dallas Wings",
    "Golden State": "Golden State Valkyries", "Indiana": "Indiana Fever",
    "Las Vegas": "Las Vegas Aces", "Los Angeles": "Los Angeles Sparks",
    "Minnesota": "Minnesota Lynx", "New York": "New York Liberty",
    "Phoenix": "Phoenix Mercury", "Seattle": "Seattle Storm",
    "Washington": "Washington Mystics", "Portland": "Portland Fire",
    "Toronto": "Toronto Tempo",
}

MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def ticker_date(ev: str) -> str | None:
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", ev)
    if not m or m.group(2) not in MONTHS:
        return None
    return f"20{m.group(1)}-{MONTHS[m.group(2)]:02d}-{int(m.group(3)):02d}"


def fee(p: float) -> float:
    return FEE_RATE * p * (1 - p)


def depth(ticker: str, within: float = 0.05) -> float:
    try:
        r = requests.get(f"{K}/markets/{ticker}/orderbook",
                         params={"depth": 8}, timeout=30)
        book = r.json().get("orderbook") or r.json().get("orderbook_fp") or {}
    except Exception:
        return 0.0
    tot = 0.0
    for side in ("yes", "yes_dollars", "no", "no_dollars"):
        lv = book.get(side) or []
        px = []
        for l in lv:
            try:
                px.append(float(l[0]))
            except (TypeError, ValueError, IndexError):
                pass
        if not px:
            continue
        best = max(px)
        for l in lv:
            try:
                p, s = float(l[0]), float(l[1])
            except (TypeError, ValueError, IndexError):
                continue
            if abs(p - best) <= within:
                tot += p * s
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-spread", type=float, default=0.06)
    ap.add_argument("--min-depth", type=float, default=500.0)
    ap.add_argument("--min-eff-n", type=float, default=12.0)
    ap.add_argument("--min-ev", type=float, default=0.02)
    args = ap.parse_args()

    g = pd.read_parquet(ROOT / "data" / "raw" / "wnba_games.parquet")
    g["date"] = pd.to_datetime(g["date"])
    hist = g[g["played"]]
    today = pd.Timestamp.now().normalize()
    f = MM.fit(hist, today)
    print(f"WNBA fit: {f.n_games:,} games, {len(f.teams)} teams, "
          f"home adv {f.home_adv:+.2f}, sigma {f.sigma_margin:.2f}\n")

    r = requests.get(f"{K}/markets", params={"series_ticker": "KXWNBAGAME",
                                             "status": "open", "limit": 100},
                     timeout=60)
    by_ev = defaultdict(list)
    for m in r.json().get("markets", []):
        by_ev[m["event_ticker"]].append(m)

    rows = []
    for ev, mk in by_ev.items():
        date = ticker_date(ev)
        legs = {}
        for m in mk:
            team = KALSHI_TO_BREF.get(str(m.get("yes_sub_title", "")).strip())
            if not team:
                print(f"  unmapped: {m.get('yes_sub_title')!r}")
                continue
            try:
                bid = float(m.get("yes_bid_dollars"))
                ask = float(m.get("yes_ask_dollars"))
            except (TypeError, ValueError):
                continue
            legs[team] = {"bid": bid, "ask": ask, "ticker": m["ticker"]}
        if len(legs) != 2:
            continue

        # Trailing code in the event ticker is the HOME side (see docstring).
        codes = re.sub(r"^.*-\d{2}[A-Z]{3}\d{2}", "", ev)
        teams = list(legs)
        # Resolve home/away from the schedule rather than trusting the ticker.
        cand = g[(g["date"] == pd.Timestamp(date)) &
                 (g["home_team"].isin(teams)) & (g["away_team"].isin(teams))]
        if cand.empty:
            continue
        home = cand.iloc[0]["home_team"]
        away = cand.iloc[0]["away_team"]

        p = MM.predict(f, home, away)
        if p is None:
            continue
        for team, side in ((home, "HOME"), (away, "AWAY")):
            leg = legs.get(team)
            if not leg:
                continue
            mp = p["p_home"] if side == "HOME" else p["p_away"]
            ask = leg["ask"]
            spr = ask - leg["bid"]
            dep = depth(leg["ticker"])
            ev_ = mp * (1 - ask) - (1 - mp) * ask - fee(ask)
            rows.append({
                "date": date, "match": f"{away} @ {home}", "team": team,
                "side": side, "model": mp, "bid": leg["bid"], "ask": ask,
                "spread": spr, "depth": dep, "ev": ev_,
                "exp_margin": p["exp_margin"], "exp_total": p["exp_total"],
                "tradeable": (spr <= args.max_spread and dep >= args.min_depth
                              and p["eff_n_min"] >= args.min_eff_n),
            })

    if not rows:
        print("no comparable WNBA markets")
        return
    d = pd.DataFrame(rows).sort_values(["date", "match"])
    out = ROOT / "reports" / f"wnba_kalshi_{today.date()}.csv"
    d.to_csv(out, index=False)

    print(f"{'date':<12}{'match':<42}{'pick':<24}{'model':>7}{'ask':>7}"
          f"{'EV':>8}  trade")
    for _, x in d.iterrows():
        print(f"{x['date']:<12}{x['match'][:41]:<42}{x['team'][:23]:<24}"
              f"{x['model']:>7.1%}{x['ask']:>7.2f}{x['ev']:>+8.1%}"
              f"  {'yes' if x['tradeable'] else 'no'}")

    # Two-way ask sums show Kalshi's effective spread on this sport.
    print(f"\n  two-way ask sums (1.00 = no spread):")
    for m_, gg in d.groupby("match"):
        if len(gg) == 2:
            print(f"    {m_[:44]:<46}{gg['ask'].sum():.3f}")

    good = d[d["tradeable"] & (d["ev"] >= args.min_ev)]
    print(f"\n  POSITIVE EV AND TRADEABLE (EV >= {args.min_ev:.0%})")
    if good.empty:
        print("    none")
    else:
        for _, x in good.sort_values("ev", ascending=False).iterrows():
            print(f"    {x['match'][:40]:<42}{x['team'][:20]:<22}"
                  f"model {x['model']:.1%} ask {x['ask']:.2f} "
                  f"EV {x['ev']:+.1%} depth ${x['depth']:,.0f}")
    print(f"\nsaved -> {out}")
    print("\n  The WNBA model has no historical market benchmark — Basketball")
    print("  Reference carries no odds. These comparisons are the first, and")
    print("  they accumulate into one as games settle.")


if __name__ == "__main__":
    main()
