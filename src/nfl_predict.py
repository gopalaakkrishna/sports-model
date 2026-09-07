"""Price the upcoming NFL slate and compare against Kalshi.

Same shape as mlb_predict.py: fit the margin model as of today on nflverse
history, fetch Kalshi's KXNFLGAME markets, buy at the ask, net off Kalshi's
fee. The model itself (margin_model.py) and its backtest (margin_backtest.py,
nfl_qb_ablation.py) already existed and were validated — this is the piece
that was missing: nothing ever wired the model to a live Kalshi price. It sat
as a forward-testing script (margin_predict.py) writing a no-price CSV that
only dashboard.py's own summary view read.

The backtest result carries over unchanged: model log loss 0.6527 vs the
closing moneyline's 0.6125 (market better by +0.040), spread pick accuracy
50.4% against the 52.4% needed past the vig. This does not beat the market,
same as everything else in this project except the soccer GBM blend. It is
published for the same reason MLB and soccer are: to be measured against a
market benchmark it has never been checked against live, not because it is
expected to win.

Screens applied before anything is called an edge:
  * two-sided quote, spread <= max-spread, book depth >= min-depth
  * both teams above a minimum effective sample
  * the starting QBs known well enough to trust (qb_eff_n_min)
  * EV computed at the ASK, net of fees, never at the mid
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
import guards as G

ROOT = Path(__file__).resolve().parents[1]
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
FEE_RATE = 0.07

# Kalshi's two team-name surfaces disagree with each other, not just with
# nflverse. The per-market leg (yes_sub_title) is a clean, consistent city
# name across every event ("Miami", "San Francisco"). The per-EVENT title is
# NOT consistent: older events read "New England vs Seattle", newer ones read
# "MIA Dolphins vs SF 49ers" — same series, two formats, no warning. Pairing
# off `title` is what produced 13 "unmapped Kalshi teams" on the first run.
# The event's sub_title field is the one thing that stayed uniform throughout
# ("NE vs SEA (Sep 9)"), so pairing uses that instead — see _nfl_event_pairs.
#
# Two abbreviations differ from nflverse outright, not just in format:
# Kalshi's JAC/LAR are nflverse's JAX/LA.
_KALSHI_ABBR_TO_NFL = {"JAC": "JAX", "LAR": "LA"}


def _nfl_abbr(a: str) -> str:
    return _KALSHI_ABBR_TO_NFL.get(a, a)


# Kalshi names cities; nflverse uses standard team abbreviations. Built from
# a live diff of Kalshi's yes_sub_title values against nflverse's abbreviation
# set — the two disambiguate the same way (Kalshi splits "Los Angeles C"/"Los
# Angeles R" for the Chargers/Rams; nflverse uses LAC/LA), so this is a clean
# 32-team bijection with none of MLB's Chicago/LA ambiguity. This map is for
# per-market LEGS (yes_sub_title) only — event pairing uses sub_title and
# _KALSHI_ABBR_TO_NFL instead, per the note above.
KALSHI_TO_NFL = {
    "Arizona": "ARI", "Atlanta": "ATL", "Baltimore": "BAL", "Buffalo": "BUF",
    "Carolina": "CAR", "Chicago": "CHI", "Cincinnati": "CIN", "Cleveland": "CLE",
    "Dallas": "DAL", "Denver": "DEN", "Detroit": "DET", "Green Bay": "GB",
    "Houston": "HOU", "Indianapolis": "IND", "Jacksonville": "JAX",
    "Kansas City": "KC", "Las Vegas": "LV", "Los Angeles C": "LAC",
    "Los Angeles R": "LA", "Miami": "MIA", "Minnesota": "MIN",
    "New England": "NE", "New Orleans": "NO", "New York G": "NYG",
    "New York J": "NYJ", "Philadelphia": "PHI", "Pittsburgh": "PIT",
    "San Francisco": "SF", "Seattle": "SEA", "Tampa Bay": "TB",
    "Tennessee": "TEN", "Washington": "WAS",
}


def kalshi_fee(p: float) -> float:
    return FEE_RATE * p * (1.0 - p)


def _f(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def _ticker_date(event_ticker: str) -> str | None:
    """KXNFLGAME-26SEP09NESEA -> '2026-09-09'. Same shape as MLB's ticker."""
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", str(event_ticker))
    if not m:
        return None
    yy, mon, dd = m.group(1), m.group(2), m.group(3)
    if mon not in _MONTHS:
        return None
    return f"20{yy}-{_MONTHS[mon]:02d}-{int(dd):02d}"


def _nfl_event_pairs() -> dict[str, tuple[str, str]]:
    """{event_ticker: (away_nfl_abbr, home_nfl_abbr)}, already in nflverse form.

    Parses `sub_title` ("NE vs SEA (Sep 9)"), not `title` — see the module
    comment above KALSHI_TO_NFL for why `title` is unusable here. Returns
    nflverse abbreviations directly (via _nfl_abbr), so the caller needs no
    further name translation for this half of the pairing.
    """
    r = requests.get(f"{KALSHI}/events",
                     params={"series_ticker": "KXNFLGAME", "status": "open",
                             "limit": 200}, timeout=60)
    if r.status_code != 200:
        return {}
    evs = r.json().get("events", [])
    out = {}
    for e in evs:
        m = re.match(r"^\s*([A-Z]{2,3})\s+vs\.?\s+([A-Z]{2,3})\s*\(",
                     str(e.get("sub_title", "")))
        if m:
            out[str(e.get("event_ticker"))] = (_nfl_abbr(m.group(1)),
                                               _nfl_abbr(m.group(2)))
    return G.parsed_or_die(evs, out, what="NFL event pairs",
                           series="KXNFLGAME",
                           samples=[e.get("sub_title") for e in evs])


def kalshi_nfl_markets() -> dict[tuple[str, str, str], dict]:
    """{(date, home, away): {team: {bid, ask, ticker, ...}}} keyed on nflverse abbreviations."""
    r = requests.get(f"{KALSHI}/markets",
                     params={"series_ticker": "KXNFLGAME", "status": "open",
                             "limit": 200}, timeout=60)
    if r.status_code != 200:
        return {}
    pairs = _nfl_event_pairs()
    by_ev = defaultdict(list)
    for m in r.json().get("markets", []):
        by_ev[m.get("event_ticker")].append(m)

    out = {}
    for ev, mk in by_ev.items():
        pair = pairs.get(str(ev))
        if pair:
            # Already nflverse abbreviations — _nfl_event_pairs parsed
            # sub_title, not the inconsistent `title` field.
            away, home = pair
        else:
            # Defensive fallback only: `title` has two known formats for this
            # series ("New England vs Seattle" and "MIA Dolphins vs SF
            # 49ers"), so this branch is weaker than the primary path and may
            # itself miss on the second format. KALSHI_TO_NFL only covers the
            # first (city-name) form.
            title = str(mk[0].get("title", "")).replace(" Winner?", "").strip()
            if " vs " not in title:
                continue
            a_raw, b_raw = [s.strip() for s in title.split(" vs ", 1)]
            away, home = KALSHI_TO_NFL.get(a_raw), KALSHI_TO_NFL.get(b_raw)
        date = _ticker_date(ev)
        if not date:
            continue
        if not away or not home:
            # Only the title-fallback path can land here — the primary path
            # (_nfl_event_pairs) always returns a real abbreviation via
            # _nfl_abbr's identity fallback.
            print(f"  unmapped Kalshi teams for {ev}: {away!r} / {home!r}")
            continue
        legs = {}
        for m in mk:
            sub = str(m.get("yes_sub_title", "")).strip()
            team = KALSHI_TO_NFL.get(sub)
            if not team:
                continue
            legs[team] = {
                "ticker": m.get("ticker"),
                "bid": _f(m.get("yes_bid_dollars")),
                "ask": _f(m.get("yes_ask_dollars")),
                "liq": _f(m.get("liquidity_dollars"), 0.0) or 0.0,
                "oi": _f(m.get("open_interest_fp"), 0.0) or 0.0,
            }
        if len(legs) == 2:
            out[(date, home, away)] = legs
    # KALSHI_TO_NFL is a fixed 32-team map; min_ratio catches it going stale
    # wholesale (a relocation, a Kalshi relabelling) while tolerating one or
    # two misses, which already print above. See src/guards.py.
    return G.parsed_or_die(by_ev, out, what="NFL fixtures priced",
                           series="KXNFLGAME", min_ratio=0.5,
                           samples=[mk[0].get("title") for mk in by_ev.values()])


def depth(ticker: str, within: float = 0.05) -> float:
    try:
        r = requests.get(f"{KALSHI}/markets/{ticker}/orderbook",
                         params={"depth": 10}, timeout=45)
        book = r.json().get("orderbook") or r.json().get("orderbook_fp") or {}
    except Exception:
        return 0.0
    tot = 0.0
    for side in ("yes", "yes_dollars", "no", "no_dollars"):
        lv = book.get(side) or []
        px = [_f(l[0], 0.0) for l in lv if len(l) >= 2]
        if not px:
            continue
        best = max(px)
        for l in lv:
            if len(l) >= 2 and abs(_f(l[0], 0.0) - best) <= within:
                tot += _f(l[0], 0.0) * _f(l[1], 0.0)
    return tot


def main():
    ap = argparse.ArgumentParser()
    # NFL plays in clusters (Thu/Sun/Mon), not daily — a short window shows a
    # full slate one week and nothing the next. 10 days spans at least one
    # full game week from any day of the week.
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--xi", type=float, default=0.0025)
    ap.add_argument("--max-spread", type=float, default=0.06)
    ap.add_argument("--min-depth", type=float, default=500.0)
    ap.add_argument("--min-eff-n", type=float, default=8.0)
    # NFL's analogue of MLB's starting-pitcher gate: an unproven or unknown
    # starting QB falls back to league average, which is exactly wrong for
    # the single factor most likely to move a game.
    ap.add_argument("--min-qb-eff-n", type=float, default=6.0, dest="min_qb_eff_n")
    ap.add_argument("--min-ev", type=float, default=0.02)
    args = ap.parse_args()

    games = pd.read_parquet(ROOT / "data" / "raw" / "nfl_games.parquet")
    games["date"] = pd.to_datetime(games["date"])
    today = pd.Timestamp.now().normalize()
    f = MM.fit(games, today, xi=args.xi)
    print(f"fitted {f.n_games:,} games to {today.date()}, {len(f.teams)} teams\n")

    mkts = kalshi_nfl_markets()
    print(f"Kalshi NFL markets: {len(mkts)} fixtures\n")

    fut = games[(~games["played"]) & (games["date"] >= today) &
               (games["date"] <= today + pd.Timedelta(days=args.days))]

    rows = []
    for _, g in fut.iterrows():
        sp = g.get("spread_line")
        tl = g.get("total_line")
        p = MM.predict(f, g["home_team"], g["away_team"],
                       spread=float(sp) if pd.notna(sp) else None,
                       total_line=float(tl) if pd.notna(tl) else None,
                       home_qb=g.get("home_qb_name"), away_qb=g.get("away_qb_name"))
        if p is None:
            continue
        date = g["date"].strftime("%Y-%m-%d")
        legs = mkts.get((date, g["home_team"], g["away_team"]), {})
        for team, side in ((g["home_team"], "HOME"), (g["away_team"], "AWAY")):
            mp = p["p_home"] if side == "HOME" else p["p_away"]
            leg = legs.get(team)
            bid = ask = spr = dep = ev = np.nan
            trade = False
            if leg and leg["bid"] is not None and leg["ask"] is not None:
                bid, ask = leg["bid"], leg["ask"]
                spr = ask - bid
                dep = max(leg["liq"], leg["oi"], depth(leg["ticker"]))
                ev = mp * (1 - ask) - (1 - mp) * ask - kalshi_fee(ask)
                trade = (spr <= args.max_spread and dep >= args.min_depth
                         and p["eff_n_min"] >= args.min_eff_n
                         and p["qb_eff_n_min"] >= args.min_qb_eff_n)
            rows.append({
                "date": date, "start": str(g.get("date"))[11:16] or "",
                "match": f"{g['away_team']} @ {g['home_team']}",
                "side": side, "team": team,
                "model": mp, "bid": bid, "ask": ask, "spread": spr,
                "depth": dep, "ev": ev, "exp_total": p["exp_total"],
                "tradeable": trade,
                "eff_n_min": p["eff_n_min"], "qb_eff_n_min": p["qb_eff_n_min"],
            })

    if not rows:
        print("no games found")
        return
    df = pd.DataFrame(rows)
    out = ROOT / "reports" / f"nfl_predictions_{today.date()}.csv"
    df.to_csv(out, index=False)

    print(f"{'date':<11}{'match':<28}{'side':<6}{'model':>7}"
          f"{'bid':>6}{'ask':>6}{'EV':>8}")
    for _, r in df.iterrows():
        b = f"{r['bid']:.2f}" if np.isfinite(r["bid"]) else "  -"
        a = f"{r['ask']:.2f}" if np.isfinite(r["ask"]) else "  -"
        e = f"{r['ev']:+.1%}" if np.isfinite(r["ev"]) else "   -"
        print(f"{r['date']:<11}{r['match'][:27]:<28}{r['side']:<6}"
              f"{r['model']:>7.1%}{b:>6}{a:>6}{e:>8}")

    good = df[df["tradeable"] & (df["ev"] >= args.min_ev)]
    print(f"\n{'=' * 70}\nPOSITIVE EV AND TRADEABLE (EV >= {args.min_ev:.0%})\n{'=' * 70}")
    if good.empty:
        print("  none")
    else:
        for _, r in good.sort_values("ev", ascending=False).iterrows():
            print(f"  {r['match']:<32}{r['team']:<20} model {r['model']:.1%} "
                  f"ask {r['ask']:.2f}  EV {r['ev']:+.1%}  depth ${r['depth']:,.0f}")
    print(f"\nsaved -> {out}")
    print("\n  Backtest: model log loss 0.6527 vs the closing moneyline's 0.6125")
    print("  (market better by +0.040). This does not beat the market, same as")
    print("  everything here except the soccer GBM blend. Published to be")
    print("  measured against Kalshi live, which margin_backtest.py never was.")


if __name__ == "__main__":
    main()
