"""Price today's MLB slate and compare against Kalshi.

Pulls the schedule (with probable starters) from MLB StatsAPI, fits the model as
of today, applies the Platt calibration fitted out-of-sample, and compares to
Kalshi's game-winner markets — buying at the ask and netting off Kalshi's fee.

Screens applied before anything is called an edge:
  * two-sided quote, spread <= max-spread, book depth >= min-depth
  * both teams above a minimum effective sample
  * EV computed at the ASK, net of fees, never at the mid
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
import mlb_model as MM

ROOT = Path(__file__).resolve().parents[1]
STATS = "https://statsapi.mlb.com/api/v1"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
FEE_RATE = 0.07

# Kalshi abbreviates; LA/NY/Chicago need the league initial to disambiguate.
KALSHI_TO_MLB = {
    "Los Angeles A": "Los Angeles Angels", "Los Angeles D": "Los Angeles Dodgers",
    "New York Y": "New York Yankees", "New York M": "New York Mets",
    "Chicago C": "Chicago Cubs", "Chicago W": "Chicago White Sox",
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


def kalshi_fee(p: float) -> float:
    return FEE_RATE * p * (1.0 - p)


def _f(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def schedule(date: str) -> list[dict]:
    r = requests.get(f"{STATS}/schedule",
                     params={"sportId": 1, "date": date,
                             "hydrate": "team,probablePitcher,venue"}, timeout=60)
    r.raise_for_status()
    out = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            if g.get("gameType") != "R":
                continue
            t = g["teams"]
            out.append({
                "date": d["date"],
                "home": t["home"]["team"]["name"], "away": t["away"]["team"]["name"],
                "home_sp": (t["home"].get("probablePitcher") or {}).get("fullName"),
                "away_sp": (t["away"].get("probablePitcher") or {}).get("fullName"),
                "venue": (g.get("venue") or {}).get("name"),
                "start": g.get("gameDate", "")[11:16],
            })
    return out


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def _ticker_date(event_ticker: str) -> str | None:
    """KXMLBGAME-26AUG051840WSHPHI -> '2026-08-05'.

    The date MUST be part of the key. Teams play multi-game series, so keying on
    the pair alone maps every game of a series onto one market — silently
    scoring Thursday's model against Wednesday's price.
    """
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", str(event_ticker))
    if not m:
        return None
    yy, mon, dd = m.group(1), m.group(2), m.group(3)
    if mon not in _MONTHS:
        return None
    return f"20{yy}-{_MONTHS[mon]:02d}-{int(dd):02d}"


def kalshi_mlb_markets() -> dict[tuple[str, str, str], dict]:
    """{(date, home, away): {team: {bid, ask, ticker, ...}}} keyed on MLB names."""
    r = requests.get(f"{KALSHI}/markets",
                     params={"series_ticker": "KXMLBGAME", "status": "open",
                             "limit": 200}, timeout=60)
    if r.status_code != 200:
        return {}
    by_ev = defaultdict(list)
    for m in r.json().get("markets", []):
        by_ev[m.get("event_ticker")].append(m)

    out = {}
    for ev, mk in by_ev.items():
        title = str(mk[0].get("title", "")).replace(" Winner?", "").strip()
        if " vs " not in title:
            continue
        date = _ticker_date(ev)
        if not date:
            continue
        a_raw, b_raw = [s.strip() for s in title.split(" vs ", 1)]
        # Kalshi lists AWAY vs HOME for baseball, matching the usual convention.
        away = KALSHI_TO_MLB.get(a_raw)
        home = KALSHI_TO_MLB.get(b_raw)
        if not away or not home:
            print(f"  unmapped Kalshi teams: {a_raw!r} / {b_raw!r}")
            continue
        legs = {}
        for m in mk:
            sub = str(m.get("yes_sub_title", "")).strip()
            team = KALSHI_TO_MLB.get(sub)
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
    return out


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
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--xi", type=float, default=0.0025)
    ap.add_argument("--max-spread", type=float, default=0.06)
    ap.add_argument("--min-depth", type=float, default=500.0)
    ap.add_argument("--min-eff-n", type=float, default=40.0)
    # A debut or rarely-seen starter falls back to league average, so the model
    # is blind to exactly the factor that matters most in baseball. Same trap as
    # the promoted-club problem in soccer, applied to pitchers.
    ap.add_argument("--min-sp-eff-n", type=float, default=8.0,
                    dest="min_sp_eff_n",
                    help="minimum weighted starts behind BOTH starting pitchers")
    ap.add_argument("--min-ev", type=float, default=0.02)
    ap.add_argument("--raw", action="store_true", help="skip calibration")
    args = ap.parse_args()

    cal_path = ROOT / "data" / "processed" / "mlb_calibration.json"
    cal = (1.0, 0.0)
    if cal_path.exists() and not args.raw:
        j = json.loads(cal_path.read_text())
        cal = (j["a"], j["b"])
    print(f"calibration: a={cal[0]:.3f} b={cal[1]:+.3f}"
          f"{'  (RAW)' if args.raw else ''}")

    games = pd.read_parquet(ROOT / "data" / "raw" / "mlb_games.parquet")
    today = pd.Timestamp.now().normalize()
    f = MM.fit(games, today, xi=args.xi, calib=cal)
    print(f"fitted {f.n_games:,} games to {today.date()}, "
          f"{len(f.teams)} teams, {len(f.pitchers)} pitchers, "
          f"home adv {f.home_adv:+.4f}\n")

    mkts = kalshi_mlb_markets()
    print(f"Kalshi MLB markets: {len(mkts)} fixtures\n")

    rows = []
    for k in range(args.days):
        day = (today + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
        for g in schedule(day):
            p = MM.predict(f, g["home"], g["away"], g["home_sp"], g["away_sp"],
                           g["venue"])
            if p is None:
                continue
            legs = mkts.get((g["date"], g["home"], g["away"]), {})
            for team, side in ((g["home"], "HOME"), (g["away"], "AWAY")):
                mp = p["p_home"] if side == "HOME" else p["p_away"]
                leg = legs.get(team)
                bid = ask = spr = dep = ev = np.nan
                trade = False
                if leg and leg["bid"] is not None and leg["ask"] is not None:
                    bid, ask = leg["bid"], leg["ask"]
                    spr = ask - bid
                    dep = max(leg["liq"], leg["oi"], depth(leg["ticker"]))
                    ev = mp * (1 - ask) - (1 - mp) * ask - kalshi_fee(ask)
                    thin_sp = min(p["eff_n_sp_home"], p["eff_n_sp_away"]) < args.min_sp_eff_n
                    trade = (spr <= args.max_spread and dep >= args.min_depth
                             and min(p["eff_n_home"], p["eff_n_away"]) >= args.min_eff_n
                             and not thin_sp)
                rows.append({
                    "date": g["date"], "start": g["start"],
                    "match": f"{g['away']} @ {g['home']}", "side": side, "team": team,
                    "model": mp, "raw_model": (p["p_home_uncalibrated"] if side == "HOME"
                                               else 1 - p["p_home_uncalibrated"]),
                    "bid": bid, "ask": ask, "spread": spr, "depth": dep, "ev": ev,
                    "exp_runs": p["exp_total"], "tradeable": trade,
                    "sp": g["home_sp"] if side == "HOME" else g["away_sp"],
                    "sp_eff_n_min": min(p["eff_n_sp_home"], p["eff_n_sp_away"]),
                    "team_eff_n_min": min(p["eff_n_home"], p["eff_n_away"]),
                })

    if not rows:
        print("no games found")
        return
    df = pd.DataFrame(rows)
    out = ROOT / "reports" / f"mlb_predictions_{today.date()}.csv"
    df.to_csv(out, index=False)

    print(f"{'date':<11}{'match':<44}{'side':<6}{'model':>7}{'raw':>7}"
          f"{'bid':>6}{'ask':>6}{'EV':>8}{'runs':>7}")
    for _, r in df.iterrows():
        b = f"{r['bid']:.2f}" if np.isfinite(r["bid"]) else "  -"
        a = f"{r['ask']:.2f}" if np.isfinite(r["ask"]) else "  -"
        e = f"{r['ev']:+.1%}" if np.isfinite(r["ev"]) else "   -"
        print(f"{r['date']:<11}{r['match'][:43]:<44}{r['side']:<6}"
              f"{r['model']:>7.1%}{r['raw_model']:>7.1%}{b:>6}{a:>6}{e:>8}"
              f"{r['exp_runs']:>7.2f}")

    good = df[df["tradeable"] & (df["ev"] >= args.min_ev)]
    print(f"\n{'=' * 92}\nPOSITIVE EV AND TRADEABLE (EV >= {args.min_ev:.0%})\n{'=' * 92}")
    if good.empty:
        print("  none")
    else:
        for _, r in good.sort_values("ev", ascending=False).iterrows():
            print(f"  {r['match'][:43]:<45}{r['team'][:20]:<22} model {r['model']:.1%} "
                  f"ask {r['ask']:.2f}  EV {r['ev']:+.1%}  depth ${r['depth']:,.0f}")
    print(f"\nsaved -> {out}")
    print("\n  Backtest: calibrated model beats the base rate by 0.0089 log loss.")
    print("  It has NOT been measured against a market benchmark historically —")
    print("  no MLB closing-odds history is loaded yet. Until that exists, these")
    print("  EVs are untested hypotheses. Log them, score them, then decide.")


if __name__ == "__main__":
    main()
