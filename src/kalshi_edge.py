"""Compare the soccer model against Kalshi's three-way game markets.

Kalshi lists each fixture as three binary markets (Home / Tie / Away), which
maps directly onto the 1X2 model. This finds the open fixtures, prices them with
the model as of today, and reports where the two disagree.

Three things this does that a naive edge scanner does not:

* **Buys at the ask.** You cannot transact at the mid. EV is computed against
  the price you would actually pay.
* **Charges Kalshi's fee.** Kalshi takes 0.07 * P * (1-P) per contract, which is
  1.75c at a 50c price. Typical model edges are of the same order, so ignoring
  the fee turns losing bets into apparent winners.
* **Screens for tradeability** before reporting anything, so an empty book
  cannot masquerade as a huge edge.

The three-way structure also gives a free diagnostic: the three asks should sum
to a little over 1. How far over is Kalshi's effective spread on that fixture.
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
import data as D
import model as M
from team_names import TeamResolver

ROOT = Path(__file__).resolve().parents[1]
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi soccer series -> (country group to FIT on, division to price in).
#
# The fit must use the whole country group, never the single division. Pooling
# all of a country's tiers is what gives promoted and relegated clubs real
# ratings: fitting Bundesliga 2 alone would rate a side just down from the top
# flight purely on its handful of second-tier games. The division is still
# needed separately, because home advantage is fitted per division.
SERIES_COUNTRY = {
    "KXLALIGAGAME": ("Spain", "SP1"),
    "KXEPLGAME": ("England", "E0"),
    "KXSERIEAGAME": ("Italy", "I1"),
    "KXBUNDESLIGAGAME": ("Germany", "D1"),
    "KXBUNDESLIGA2GAME": ("Germany", "D2"),
    "KXLIGUE1GAME": ("France", "F1"),
    "KXMLSGAME": ("USA", "USA:MLS"),
    "KXLIGAMXGAME": ("Mexico", "MEX:Liga MX"),
    # Added after surveying Kalshi for actual liquidity. These four are quoted
    # with real orderbook depth and the model already has their history — pure
    # coverage gain, no new data required. They are also in season during the
    # European summer, when the big leagues are dark.
    "KXALLSVENSKANGAME": ("Sweden", "SWE:Allsvenskan"),
    "KXSCOTTISHPREMGAME": ("Scotland", "SC0"),
    "KXJLEAGUEGAME": ("Japan", "JPN:J1 League"),
    "KXELITESERIENGAME": ("Norway", "NOR:Eliteserien"),
}

FEE_RATE = 0.07  # Kalshi trading fee coefficient


def kalshi_fee(price: float) -> float:
    """Fee per $1 contract at a given price."""
    return FEE_RATE * price * (1.0 - price)


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def fetch_fixtures(series: str) -> list[dict]:
    """Group Kalshi's per-outcome markets back into fixtures."""
    r = requests.get(f"{KALSHI}/markets",
                     params={"series_ticker": series, "status": "open", "limit": 200},
                     timeout=60)
    if r.status_code != 200:
        return []
    by_event = defaultdict(list)
    for m in r.json().get("markets", []):
        by_event[m.get("event_ticker")].append(m)

    fixtures = []
    for ev, mk in by_event.items():
        title = str(mk[0].get("title", ""))
        mt = re.match(r"^(.*?)\s+vs\.?\s+(.*?)\s+Winner\?*$", title)
        if not mt:
            continue
        home, away = mt.group(1).strip(), mt.group(2).strip()
        when = mk[0].get("occurrence_datetime") or mk[0].get("expected_expiration_time")
        legs = {}
        for m in mk:
            sub = str(m.get("yes_sub_title", "")).strip()
            key = ("DRAW" if sub.lower() in ("tie", "draw")
                   else "HOME" if sub == home else "AWAY" if sub == away else sub)
            legs[key] = {
                "ticker": m.get("ticker"),
                "bid": _f(m.get("yes_bid_dollars")),
                "ask": _f(m.get("yes_ask_dollars")),
                "liq": _f(m.get("liquidity_dollars"), 0.0),
                "oi": _f(m.get("open_interest_fp"), 0.0),
            }
        if {"HOME", "DRAW", "AWAY"} <= set(legs):
            fixtures.append({"event": ev, "home": home, "away": away,
                             "when": when, "legs": legs})
    return fixtures


def orderbook_depth(ticker: str, within: float = 0.05) -> float:
    try:
        r = requests.get(f"{KALSHI}/markets/{ticker}/orderbook",
                         params={"depth": 10}, timeout=45)
        if r.status_code != 200:
            return 0.0
        book = r.json().get("orderbook") or r.json().get("orderbook_fp") or {}
    except requests.RequestException:
        return 0.0
    total = 0.0
    for side in ("yes", "yes_dollars", "no", "no_dollars"):
        levels = book.get(side) or []
        px = [_f(l[0], 0.0) for l in levels if len(l) >= 2]
        if not px:
            continue
        best = max(px)
        for l in levels:
            if len(l) >= 2 and abs(_f(l[0], 0.0) - best) <= within:
                total += _f(l[0], 0.0) * _f(l[1], 0.0)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xi", type=float, default=0.0018)
    ap.add_argument("--reg", type=float, default=2.0)
    ap.add_argument("--max-spread", type=float, default=0.06)
    ap.add_argument("--min-depth", type=float, default=500.0)
    ap.add_argument("--min-ev", type=float, default=0.02)
    ap.add_argument("--min-eff-n", type=float, default=15.0,
                    help="minimum time-weighted matches behind BOTH teams' ratings")
    args = ap.parse_args()

    hist = D.load_history()
    hist = hist[hist["FTHG"].notna()].copy()
    today = pd.Timestamp.now().normalize()

    groups = D.country_groups(hist)
    fits: dict[str, M.FitResult] = {}
    rows = []

    for series, (country, price_div) in SERIES_COUNTRY.items():
        fx = fetch_fixtures(series)
        if not fx:
            continue
        divs = groups.get(country)
        if not divs:
            print(f"  {series}: no history for country {country}")
            continue
        sub = hist[hist["Div"].isin(divs)]
        if sub.empty:
            continue
        if country not in fits:
            try:
                fits[country] = M.fit(sub, today, xi=args.xi, reg=args.reg)
            except ValueError as e:
                print(f"  {series}: cannot fit ({e})")
                continue
        fr = fits[country]
        resolver = TeamResolver(fr.teams)
        print(f"\n{series}: {len(fx)} open fixtures, fitted on {country} "
              f"{divs} ({fr.n_matches:,} matches, {len(fr.teams)} teams), "
              f"pricing in {price_div}")

        for f in fx:
            h = resolver.resolve(f["home"])
            a = resolver.resolve(f["away"])
            if h is None or a is None:
                print(f"  unresolved: {f['home']!r} v {f['away']!r}")
                continue
            pred = M.predict(fr, h, a, price_div)
            if pred is None:
                continue
            model = {"HOME": pred["p_home"], "DRAW": pred["p_draw"], "AWAY": pred["p_away"]}

            asks = [f["legs"][k]["ask"] for k in ("HOME", "DRAW", "AWAY")]
            book_sum = sum(x for x in asks if x is not None)

            for k in ("HOME", "DRAW", "AWAY"):
                leg = f["legs"][k]
                bid, ask = leg["bid"], leg["ask"]
                if bid is None or ask is None:
                    continue
                spread = ask - bid
                depth = max(leg["liq"], leg["oi"], orderbook_depth(leg["ticker"]))
                p = model[k]
                # Buy at the ask; profit is (1 - ask) less fee, loss is the ask.
                fee = kalshi_fee(ask)
                ev = p * (1.0 - ask) - (1.0 - p) * ask - fee
                enough_data = pred["eff_n_min"] >= args.min_eff_n
                tradeable = (spread <= args.max_spread
                             and depth >= args.min_depth
                             and enough_data)
                rows.append({
                    "series": series, "event": f["event"], "when": str(f["when"])[:16],
                    "match": f"{h} v {a}", "leg": k,
                    "model": p, "bid": bid, "ask": ask, "spread": spread,
                    "depth": depth, "fee": fee, "ev": ev,
                    "eff_n_min": pred["eff_n_min"],
                    "book_sum": book_sum, "tradeable": tradeable,
                    "thin_data": not enough_data,
                })

    if not rows:
        print("\nno comparable fixtures found")
        return

    df = pd.DataFrame(rows)
    out = ROOT / "reports" / f"kalshi_edge_{today.date()}.csv"
    df.to_csv(out, index=False)

    print(f"\n{'=' * 100}\nALL COMPARISONS ({len(df)} legs, "
          f"{df['event'].nunique()} fixtures)\n{'=' * 100}")
    print(f"{'match':<34}{'leg':<6}{'model':>7}{'bid':>6}{'ask':>6}"
          f"{'spr':>6}{'depth':>9}{'EV':>8}  trade")
    for _, r in df.sort_values(["match", "leg"]).iterrows():
        print(f"{r['match'][:33]:<34}{r['leg']:<6}{r['model']:>7.1%}"
              f"{r['bid']:>6.2f}{r['ask']:>6.2f}{r['spread']:>6.2f}"
              f"{r['depth']:>9,.0f}{r['ev']:>+8.1%}  {'yes' if r['tradeable'] else 'no'}")

    print(f"\n  Kalshi three-way ask sums (1.00 = no spread at all):")
    for ev_, g in df.groupby("event"):
        print(f"    {g['match'].iloc[0][:36]:<38} {g['book_sum'].iloc[0]:.3f}")

    thin = df[df["thin_data"] & (df["ev"] >= args.min_ev)]
    if not thin.empty:
        print(f"\n  SUPPRESSED — thin data (eff_n < {args.min_eff_n:.0f} "
              f"weighted matches behind a team's rating):")
        for _, r in thin.sort_values("ev", ascending=False).iterrows():
            print(f"    {r['match'][:33]:<34}{r['leg']:<6} model {r['model']:>6.1%} "
                  f"ask {r['ask']:.2f}  apparent EV {r['ev']:+.1%}  "
                  f"eff_n {r['eff_n_min']:.1f}")
        print("    A promoted club with a handful of games can land a mid-table")
        print("    rating on luck of the draw. These are model error, not edge.")

    good = df[df["tradeable"] & (df["ev"] >= args.min_ev)]
    print(f"\n{'=' * 100}\nPOSITIVE EV AND TRADEABLE (EV >= {args.min_ev:.0%})\n{'=' * 100}")
    if good.empty:
        print("  none")
    else:
        for _, r in good.sort_values("ev", ascending=False).iterrows():
            print(f"  {r['match'][:33]:<34}{r['leg']:<6} model {r['model']:>6.1%} "
                  f"ask {r['ask']:.2f}  EV {r['ev']:+.1%}  depth ${r['depth']:,.0f}"
                  f"  eff_n {r['eff_n_min']:.0f}")
    print(f"\nsaved -> {out}")
    print("\n  Backtest context: this model loses to sharp closing lines by 0.017")
    print("  log loss and betting its disagreements lost 4.6-7.4%. Any EV above")
    print("  is a hypothesis to be logged in the ledger and scored, not a signal.")


if __name__ == "__main__":
    main()
