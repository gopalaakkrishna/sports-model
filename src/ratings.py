"""Current team ratings for a country, and on-demand matchup predictions.

Useful when a league is mid-season but the fixtures feed has not yet published
the next round. Ratings are as of today; the matchup predictor takes any two
teams in the group.

    python ratings.py --country USA
    python ratings.py --country USA --match "Inter Miami" "LA Galaxy"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data as D
import model as M
from team_names import TeamResolver

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True, help="e.g. USA, Argentina, England")
    ap.add_argument("--xi", type=float, default=0.0018)
    ap.add_argument("--reg", type=float, default=2.0)
    ap.add_argument("--match", nargs=2, metavar=("HOME", "AWAY"), default=None)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    hist = D.load_history()
    hist = hist[hist["FTHG"].notna()].copy()
    groups = D.country_groups(hist)
    if args.country not in groups:
        print(f"unknown country {args.country!r}. available: {sorted(groups)}")
        sys.exit(1)

    divs = groups[args.country]
    sub = hist[hist["Div"].isin(divs)]
    today = pd.Timestamp.now().normalize()
    fr = M.fit(sub, today, xi=args.xi, reg=args.reg)

    print(f"{args.country}: fitted {fr.n_matches:,} matches to {sub['Date'].max().date()}, "
          f"{len(fr.teams)} teams, effective sample {fr.eff_n:.0f}")
    for d, h in zip(fr.divisions, fr.home_adv):
        print(f"  home advantage {d}: {h:+.3f}  "
              f"(x{np.exp(h):.2f} on expected home goals)")

    if args.match:
        resolver = TeamResolver(fr.teams)
        home = resolver.resolve(args.match[0])
        away = resolver.resolve(args.match[1])
        if home is None or away is None:
            print("\ncould not resolve team names:")
            resolver.report()
            sys.exit(1)
        div = divs[0]
        p = M.predict(fr, home, away, div)
        print(f"\n  {home} vs {away}  ({div})")
        print(f"    expected goals   {p['lambda_home']:.2f} - {p['lambda_away']:.2f}")
        print(f"    home / draw / away  {p['p_home']:.1%} / {p['p_draw']:.1%} / {p['p_away']:.1%}")
        print(f"    over 2.5 goals   {p['p_over25']:.1%}")
        print(f"    both teams score {p['p_btts']:.1%}")
        print("    most likely scorelines:")
        for h, a, pr in p["top_scorelines"]:
            print(f"      {h}-{a}  {pr:.1%}")
        print("\n  Reminder: the backtest gives this model 0 weight against the")
        print("  closing line. Use as a view, not as a bet.")
        return

    # Rank by a neutral-venue net rating.
    net = fr.attack - fr.defence
    # Only show teams active in the last 18 months.
    recent = sub[sub["Date"] >= today - pd.Timedelta(days=548)]
    active = set(recent["HomeTeam"]) | set(recent["AwayTeam"])
    rows = [(fr.teams[i], fr.attack[i], fr.defence[i], net[i])
            for i in range(len(fr.teams)) if fr.teams[i] in active]
    rows.sort(key=lambda r: -r[3])

    print(f"\n  {'#':<4}{'team':<26}{'attack':>9}{'defence':>9}{'net':>8}")
    for k, (t, a, d, n) in enumerate(rows[:args.top], 1):
        print(f"  {k:<4}{t:<26}{a:>+9.3f}{d:>+9.3f}{n:>+8.3f}")
    print("\n  attack: higher scores more. defence: LOWER concedes fewer.")
    print("  net = attack - defence, a neutral-venue strength proxy.")


if __name__ == "__main__":
    main()
