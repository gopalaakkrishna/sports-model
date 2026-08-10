"""Predict matches between two leagues that never meet in the training data.

Leagues Cup pits MLS against Liga MX. The history used here contains ZERO
matches between the two, so the rating graphs are disconnected and the relative
strength of the leagues is *not identifiable* from the data. Fitting them
jointly does not solve this: with no connecting edges the ridge simply pulls
both leagues to a common mean, which silently imposes "the two leagues are
equally strong" — an assumption, not a finding.

So the offset is made an explicit input. `delta` is the net rating advantage of
league B over league A on the log-goal scale:

    delta = 0.0   the two leagues are equally strong
    delta > 0     league B (Liga MX here) is stronger

Predictions are reported across a range of delta so the sensitivity is visible.
If the answer flips within a plausible delta range, the honest output is "this
model cannot call this match", and that is a legitimate result.
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

# (home team, away team, home division) — home assigned by VENUE, not by the
# way the fixture happens to be listed.
LEAGUES_CUP_AUG6 = [
    ("New York City FC", "Santos Laguna", "USA:MLS", "Sports Illustrated Stadium"),
    ("Philadelphia Union", "Cruz Azul", "USA:MLS", "Subaru Park, Chester"),
    ("Chicago Fire FC", "Necaxa", "USA:MLS", "SeatGeek Stadium"),
    ("Austin FC", "Tijuana", "USA:MLS", "Q2 Stadium"),
    ("Portland Timbers", "Puebla", "USA:MLS", "Providence Park"),
    ("America", "San Diego FC", "MEX:Liga MX", "Estadio Azteca"),
]


def fit_group(hist, divs, as_of, xi, reg):
    sub = hist[hist["Div"].isin(divs)]
    return M.fit(sub, as_of, xi=xi, reg=reg)


def predict_cross(fr_a, fr_b, home, away, home_div, delta, div_a, div_b):
    """delta shifts league B's net rating relative to league A."""
    ia = fr_a.team_index()
    ib = fr_b.team_index()

    def rating(team):
        if team in ia:
            return fr_a.attack[ia[team]], fr_a.defence[ia[team]], "A"
        if team in ib:
            # Split the offset across attack and defence so `delta` is the
            # change in net rating (attack - defence).
            return (fr_b.attack[ib[team]] + delta / 2,
                    fr_b.defence[ib[team]] - delta / 2, "B")
        return None

    rh, ra = rating(home), rating(away)
    if rh is None or ra is None:
        return None

    # Home advantage from whichever league is hosting.
    fr_home = fr_a if home_div == div_a else fr_b
    di = fr_home.div_index()
    ha = fr_home.home_adv[di[home_div]] if home_div in di else float(fr_home.home_adv.mean())

    lam = float(np.exp(np.clip(rh[0] + ra[1] + ha, -10, 4)))
    mu = float(np.exp(np.clip(ra[0] + rh[1], -10, 4)))
    rho = (fr_a.rho + fr_b.rho) / 2
    m = M.score_matrix(lam, mu, rho)
    return {
        "lam": lam, "mu": mu,
        "p_home": float(np.tril(m, -1).sum()),
        "p_draw": float(np.trace(m)),
        "p_away": float(np.triu(m, 1).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xi", type=float, default=0.0018)
    ap.add_argument("--reg", type=float, default=2.0)
    ap.add_argument("--deltas", default="-0.3,-0.15,0,0.15,0.3")
    ap.add_argument("--match", nargs=2, metavar=("HOST", "VISITOR"), default=None,
                    help="one-off fixture; HOST is whoever actually plays at home")
    ap.add_argument("--host-league", default="USA:MLS",
                    choices=["USA:MLS", "MEX:Liga MX"],
                    help="league of the hosting side (sets which home advantage applies)")
    ap.add_argument("--venue", default="")
    args = ap.parse_args()

    div_a, div_b = "USA:MLS", "MEX:Liga MX"
    hist = D.load_history()
    hist = hist[hist["FTHG"].notna()]
    as_of = pd.Timestamp.now().normalize()

    fr_a = fit_group(hist, [div_a], as_of, args.xi, args.reg)
    fr_b = fit_group(hist, [div_b], as_of, args.xi, args.reg)
    print(f"MLS      : {fr_a.n_matches:,} matches, {len(fr_a.teams)} teams, "
          f"home adv {fr_a.home_adv[0]:+.3f}")
    print(f"Liga MX  : {fr_b.n_matches:,} matches, {len(fr_b.teams)} teams, "
          f"home adv {fr_b.home_adv[0]:+.3f}")
    print("\nCross-league matches in the training data: 0")
    print("The offset between the two leagues is NOT identifiable from this data.")
    print("Predictions below are shown across a range of assumed offsets.\n")

    resolver = TeamResolver(list(fr_a.teams) + list(fr_b.teams))
    deltas = [float(x) for x in args.deltas.split(",")]

    fixtures = (
        [(args.match[0], args.match[1], args.host_league, args.venue or "one-off")]
        if args.match else LEAGUES_CUP_AUG6
    )

    for home, away, home_div, venue in fixtures:
        h = resolver.resolve(home)
        a = resolver.resolve(away)
        if h is None or a is None:
            print(f"  could not resolve {home!r} / {away!r}")
            resolver.report()
            continue
        print(f"  {h} (host) vs {a}   [{venue}]")
        print(f"    {'offset':<26}{'H':>7}{'D':>7}{'A':>7}{'xG':>13}")
        calls = []
        for d in deltas:
            p = predict_cross(fr_a, fr_b, h, a, home_div, d, div_a, div_b)
            lbl = ("Liga MX stronger" if d > 0 else
                   "MLS stronger" if d < 0 else "leagues equal")
            print(f"    {f'{d:+.2f} ({lbl})':<26}"
                  f"{p['p_home']:>7.0%}{p['p_draw']:>7.0%}{p['p_away']:>7.0%}"
                  f"{f'{p['lam']:.2f}-{p['mu']:.2f}':>13}")
            calls.append(max(("H", p["p_home"]), ("D", p["p_draw"]),
                             ("A", p["p_away"]), key=lambda z: z[1])[0])
        stable = len(set(calls)) == 1
        print(f"    -> most likely outcome {'STABLE' if stable else 'FLIPS'} "
              f"across the offset range: {' '.join(calls)}\n")

    print("Reading this: where the call flips across plausible offsets, the model")
    print("genuinely cannot separate the teams and should not be used to pick a side.")
    print("Fixing it properly needs Leagues Cup / Champions Cup results in the")
    print("training data to connect the two rating graphs.")


if __name__ == "__main__":
    main()
