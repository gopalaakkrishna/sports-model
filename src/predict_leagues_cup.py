"""Leagues Cup predictions from a CONNECTED MLS + Liga MX rating scale.

Earlier cross-league predictions rested on an unverifiable assumption: that the
two leagues are equally strong. With the 2025 Leagues Cup results included, the
graphs are joined by 58 genuine cross-league matches and the offset is estimated
from data.

Home advantage is fitted per division, so the Leagues Cup gets its own — cup
crowds and neutral-ish hosting differ from a league fixture.

Venue, not listing order, decides who is at home. That trap has already bitten
twice in this project (Cruz Azul "vs" Philadelphia was at Philadelphia's ground;
Monterrey "vs" Orlando is played in Orlando), and getting it backwards moves a
prediction by roughly 20 points.
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

MLS_TEAMS = {
    "Columbus Crew", "Houston Dynamo", "Los Angeles FC", "CF Montreal",
    "New York City", "San Diego FC", "Atlanta Utd", "Inter Miami",
    "Minnesota United", "Orlando City", "Portland Timbers", "Real Salt Lake",
    "FC Cincinnati", "Charlotte", "Colorado Rapids", "Los Angeles Galaxy",
    "New York Red Bulls", "Seattle Sounders", "Nashville SC", "FC Dallas",
    "Austin FC", "Chicago Fire", "Toronto FC", "Vancouver Whitecaps",
    "Sporting Kansas City", "St. Louis City", "Philadelphia Union",
    "New England Revolution", "DC United", "San Jose Earthquakes",
}

# Tonight's slate: (host, visitor, venue). Host verified by VENUE.
TONIGHT = [
    ("Inter Miami", "Atl. San Luis", "Nu Stadium, Miami"),
    ("Orlando City", "Monterrey", "Inter&Co Stadium, Orlando"),
    ("Nashville SC", "Club Leon", "Geodis Park, Nashville"),
    ("FC Dallas", "Queretaro", "Texas Health Mansfield Stadium"),
    ("Toluca", "Seattle Sounders", "Estadio Nemesio Diez, Toluca"),
    ("Los Angeles FC", "Guadalajara Chivas", "BMO Stadium, Los Angeles"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xi", type=float, default=0.0018)
    ap.add_argument("--reg", type=float, default=2.0)
    args = ap.parse_args()

    hist = D.load_history()
    hist = hist[hist["FTHG"].notna()].copy()
    groups = D.country_groups(hist)
    divs = groups.get("NorthAmerica")
    if not divs:
        print("NorthAmerica group unavailable — is the Leagues Cup bridge loaded?")
        sys.exit(1)

    sub = hist[hist["Div"].isin(divs)]
    today = pd.Timestamp.now().normalize()
    fr = M.fit(sub, today, xi=args.xi, reg=args.reg)

    print(f"CONNECTED FIT: {divs}")
    print(f"  {fr.n_matches:,} matches, {len(fr.teams)} teams, eff_n {fr.eff_n:.0f}")
    for d, h in zip(fr.divisions, fr.home_adv):
        print(f"  home advantage {d:<16} {h:+.4f}  (x{np.exp(h):.3f} on expected goals)")

    # Measured league gap: mean net rating of MLS clubs vs Liga MX clubs.
    net = fr.attack - fr.defence
    mls_idx = [i for i, t in enumerate(fr.teams) if t in MLS_TEAMS]
    mex_idx = [i for i, t in enumerate(fr.teams) if t not in MLS_TEAMS]
    gap = float(np.mean(net[mls_idx]) - np.mean(net[mex_idx]))
    print(f"\n  MEASURED league gap (MLS minus Liga MX net rating): {gap:+.3f}")
    print(f"    positive = MLS stronger. Previously this was ASSUMED to be 0.")
    print(f"    2025 Leagues Cup head-to-head was 29W 18D 15L to MLS.")

    print(f"\n  strongest across both leagues:")
    for k in np.argsort(-net)[:8]:
        tag = "MLS" if fr.teams[k] in MLS_TEAMS else "LigaMX"
        print(f"    {fr.teams[k]:<24}{tag:<8}net {net[k]:+.3f}  "
              f"eff_n {fr.team_eff_n[k]:.0f}")

    resolver = TeamResolver(fr.teams)
    lc_div = "LC:Leagues Cup"
    print(f"\n{'=' * 96}\nTONIGHT — Leagues Cup, connected ratings\n{'=' * 96}")
    print(f"{'match (host first)':<44}{'H':>6}{'D':>6}{'A':>6}{'xG':>13}"
          f"{'score':>7}{'eff_n':>8}")
    out = []
    for host, vis, venue in TONIGHT:
        h, a = resolver.resolve(host), resolver.resolve(vis)
        if h is None or a is None:
            print(f"  unresolved: {host!r} / {vis!r}")
            continue
        p = M.predict(fr, h, a, lc_div)
        if p is None:
            continue
        print(f"{(h + ' v ' + a)[:43]:<44}{p['p_home']:>6.0%}{p['p_draw']:>6.0%}"
              f"{p['p_away']:>6.0%}"
              f"{f'{p[chr(108)+chr(97)+chr(109)+chr(98)+chr(100)+chr(97)+chr(95)+chr(104)+chr(111)+chr(109)+chr(101)]:.2f}-{p['lambda_away']:.2f}':>13}"
              f"{p['top_scorelines'][0][0]}-{p['top_scorelines'][0][1]:>1}"
              f"{p['eff_n_min']:>8.0f}")
        out.append({"host": h, "visitor": a, "venue": venue,
                    "p_home": p["p_home"], "p_draw": p["p_draw"],
                    "p_away": p["p_away"], "xg_h": p["lambda_home"],
                    "xg_a": p["lambda_away"], "p_over25": p["p_over25"],
                    "p_btts": p["p_btts"],
                    "score": f"{p['top_scorelines'][0][0]}-{p['top_scorelines'][0][1]}",
                    "eff_n_min": p["eff_n_min"]})

    if out:
        df = pd.DataFrame(out)
        outp = ROOT / "reports" / f"leagues_cup_{today.date()}.csv"
        df.to_csv(outp, index=False)
        print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
