"""Why does the model rate MLS above Liga MX when the market does not?

The gap comes from one input: 58 cross-league matches in the 2025 Leagues Cup.
This tests the ways that estimate could be biased.

  1. Venue imbalance — did MLS simply host most of the games?
  2. Selection — do the knockout rounds over-weight the MLS sides that advanced?
  3. Sample size — what is the confidence interval on the gap?
  4. Staleness — the bridge is 2025 only; squads have since changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data as D
import leagues_cup_data as LC
import model as M

MLS = {
    "Columbus Crew", "Houston Dynamo", "Los Angeles FC", "CF Montreal",
    "New York City", "San Diego FC", "Atlanta Utd", "Inter Miami",
    "Minnesota United", "Orlando City", "Portland Timbers", "Real Salt Lake",
    "FC Cincinnati", "Charlotte", "Colorado Rapids", "Los Angeles Galaxy",
    "New York Red Bulls", "Seattle Sounders", "Nashville SC", "FC Dallas",
}
KNOCKOUT_FROM = pd.Timestamp("2025-08-18")


def summarise(df: pd.DataFrame, label: str) -> None:
    cross = df[df["HomeTeam"].isin(MLS) != df["AwayTeam"].isin(MLS)]
    if cross.empty:
        print(f"  {label}: no cross-league matches")
        return
    mls_home = cross[cross["HomeTeam"].isin(MLS)]
    mls_away = cross[cross["AwayTeam"].isin(MLS)]

    def rec(sub, mls_is_home):
        gf = sub["FTHG"] if mls_is_home else sub["FTAG"]
        ga = sub["FTAG"] if mls_is_home else sub["FTHG"]
        w = int((gf > ga).sum()); d = int((gf == ga).sum()); l = int((gf < ga).sum())
        return w, d, l, float(gf.mean()), float(ga.mean())

    hw, hd, hl, hgf, hga = rec(mls_home, True)
    aw, ad, al, agf, aga = rec(mls_away, False)
    print(f"\n  {label}  ({len(cross)} cross-league matches)")
    print(f"    MLS at home  {len(mls_home):>3} games   {hw}W {hd}D {hl}L   "
          f"scored {hgf:.2f} conceded {hga:.2f}")
    print(f"    MLS away     {len(mls_away):>3} games   {aw}W {ad}D {al}L   "
          f"scored {agf:.2f} conceded {aga:.2f}")
    tot = len(mls_home) + len(mls_away)
    print(f"    share of games with MLS hosting: {len(mls_home)/tot:.0%}")
    print(f"    overall MLS record: {hw+aw}W {hd+ad}D {hl+al}L")
    # Neutral-ish read: goal difference per game, home and away averaged
    gd_home = hgf - hga
    gd_away = agf - aga
    print(f"    MLS goal difference: home {gd_home:+.2f}, away {gd_away:+.2f}, "
          f"venue-balanced mean {((gd_home + gd_away) / 2):+.2f}")


def fit_gap(bridge: pd.DataFrame, hist: pd.DataFrame) -> float:
    """League gap (mean MLS net rating minus mean Liga MX) from a given bridge."""
    sub = pd.concat([
        hist[hist["Div"].isin(["USA:MLS", "MEX:Liga MX"])],
        bridge,
    ], ignore_index=True)
    fr = M.fit(sub, pd.Timestamp.now().normalize(), xi=0.0018, reg=2.0)
    net = fr.attack - fr.defence
    mi = [i for i, t in enumerate(fr.teams) if t in MLS]
    xi_ = [i for i, t in enumerate(fr.teams) if t not in MLS]
    return float(np.mean(net[mi]) - np.mean(net[xi_]))


def main():
    lc = LC.load()
    hist = D.load_history(include_bridges=False)
    hist = hist[hist["FTHG"].notna()]

    print("=" * 68)
    print("1. VENUE IMBALANCE AND RECORD")
    print("=" * 68)
    summarise(lc, "all 2025 Leagues Cup")
    summarise(lc[lc["Date"] < KNOCKOUT_FROM], "group phase only")
    summarise(lc[lc["Date"] >= KNOCKOUT_FROM], "knockout phase only")

    print("\n" + "=" * 68)
    print("2. HOW MUCH OF THE GAP SURVIVES WITHOUT THE KNOCKOUTS?")
    print("=" * 68)
    full = fit_gap(lc, hist)
    group = fit_gap(lc[lc["Date"] < KNOCKOUT_FROM], hist)
    print(f"  gap using ALL Leagues Cup matches : {full:+.3f}")
    print(f"  gap using GROUP PHASE only        : {group:+.3f}")
    print(f"  knockout contribution             : {full - group:+.3f}")
    print("  (knockout rounds contain only the teams that advanced, so a large")
    print("   contribution here is a selection effect rather than league strength)")

    print("\n" + "=" * 68)
    print("3. HOW PRECISE IS THE ESTIMATE?")
    print("=" * 68)
    cross = lc[lc["HomeTeam"].isin(MLS) != lc["AwayTeam"].isin(MLS)]
    rng = np.random.default_rng(0)
    gaps = []
    for _ in range(60):
        samp = cross.sample(len(cross), replace=True, random_state=int(rng.integers(1e9)))
        try:
            gaps.append(fit_gap(samp, hist))
        except Exception:
            pass
    if gaps:
        lo, hi = np.percentile(gaps, [2.5, 97.5])
        print(f"  bootstrap over the {len(cross)} cross-league matches "
              f"({len(gaps)} resamples)")
        print(f"    point estimate {full:+.3f}")
        print(f"    95% CI         [{lo:+.3f}, {hi:+.3f}]")
        if lo < 0 < hi:
            print("    -> the CI includes ZERO: the data cannot distinguish the")
            print("       two leagues at this sample size.")

    print("\n" + "=" * 68)
    print("4. STALENESS")
    print("=" * 68)
    print(f"  bridge covers {lc['Date'].min().date()} .. {lc['Date'].max().date()}")
    print(f"  today is {pd.Timestamp.now().date()} — the estimate is a year old,")
    print("  across a full transfer cycle in both leagues.")


if __name__ == "__main__":
    main()
