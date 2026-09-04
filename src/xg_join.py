"""Join Understat xG onto the football-data match table.

WHY AN EXPLICIT ALIAS MAP AND NOT FUZZY MATCHING

TeamResolver's fuzzy matcher resolved only 66% of Understat names, and the
failures were not random — they were the biggest clubs in each league
(Manchester City, Manchester United, Borussia Dortmund, Bayer Leverkusen,
AC Milan, Paris Saint Germain). Understat writes full formal club names while
football-data writes terse ones, and the two diverge most for clubs whose
formal name carries a city or sponsor prefix.

Dropping 34% of matches would be survivable. Dropping a third of matches that
is disproportionately the strong teams is not: it would bias every downstream
comparison toward mid-table fixtures, where the model and the market agree
most, and quietly flatter any result computed on the remainder.

So the mapping is explicit and auditable, and `join()` reports what it could
not resolve rather than silently dropping it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

XG = ROOT / "data" / "raw" / "understat_xg.csv"
RAW = ROOT / "data" / "raw" / "football_data_raw.parquet"

# Understat league -> football-data division code
LEAGUE_DIV = {"EPL": "E0", "La_liga": "SP1", "Bundesliga": "D1",
              "Serie_A": "I1", "Ligue_1": "F1"}

# Understat name -> football-data name. Only entries that actually differ.
ALIAS = {
    # England
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Sheffield United": "Sheffield United",
    "Leeds United": "Leeds",
    "Ipswich Town": "Ipswich",
    "Luton": "Luton",
    "West Bromwich Albion": "West Brom",
    # Spain
    "Athletic Club": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid",
    "Real Betis": "Betis",
    "Real Valladolid": "Valladolid",
    "Real Sociedad": "Sociedad",
    "Espanyol": "Espanol",
    "Rayo Vallecano": "Vallecano",
    "Celta Vigo": "Celta",
    "Deportivo La Coruna": "Dep. A Coruna",
    "Racing Santander": "Santander",
    "Real Oviedo": "Oviedo",
    # Germany
    "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia M.Gladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Cologne": "FC Koln",
    "Hertha Berlin": "Hertha",
    "Mainz 05": "Mainz",
    "RasenBallsport Leipzig": "RB Leipzig",
    "VfB Stuttgart": "Stuttgart",
    "VfL Wolfsburg": "Wolfsburg",
    "VfL Bochum": "Bochum",
    "FC Heidenheim": "Heidenheim",
    "SC Freiburg": "Freiburg",
    "FC Augsburg": "Augsburg",
    "Hamburger SV": "Hamburg",
    "TSG Hoffenheim": "Hoffenheim",
    "St. Pauli": "St Pauli",
    "Werder Bremen": "Werder Bremen",
    "Schalke 04": "Schalke 04",
    # Italy
    "AC Milan": "Milan",
    "Parma Calcio 1913": "Parma",
    "Hellas Verona": "Verona",
    # France
    "Paris Saint Germain": "Paris SG",
    "Saint-Etienne": "St Etienne",
    "Clermont Foot": "Clermont",
    "Stade Brestois 29": "Brest",
    "Olympique Lyonnais": "Lyon",
    "Olympique Marseille": "Marseille",
    "Stade Rennais": "Rennes",
    "RC Lens": "Lens",
    "FC Nantes": "Nantes",
    "Paris FC": "Paris FC",
}


def _norm(s: str) -> str:
    return str(s).strip()


def join(verbose: bool = True) -> pd.DataFrame:
    """Return football-data matches with xg_h / xg_a attached where available."""
    xg = pd.read_csv(XG, parse_dates=["date"])
    raw = pd.read_parquet(RAW)
    raw["Date"] = pd.to_datetime(raw["Date"])

    xg["Div"] = xg["league"].map(LEAGUE_DIV)
    xg["HomeTeam"] = xg["home"].map(lambda s: ALIAS.get(_norm(s), _norm(s)))
    xg["AwayTeam"] = xg["away"].map(lambda s: ALIAS.get(_norm(s), _norm(s)))

    # Understat's date is the local kickoff date; football-data's occasionally
    # differs by a day for late kickoffs. Join on the exact date first, then
    # retry the misses at +/-1 day. Both team names must still match exactly,
    # and two clubs do not meet twice inside 24 hours, so the widening cannot
    # attach a result to the wrong fixture.
    keys = ["Div", "HomeTeam", "AwayTeam"]
    out = []
    for off in (0, 1, -1):
        if xg.empty:
            break
        probe = xg.copy()
        probe["Date"] = probe["date"] + pd.Timedelta(days=off)
        m = raw.merge(probe[keys + ["Date", "xg_h", "xg_a"]],
                      on=keys + ["Date"], how="inner")
        if len(m):
            out.append(m)
            done = set(zip(probe["Div"], probe["HomeTeam"], probe["AwayTeam"],
                           probe["date"]))
            hit = set(zip(m["Div"], m["HomeTeam"], m["AwayTeam"],
                          m["Date"] - pd.Timedelta(days=off)))
            xg = xg[~pd.Series(list(zip(xg["Div"], xg["HomeTeam"],
                                        xg["AwayTeam"], xg["date"])),
                               index=xg.index).isin(hit)]
    joined = pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    if verbose:
        total = len(pd.read_csv(XG))
        print(f"xG rows: {total:,}   joined to football-data: {len(joined):,} "
              f"({len(joined) / total:.1%})")
        if len(xg):
            miss = sorted(set(xg["home"]) | set(xg["away"]))
            print(f"  unresolved after aliasing ({len(xg)} rows): {miss[:12]}")
        if len(joined):
            print(f"  date range {joined['Date'].min().date()} .. "
                  f"{joined['Date'].max().date()}")
            print(joined.groupby("Div").size().to_string())
    return joined


if __name__ == "__main__":
    join()
