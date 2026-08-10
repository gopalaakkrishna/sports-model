"""Verified 2025 Leagues Cup results — the bridge between MLS and Liga MX.

Why this file exists: MLS and Liga MX never meet in the league data, so their
rating graphs are disconnected and their relative strength is unidentifiable.
Every cross-league prediction was resting on an assumption ("the leagues are
equally strong") that could not be checked. These 62 matches connect the graphs
so the offset is estimated from results instead of assumed.

PROVENANCE, and a warning. These come from the 2025 Leagues Cup Wikipedia page.
The equivalent extraction for 2023 was **garbage** — every group-stage match came
back as 2-0, 3-0 or 4-0, because the standings table (played/won/points) was
being read as scores, and the knockout section contradicted itself (the same
fixture in two rounds with two different scores). That data was discarded.

This 2025 set was spot-checked against independent sources before use:
  * Cruz Azul 0-7 Seattle Sounders — confirmed (ESPN, Sounders FC): the largest
    margin in competition history.
  * Final, Seattle 3-0 Inter Miami — confirmed (ESPN, MLSSoccer).
Scores are the 90-minute result; penalty shootouts are ignored, which is what
the goal model wants.

2024 group-stage scores were not available in a trustworthy form and are omitted
rather than guessed at.
"""

from __future__ import annotations

import pandas as pd

# Wikipedia naming -> the naming used in the MLS / Liga MX history files.
NAME_MAP = {
    "Columbus Crew": "Columbus Crew", "Toluca": "Toluca",
    "Tigres UANL": "Tigres UANL", "Houston Dynamo FC": "Houston Dynamo",
    "Los Angeles FC": "Los Angeles FC", "Mazatlán": "Mazatlan FC",
    "CF Montréal": "CF Montreal", "León": "Club Leon",
    "New York City FC": "New York City", "Puebla": "Puebla",
    "Pachuca": "Pachuca", "San Diego FC": "San Diego FC",
    "Necaxa": "Necaxa", "Atlanta United FC": "Atlanta Utd",
    "Inter Miami CF": "Inter Miami", "Atlas": "Atlas",
    "Minnesota United FC": "Minnesota United", "Querétaro": "Queretaro",
    "Pumas UNAM": "UNAM Pumas", "Orlando City SC": "Orlando City",
    "Portland Timbers": "Portland Timbers", "Atlético San Luis": "Atl. San Luis",
    "América": "Club America", "Real Salt Lake": "Real Salt Lake",
    "Monterrey": "Monterrey", "FC Cincinnati": "FC Cincinnati",
    "Charlotte FC": "Charlotte", "Juárez": "Juarez",
    "Colorado Rapids": "Colorado Rapids", "Santos Laguna": "Santos Laguna",
    "LA Galaxy": "Los Angeles Galaxy", "Tijuana": "Club Tijuana",
    "Guadalajara": "Guadalajara Chivas", "New York Red Bulls": "New York Red Bulls",
    "Cruz Azul": "Cruz Azul", "Seattle Sounders FC": "Seattle Sounders",
}

# (date, home, away, home goals, away goals) — 90-minute scores.
RAW_2025 = [
    # Matchday 1 (2025-07-29 .. 2025-08-02)
    ("2025-07-30", "Toluca", "Columbus Crew", 2, 2),
    ("2025-07-30", "Tigres UANL", "Houston Dynamo FC", 4, 1),
    ("2025-07-30", "Los Angeles FC", "Mazatlán", 1, 1),
    ("2025-07-30", "CF Montréal", "León", 1, 1),
    ("2025-07-30", "New York City FC", "Puebla", 0, 3),
    ("2025-07-30", "Pachuca", "San Diego FC", 3, 2),
    ("2025-07-31", "Necaxa", "Atlanta United FC", 3, 1),
    ("2025-07-31", "Inter Miami CF", "Atlas", 2, 1),
    ("2025-07-31", "Minnesota United FC", "Querétaro", 4, 1),
    ("2025-07-31", "Pumas UNAM", "Orlando City SC", 1, 1),
    ("2025-07-31", "Portland Timbers", "Atlético San Luis", 4, 0),
    ("2025-07-31", "América", "Real Salt Lake", 2, 2),
    ("2025-07-31", "Monterrey", "FC Cincinnati", 2, 3),
    ("2025-07-31", "Charlotte FC", "Juárez", 1, 4),
    ("2025-07-31", "Colorado Rapids", "Santos Laguna", 2, 1),
    ("2025-07-31", "LA Galaxy", "Tijuana", 5, 2),
    ("2025-07-31", "Guadalajara", "New York Red Bulls", 0, 1),
    ("2025-07-31", "Cruz Azul", "Seattle Sounders FC", 0, 7),
    # Matchday 2 (2025-08-05 .. 2025-08-09)
    ("2025-08-06", "Columbus Crew", "Puebla", 3, 1),
    ("2025-08-06", "Houston Dynamo FC", "Mazatlán", 0, 2),
    ("2025-08-06", "Los Angeles FC", "Pachuca", 1, 1),
    ("2025-08-06", "Toluca", "CF Montréal", 2, 1),
    ("2025-08-06", "New York City FC", "León", 2, 0),
    ("2025-08-06", "Tigres UANL", "San Diego FC", 2, 1),
    ("2025-08-07", "Pumas UNAM", "Atlanta United FC", 3, 2),
    ("2025-08-07", "Inter Miami CF", "Necaxa", 2, 2),
    ("2025-08-07", "América", "Minnesota United FC", 3, 3),
    ("2025-08-07", "Orlando City SC", "Atlas", 3, 1),
    ("2025-08-07", "Portland Timbers", "Querétaro", 1, 0),
    ("2025-08-07", "Real Salt Lake", "Atlético San Luis", 2, 2),
    ("2025-08-07", "FC Cincinnati", "Juárez", 2, 2),
    ("2025-08-07", "Guadalajara", "Charlotte FC", 2, 2),
    ("2025-08-07", "Colorado Rapids", "Tijuana", 1, 2),
    ("2025-08-07", "LA Galaxy", "Cruz Azul", 1, 1),
    ("2025-08-07", "Monterrey", "New York Red Bulls", 1, 1),
    ("2025-08-07", "Seattle Sounders FC", "Santos Laguna", 2, 1),
    # Matchday 3 (2025-08-12 .. 2025-08-14)
    ("2025-08-13", "Columbus Crew", "León", 1, 0),
    ("2025-08-13", "Houston Dynamo FC", "Pachuca", 1, 2),
    ("2025-08-13", "Tigres UANL", "Los Angeles FC", 1, 2),
    ("2025-08-13", "CF Montréal", "Puebla", 1, 2),
    ("2025-08-13", "Toluca", "New York City FC", 2, 1),
    ("2025-08-13", "Mazatlán", "San Diego FC", 0, 2),
    ("2025-08-13", "Atlanta United FC", "Atlas", 4, 1),
    ("2025-08-13", "Inter Miami CF", "Pumas UNAM", 3, 1),
    ("2025-08-13", "Minnesota United FC", "Atlético San Luis", 0, 2),
    ("2025-08-13", "Orlando City SC", "Necaxa", 5, 1),
    ("2025-08-13", "América", "Portland Timbers", 1, 1),
    ("2025-08-13", "Real Salt Lake", "Querétaro", 1, 0),
    ("2025-08-13", "Seattle Sounders FC", "Tijuana", 2, 1),
    ("2025-08-13", "FC Cincinnati", "Guadalajara", 1, 2),
    ("2025-08-13", "Monterrey", "Charlotte FC", 0, 2),
    ("2025-08-13", "Cruz Azul", "Colorado Rapids", 2, 2),
    ("2025-08-13", "LA Galaxy", "Santos Laguna", 4, 0),
    ("2025-08-13", "New York Red Bulls", "Juárez", 1, 1),
    # Knockout
    ("2025-08-20", "Inter Miami CF", "Tigres UANL", 2, 1),
    ("2025-08-20", "Toluca", "Orlando City SC", 0, 0),
    ("2025-08-20", "Seattle Sounders FC", "Puebla", 0, 0),
    ("2025-08-20", "LA Galaxy", "Pachuca", 2, 1),
    ("2025-08-27", "Inter Miami CF", "Orlando City SC", 3, 1),
    ("2025-08-27", "LA Galaxy", "Seattle Sounders FC", 0, 2),
    ("2025-08-31", "LA Galaxy", "Orlando City SC", 2, 1),
    ("2025-08-31", "Seattle Sounders FC", "Inter Miami CF", 0, 3),
]

# 2026 Leagues Cup — the first current-season cross-league evidence. The 2025
# bridge is a full transfer cycle old, and most of the model's disagreement with
# the market on 2026 fixtures traces to that staleness.
#
# HOME IS BY VENUE, NOT BY LISTING ORDER. Wikipedia lists "Monterrey 1-2 Orlando
# City", but the match was played at Inter&Co Stadium in Orlando, so Orlando
# were the home side and it is recorded here as Orlando 2-1 Monterrey. Getting
# this backwards moves a prediction by roughly 20 points.
RAW_2026 = [
    ("2026-08-05", "Inter Miami CF", "Atlético San Luis", 4, 2),
    ("2026-08-05", "Orlando City SC", "Monterrey", 2, 1),      # at Orlando
    ("2026-08-05", "Nashville SC", "León", 0, 1),
    ("2026-08-05", "FC Dallas", "Querétaro", 2, 0),
    ("2026-08-05", "Toluca", "Seattle Sounders FC", 3, 0),
    # 1-1 after 90 minutes; LAFC won 5-4 on penalties. The goal model uses the
    # regulation score, which is also how these markets settle.
    ("2026-08-05", "Los Angeles FC", "Guadalajara", 1, 1),
]

NAME_MAP["Nashville SC"] = "Nashville SC"
NAME_MAP["FC Dallas"] = "FC Dallas"

DIV = "LC:Leagues Cup"


def load(include_2026: bool = True) -> pd.DataFrame:
    rows = []
    source = RAW_2025 + (RAW_2026 if include_2026 else [])
    for date, h, a, hg, ag in source:
        hm, am = NAME_MAP.get(h), NAME_MAP.get(a)
        if hm is None or am is None:
            raise KeyError(f"unmapped Leagues Cup team: {h!r} / {a!r}")
        rows.append({
            "Div": DIV, "Date": pd.Timestamp(date),
            "HomeTeam": hm, "AwayTeam": am,
            "FTHG": float(hg), "FTAG": float(ag),
            "FTR": "H" if hg > ag else ("A" if ag > hg else "D"),
            "HS": pd.NA, "AS": pd.NA, "HST": pd.NA, "AST": pd.NA,
            "PSH": pd.NA, "PSD": pd.NA, "PSA": pd.NA,
            "AvgH": pd.NA, "AvgD": pd.NA, "AvgA": pd.NA,
            "MaxH": pd.NA, "MaxD": pd.NA, "MaxA": pd.NA,
            "B365H": pd.NA, "B365D": pd.NA, "B365A": pd.NA,
            "season": str(pd.Timestamp(date).year), "league": DIV, "Time": pd.NA,
        })
    df = pd.DataFrame(rows)
    for c in ("Div", "HomeTeam", "AwayTeam", "FTR", "season", "league"):
        df[c] = df[c].astype("string")
    return df


if __name__ == "__main__":
    d = load()
    mls = {"Columbus Crew", "Houston Dynamo", "Los Angeles FC", "CF Montreal",
           "New York City", "San Diego FC", "Atlanta Utd", "Inter Miami",
           "Minnesota United", "Orlando City", "Portland Timbers",
           "Real Salt Lake", "FC Cincinnati", "Charlotte", "Colorado Rapids",
           "Los Angeles Galaxy", "New York Red Bulls", "Seattle Sounders"}
    cross = sum(1 for _, r in d.iterrows()
                if (r["HomeTeam"] in mls) != (r["AwayTeam"] in mls))
    print(f"{len(d)} Leagues Cup matches, {cross} genuinely cross-league")
    print(f"  date range {d['Date'].min().date()} .. {d['Date'].max().date()}")
    hg, ag = d["FTHG"].mean(), d["FTAG"].mean()
    print(f"  mean goals: home {hg:.2f}, away {ag:.2f}")
    mls_home = d[d["HomeTeam"].isin(mls)]
    mls_away = d[d["AwayTeam"].isin(mls)]
    print(f"  MLS at home: {len(mls_home)} games, "
          f"scored {mls_home['FTHG'].mean():.2f} conceded {mls_home['FTAG'].mean():.2f}")
    print(f"  MLS away:    {len(mls_away)} games, "
          f"scored {mls_away['FTAG'].mean():.2f} conceded {mls_away['FTHG'].mean():.2f}")
    w = sum(1 for _, r in d.iterrows()
            if (r["HomeTeam"] in mls and r["FTHG"] > r["FTAG"])
            or (r["AwayTeam"] in mls and r["FTAG"] > r["FTHG"]))
    dr = int((d["FTHG"] == d["FTAG"]).sum())
    print(f"  MLS record vs Liga MX: {w}W {dr}D {len(d) - w - dr}L")
