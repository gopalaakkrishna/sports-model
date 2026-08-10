"""Understat xG extraction — requires a real browser, not requests.

ACCESS NOTE, learned the hard way. Understat returns HTTP 200 with the correct
page title to plain `requests`, but the response is an 18KB shell: the match
data is absent and the only JSON.parse on the page is an AdSense payload. A
keyword check for "xg" passes and means nothing. Cloudflare is in the response
headers. `fetch()` from inside the page is served the same stripped HTML.

Under a real browser navigation the globals ARE populated:
    datesData  -> 380 fixtures/season, each with match-level xG for both sides
    teamsData  -> per-team history: xG, xGA, npxG, npxGA, ppda, deep, xpts

So extraction goes through the browser tool, one navigation per league-season:

    1. navigate to https://understat.com/league/{LEAGUE}/{SEASON}
    2. evaluate the extractor below
    3. append the returned rows to data/raw/understat_xg.csv

LEAGUES: EPL, La_liga, Bundesliga, Serie_A, Ligue_1, RFPL
SEASONS: 2014 onward.

The extractor, to run in the page:

    (() => {
      const rows = [];
      for (const g of datesData) {
        if (!g.isResult) continue;
        rows.push([g.datetime.slice(0,10), g.h.title, g.a.title,
                   g.goals.h, g.goals.a,
                   (+g.xG.h).toFixed(3), (+g.xG.a).toFixed(3)].join('|'));
      }
      return rows.join('\\n');
    })()

Why this is worth the awkwardness: the shots-on-target hybrid passed its ablation
(-0.00086 log loss) precisely because shot QUALITY carries signal that goals
alone do not. Raw SOT is a crude proxy for xG; this is the real quantity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "understat_xg.csv"

COLUMNS = ["date", "home", "away", "goals_h", "goals_a", "xg_h", "xg_a",
           "league", "season"]


def ingest(raw: str, league: str, season: str) -> int:
    """Append pipe-delimited rows produced by the in-page extractor."""
    rows = []
    for line in raw.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) != 7:
            continue
        rows.append(parts + [league, season])
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in ("goals_h", "goals_a", "xg_h", "xg_a"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home", "away"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        prior = pd.read_csv(OUT, parse_dates=["date"])
        df = pd.concat([prior, df], ignore_index=True)
        df = df.drop_duplicates(subset=["date", "home", "away"], keep="last")
    df.sort_values("date").to_csv(OUT, index=False)
    return len(rows)


def status() -> None:
    if not OUT.exists():
        print(f"no data yet at {OUT}")
        print("Extract via the browser tool — see the module docstring.")
        return
    df = pd.read_csv(OUT, parse_dates=["date"])
    print(f"{len(df):,} matches with xG   {OUT}")
    print(f"  range {df['date'].min().date()} .. {df['date'].max().date()}")
    print(df.groupby(["league", "season"]).size().to_string())
    print(f"\n  mean xG: home {df['xg_h'].mean():.3f}, away {df['xg_a'].mean():.3f}")
    print(f"  mean goals: home {df['goals_h'].mean():.3f}, "
          f"away {df['goals_a'].mean():.3f}")


if __name__ == "__main__":
    status()
