"""Download MLB game results and probable starting pitchers from MLB StatsAPI.

Free, public, no API key. Date-range queries let a whole month come back in one
call, so a dozen seasons is a few hundred requests.

What is kept, and why:

* runs scored / allowed per team per game — the quantity the model predicts
* starting pitcher for each side — in baseball this is the single largest
  game-level factor, far more than any lineup effect in soccer
* venue and date, for home advantage and time decay

Regular season only. Spring training and exhibition games (gameType != 'R')
would pollute the ratings badly, since teams field non-competitive lineups.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
STATS = "https://statsapi.mlb.com/api/v1"

# 'R' regular season. 'P'/'D'/'L'/'W' are the various playoff rounds — kept
# separate so they can be included or excluded deliberately.
KEEP_TYPES = {"R", "F", "D", "L", "W"}


def fetch_range(start: str, end: str) -> list[dict]:
    params = {
        "sportId": 1,
        "startDate": start,
        "endDate": end,
        "hydrate": "team,probablePitcher,linescore",
    }
    for attempt in range(4):
        try:
            r = requests.get(f"{STATS}/schedule", params=params, timeout=90)
            if r.status_code == 200:
                return r.json().get("dates", [])
            print(f"  {start}..{end}: HTTP {r.status_code}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"  {start}..{end}: {e} (attempt {attempt + 1})", file=sys.stderr)
        time.sleep(2 * (attempt + 1))
    return []


def parse(dates: list[dict]) -> list[dict]:
    rows = []
    for d in dates:
        for g in d.get("games", []):
            if g.get("gameType") not in KEEP_TYPES:
                continue
            status = (g.get("status") or {}).get("detailedState", "")
            t = g.get("teams") or {}
            home, away = t.get("home") or {}, t.get("away") or {}
            ht, at = home.get("team") or {}, away.get("team") or {}
            if not ht.get("name") or not at.get("name"):
                continue
            ls = g.get("linescore") or {}
            rows.append({
                "gamePk": g.get("gamePk"),
                "date": d.get("date"),
                "gameType": g.get("gameType"),
                "status": status,
                "final": status == "Final",
                "doubleHeader": (g.get("doubleHeader") or "N"),
                "home_team": ht.get("name"),
                "away_team": at.get("name"),
                "home_runs": home.get("score"),
                "away_runs": away.get("score"),
                "innings": ls.get("currentInning"),
                "home_sp": ((home.get("probablePitcher") or {}).get("fullName")),
                "away_sp": ((away.get("probablePitcher") or {}).get("fullName")),
                "home_sp_id": ((home.get("probablePitcher") or {}).get("id")),
                "away_sp_id": ((away.get("probablePitcher") or {}).get("id")),
                "venue": (g.get("venue") or {}).get("name"),
            })
    return rows


def main(start_year: int = 2015, end_year: int = 2026) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for year in range(start_year, end_year + 1):
        yr_rows = []
        # Month at a time keeps each response a sane size.
        for m in range(3, 12):
            last = pd.Period(f"{year}-{m:02d}").days_in_month
            dates = fetch_range(f"{year}-{m:02d}-01", f"{year}-{m:02d}-{last}")
            yr_rows.extend(parse(dates))
            time.sleep(0.1)
        all_rows.extend(yr_rows)
        played = sum(1 for r in yr_rows if r["final"])
        print(f"{year}: {len(yr_rows):>5} games ({played} final)")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("nothing downloaded", file=sys.stderr)
        sys.exit(1)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates("gamePk")
    df = df.sort_values("date").reset_index(drop=True)

    out = RAW / "mlb_games.parquet"
    df.to_parquet(out, index=False)

    final = df[df["final"]]
    print(f"\nSaved {len(df):,} games -> {out}")
    print(f"  final (have score): {len(final):,}")
    print(f"  with both probable pitchers: {df[['home_sp','away_sp']].notna().all(axis=1).sum():,}")
    print(f"  date range: {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"  teams: {df['home_team'].nunique()}")

    # Integrity checks — the soccer build was nearly wrecked by silent truncation.
    problems = []
    per_year = final.groupby(final["date"].dt.year).size()
    for y in range(start_year, end_year):
        n = per_year.get(y, 0)
        # A full season is ~2,430 games; 2020 was shortened to ~900.
        floor = 700 if y == 2020 else 2000
        if n < floor:
            problems.append(f"only {n} final games in {y} (expected >= {floor})")
    if final["home_runs"].isna().any():
        problems.append(f"{final['home_runs'].isna().sum()} final games missing runs")
    if problems:
        print("\n  DATA INTEGRITY WARNINGS:")
        for p in problems:
            print(f"    ! {p}")
    else:
        print("  integrity checks passed")


if __name__ == "__main__":
    main()
