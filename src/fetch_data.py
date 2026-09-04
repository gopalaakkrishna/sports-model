"""Download historical football results + closing odds from football-data.co.uk.

Each CSV is one league-season. Files are cached to data/raw/csv so re-runs only
fetch the current season. The combined output keeps results, shots and the
closing odds we later benchmark the model against.
"""

import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CSV_CACHE = RAW / "csv"
BASE = "https://www.football-data.co.uk/mmz4281"

LEAGUES = {
    "E0": "England Premier League",
    "E1": "England Championship",
    "E2": "England League One",
    "E3": "England League Two",
    "SC0": "Scotland Premiership",
    "SC1": "Scotland Championship",
    "SC2": "Scotland League One",
    "SC3": "Scotland League Two",
    "D1": "Germany Bundesliga",
    "D2": "Germany 2. Bundesliga",
    "I1": "Italy Serie A",
    "I2": "Italy Serie B",
    "SP1": "Spain La Liga",
    "SP2": "Spain Segunda",
    "F1": "France Ligue 1",
    "F2": "France Ligue 2",
    "N1": "Netherlands Eredivisie",
    "B1": "Belgium Pro League",
    "P1": "Portugal Primeira Liga",
    "T1": "Turkey Super Lig",
    "G1": "Greece Super League",
}

# Columns we care about. Everything else in the source files is dropped.
KEEP = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",          # full-time goals and result
    "HS", "AS", "HST", "AST",       # shots / shots on target
    # PRE-CLOSING (opening) odds. football-data's notes.txt is explicit:
    # "These are for pre-closing odds. For the closing odds, as below but with
    # an additional C character". These columns were labelled "closing" here
    # for months and they are not — every market benchmark computed from them
    # was measured against the OPENING line, which is the weaker one.
    "PSH", "PSD", "PSA",            # Pinnacle opening
    "AvgH", "AvgD", "AvgA",         # market average opening
    "MaxH", "MaxD", "MaxA",         # best available opening
    "B365H", "B365D", "B365A",      # Bet365 opening, deepest history
    # CLOSING odds — the actual sharp benchmark, and the thing our model has
    # to be measured against. The move from opening to closing is the market
    # pricing information that arrives late (team news, lineups, money), so
    # the gap between these two columns is a direct measure of how much that
    # late information is worth.
    "PSCH", "PSCD", "PSCA",         # Pinnacle closing (sharpest)
    "AvgCH", "AvgCD", "AvgCA",      # market average closing
    "MaxCH", "MaxCD", "MaxCA",      # best available closing
    # Over/under 2.5 CLOSING odds. Absent until 2026-08-19, which is why the
    # totals market could never be compared against the line retrospectively —
    # we had the goals but not the price. "C" is the closing quote; the
    # opening one is the wrong benchmark because it has not absorbed team news.
    # Pinnacle first (sharpest), market average as the fallback that survives
    # Pinnacle's withdrawal from the feed in 2025/26.
    "PC>2.5", "PC<2.5",             # Pinnacle closing over/under
    "AvgC>2.5", "AvgC<2.5",         # market average closing over/under
    "MaxC>2.5", "MaxC<2.5",         # best available closing over/under
]
NUMERIC = [c for c in KEEP if c not in ("Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTR")]


def season_codes(start_year: int, end_year: int) -> list[str]:
    """2005 -> '0506'. end_year is the last season START year, inclusive."""
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(start_year, end_year + 1)]


def load_csv(season: str, league: str, refresh: bool) -> pd.DataFrame | None:
    """Fetch one league-season, using the on-disk cache unless refresh is set."""
    path = CSV_CACHE / f"{season}_{league}.csv"
    if path.exists() and not refresh:
        content = path.read_bytes()
    else:
        url = f"{BASE}/{season}/{league}.csv"
        try:
            r = requests.get(url, timeout=30)
        except requests.RequestException as e:
            print(f"  {season}/{league}: network error {e}", file=sys.stderr)
            return None
        if r.status_code != 200 or not r.content.strip():
            return None
        content = r.content
        path.write_bytes(content)
        time.sleep(0.15)  # be polite to a free data host

    try:
        df = pd.read_csv(
            path, encoding="latin-1", on_bad_lines="skip", low_memory=False
        )
    except Exception as e:
        print(f"  {season}/{league}: parse error {e}", file=sys.stderr)
        return None

    # Recent files carry a UTF-8 BOM, which latin-1 turns into a "ï»¿" prefix on
    # the first column name. Left alone this silently nulls out the Div column.
    df.columns = [
        str(c).replace("﻿", "").replace("ï»¿", "").strip() for c in df.columns
    ]

    if df.empty or "HomeTeam" not in df.columns:
        return None

    # Keep only known columns; add any that this season's file lacks.
    present = [c for c in KEEP if c in df.columns]
    df = df[present].copy()
    for col in KEEP:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[KEEP]

    # Force consistent dtypes so the seasons concat cleanly.
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("Div", "Time", "HomeTeam", "AwayTeam", "FTR"):
        df[col] = df[col].astype("string")

    # The league code from the URL is authoritative — never trust the file's own
    # Div column, which is occasionally blank or malformed.
    df["Div"] = league
    df["season"] = season
    df["league"] = league
    # Drop rows with no teams (trailing blank lines are common in these files).
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    return df


def main(start_year: int = 2005, end_year: int = 2026, refresh_last: int = 1) -> None:
    CSV_CACHE.mkdir(parents=True, exist_ok=True)
    seasons = season_codes(start_year, end_year)
    frames = []
    for i, season in enumerate(seasons):
        # Always re-fetch the most recent season(s) — they're still being played.
        refresh = i >= len(seasons) - refresh_last
        got = 0
        for league in LEAGUES:
            df = load_csv(season, league, refresh)
            if df is not None:
                frames.append(df)
                got += len(df)
        print(f"{season}: {got:>5} matches{'  (refreshed)' if refresh else ''}")

    if not frames:
        print("No data downloaded.", file=sys.stderr)
        sys.exit(1)

    all_df = pd.concat(frames, ignore_index=True)

    # Dates appear as both dd/mm/yy and dd/mm/yyyy across seasons.
    all_df["Date"] = pd.to_datetime(
        all_df["Date"].astype("string"), format="mixed", dayfirst=True, errors="coerce"
    )
    all_df = all_df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    out = RAW / "football_data_raw.parquet"
    all_df.to_parquet(out, index=False)

    played = all_df["FTHG"].notna().sum()
    with_odds = all_df["PSH"].notna().sum()
    print(f"\nSaved {len(all_df):,} rows -> {out}")
    print(f"  played (have score): {played:,}")
    print(f"  with Pinnacle closing odds: {with_odds:,}")
    print(f"  date range: {all_df['Date'].min().date()} .. {all_df['Date'].max().date()}")

    # Integrity checks. A BOM in the source files once silently nulled the Div
    # column and dropped two whole seasons, so these are worth keeping.
    problems = []
    if all_df["Div"].isna().any():
        problems.append(f"{all_df['Div'].isna().sum()} rows with null Div")
    per_year = all_df.groupby(all_df["Date"].dt.year).size()
    for yr in range(start_year + 1, end_year + 1):
        if per_year.get(yr, 0) < 2000:
            problems.append(f"only {per_year.get(yr, 0)} matches in {yr}")
    for lg in LEAGUES:
        yrs = all_df.loc[all_df["Div"] == lg, "Date"].dt.year
        if len(yrs) == 0:
            problems.append(f"no rows at all for {lg}")
        elif yrs.max() < end_year:
            problems.append(f"{lg} ends at {yrs.max()}, expected {end_year}")
    if problems:
        print("\n  DATA INTEGRITY WARNINGS:")
        for p in problems:
            print(f"    ! {p}")
    else:
        print("  integrity checks passed")


if __name__ == "__main__":
    main()
