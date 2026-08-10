"""Download the year-round / summer leagues from football-data.co.uk.

These live in a different file family from the European divisions and use a
different schema (Home/Away, HG/AG/Res, and the "C" closing-odds columns). They
are mapped here onto the same canonical column names the rest of the pipeline
uses, so the model and backtest need no special-casing.

They matter because they run through the European off-season: in August the
Argentine, Brazilian, Mexican, MLS, Scandinavian and other seasons are live
while the big European leagues have not restarted.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CSV_CACHE = RAW / "new_csv"
BASE = "https://www.football-data.co.uk/new"

COUNTRIES = {
    "ARG": "Argentina", "BRA": "Brazil", "MEX": "Mexico", "USA": "USA",
    "JPN": "Japan", "CHN": "China", "NOR": "Norway", "SWE": "Sweden",
    "FIN": "Finland", "IRL": "Ireland", "DNK": "Denmark", "POL": "Poland",
    "ROU": "Romania", "RUS": "Russia", "AUT": "Austria", "SWZ": "Switzerland",
}

# source column -> canonical column
RENAME = {
    "Home": "HomeTeam", "Away": "AwayTeam",
    "HG": "FTHG", "AG": "FTAG", "Res": "FTR",
    "PSCH": "PSH", "PSCD": "PSD", "PSCA": "PSA",
    "AvgCH": "AvgH", "AvgCD": "AvgD", "AvgCA": "AvgA",
    "MaxCH": "MaxH", "MaxCD": "MaxD", "MaxCA": "MaxA",
    "B365CH": "B365H", "B365CD": "B365D", "B365CA": "B365A",
}
NUMERIC = ["FTHG", "FTAG", "PSH", "PSD", "PSA", "AvgH", "AvgD", "AvgA",
           "MaxH", "MaxD", "MaxA", "B365H", "B365D", "B365A"]
CANONICAL = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
             "HS", "AS", "HST", "AST", "PSH", "PSD", "PSA", "AvgH", "AvgD", "AvgA",
             "MaxH", "MaxD", "MaxA", "B365H", "B365D", "B365A", "season", "league"]


def div_code(country_code: str, league: str) -> str:
    """Stable division id. League names carry stray whitespace in the source."""
    return f"{country_code}:{str(league).strip()}"


def load_country(code: str, refresh: bool) -> pd.DataFrame | None:
    path = CSV_CACHE / f"{code}.csv"
    if not path.exists() or refresh:
        url = f"{BASE}/{code}.csv"
        try:
            r = requests.get(url, timeout=60)
        except requests.RequestException as e:
            print(f"  {code}: network error {e}", file=sys.stderr)
            return None
        if r.status_code != 200 or not r.content.strip():
            print(f"  {code}: HTTP {r.status_code}", file=sys.stderr)
            return None
        path.write_bytes(r.content)
        time.sleep(0.15)

    df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip", low_memory=False)
    df.columns = [str(c).replace("﻿", "").replace("ï»¿", "").strip() for c in df.columns]
    if "Home" not in df.columns:
        print(f"  {code}: unexpected schema {list(df.columns)[:8]}", file=sys.stderr)
        return None

    df = df.rename(columns=RENAME)
    df["Div"] = [div_code(code, lg) for lg in df["League"]]
    df["season"] = df["Season"].astype("string")
    df["league"] = df["Div"]
    # These files carry no shot data.
    for c in ("HS", "AS", "HST", "AST"):
        df[c] = pd.NA
    for c in CANONICAL:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[CANONICAL].copy()

    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("Div", "Time", "HomeTeam", "AwayTeam", "FTR", "season", "league"):
        df[c] = df[c].astype("string")
    df["Date"] = pd.to_datetime(df["Date"].astype("string"), format="mixed",
                                dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    return df


def main(refresh: bool = True) -> None:
    CSV_CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for code in COUNTRIES:
        df = load_country(code, refresh)
        if df is None:
            continue
        frames.append(df)
        played = df["FTHG"].notna().sum()
        odds = df["AvgH"].notna().sum()
        print(f"  {code}: {len(df):>6} rows, played {played:>6}, odds {odds:>6}, "
              f"to {df['Date'].max().date()}")
    if not frames:
        print("nothing downloaded", file=sys.stderr)
        sys.exit(1)

    allc = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    out = RAW / "new_leagues_raw.parquet"
    allc.to_parquet(out, index=False)
    print(f"\nSaved {len(allc):,} rows -> {out}")
    print(f"  divisions: {allc['Div'].nunique()}")
    print(f"  with avg closing odds: {allc['AvgH'].notna().sum():,}")
    print(f"  date range: {allc['Date'].min().date()} .. {allc['Date'].max().date()}")


if __name__ == "__main__":
    main()
