"""Download NFL and WNBA game history.

NFL comes from nflverse (public CSV, no key). It is by far the richest source in
this project: scores back to 1999 plus **closing spread, total and moneyline**,
so the market benchmark arrives with the data rather than needing a separate
scrape. It also carries starting QBs — the NFL analogue of MLB's starting
pitcher — along with rest days, roof, surface, temperature and wind.

WNBA comes from Basketball Reference's schedule tables (one page per season).
ESPN's API returns 403 and stats.wnba.com times out, so this is the route that
works.

Both sports are modelled on MARGIN rather than counts: NFL and basketball scores
are far from Poisson. A margin model prices moneyline, spread and total from one
fit, which is three markets per game instead of one.
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
HDRS = {"User-Agent": "Mozilla/5.0 (sports model research)"}

NFL_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
BREF = "https://www.basketball-reference.com/wnba/years/{year}_games.html"


def fetch_nfl() -> pd.DataFrame:
    r = requests.get(NFL_URL, headers=HDRS, timeout=90)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df["date"] = pd.to_datetime(df["gameday"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    df["margin"] = df["home_score"] - df["away_score"]
    df["total_points"] = df["home_score"] + df["away_score"]
    out = RAW / "nfl_games.parquet"
    df.to_parquet(out, index=False)

    played = df[df["played"]]
    print(f"NFL: {len(df):,} games -> {out}")
    print(f"  played {len(played):,}   "
          f"{df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"  with closing spread {df['spread_line'].notna().sum():,}, "
          f"total {df['total_line'].notna().sum():,}, "
          f"moneyline {df['home_moneyline'].notna().sum():,}")
    print(f"  home margin mean {played['margin'].mean():+.2f}, "
          f"sd {played['margin'].std():.2f}")
    print(f"  home win rate {(played['margin'] > 0).mean():.3%}")
    print(f"  mean total {played['total_points'].mean():.1f}")

    # Integrity: every season since 2000 should hold a full slate.
    per = played.groupby("season").size()
    bad = [int(s) for s in per.index if 2000 <= s <= 2025 and per[s] < 200]
    if bad:
        print(f"  ! thin seasons: {bad}")
    else:
        print("  integrity checks passed")
    return df


def fetch_wnba(start: int = 2015, end: int = 2026) -> pd.DataFrame:
    frames = []
    for yr in range(start, end + 1):
        try:
            r = requests.get(BREF.format(year=yr), headers=HDRS, timeout=60)
            if r.status_code != 200:
                print(f"  {yr}: HTTP {r.status_code}")
                continue
            tables = pd.read_html(io.StringIO(r.text))
        except Exception as e:
            print(f"  {yr}: {type(e).__name__}")
            continue
        if not tables:
            continue
        t = tables[0]
        cols = {c: str(c) for c in t.columns}
        t = t.rename(columns=cols)
        need = ["Date", "Visitor/Neutral", "PTS", "Home/Neutral", "PTS.1"]
        if not all(c in t.columns for c in need):
            print(f"  {yr}: unexpected columns {list(t.columns)[:6]}")
            continue
        t = t[need].copy()
        t.columns = ["date", "away_team", "away_score", "home_team", "home_score"]
        t["date"] = pd.to_datetime(t["date"], errors="coerce", format="mixed")
        for c in ("away_score", "home_score"):
            t[c] = pd.to_numeric(t[c], errors="coerce")
        t = t.dropna(subset=["date", "home_team", "away_team"])
        t["season"] = yr
        frames.append(t)
        print(f"  {yr}: {len(t)} games ({t['home_score'].notna().sum()} played)")
        time.sleep(3.0)   # Basketball Reference rate-limits aggressively

    if not frames:
        print("WNBA: nothing downloaded", file=sys.stderr)
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).sort_values("date")
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    df["margin"] = df["home_score"] - df["away_score"]
    df["total_points"] = df["home_score"] + df["away_score"]
    out = RAW / "wnba_games.parquet"
    df.to_parquet(out, index=False)

    played = df[df["played"]]
    print(f"\nWNBA: {len(df):,} games -> {out}")
    print(f"  played {len(played):,}   "
          f"{df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"  teams {df['home_team'].nunique()}")
    print(f"  home margin mean {played['margin'].mean():+.2f}, "
          f"sd {played['margin'].std():.2f}")
    print(f"  home win rate {(played['margin'] > 0).mean():.3%}")
    print(f"  mean total {played['total_points'].mean():.1f}")
    return df


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "nfl"):
        fetch_nfl()
    if which in ("both", "wnba"):
        print()
        fetch_wnba()
