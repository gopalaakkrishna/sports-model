"""Download The Hundred (and T20) match data from Cricsheet.

Cricsheet publishes free ball-by-ball data as zipped CSVs. Each match has an
_info file (teams, venue, toss, outcome) and a deliveries file.

Scope warning worth stating up front: The Hundred began in 2021 and plays ~34
matches a season, so the whole competition is a couple of hundred matches. That
is one or two seasons' worth of signal by the standards of the other sports
here — MLB alone has 28,000 games. Franchise squads also turn over heavily
between seasons.

There is also no market to check against: Kalshi lists 62 cricket series and
quotes exactly none of them. So a Hundred model can be built and calibrated
against itself, but cannot be measured against a market the way soccer, MLB and
NFL were.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HDRS = {"User-Agent": "Mozilla/5.0 (sports model research)"}

SOURCES = {
    "hundred_male": "https://cricsheet.org/downloads/hnd_male_csv2.zip",
    "hundred_female": "https://cricsheet.org/downloads/hnd_female_csv2.zip",
}


def parse_info(text: str) -> dict:
    """Cricsheet _info files are 'info,key,value...' rows."""
    out: dict = {"teams": []}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or parts[0] != "info":
            continue
        key, val = parts[1], parts[2]
        if key == "team":
            out["teams"].append(val)
        elif key in ("date", "venue", "city", "gender", "season", "event",
                     "toss_winner", "toss_decision", "winner", "player_of_match",
                     "match_number"):
            out[key] = val
        elif key == "winner_runs":
            out["winner_runs"] = val
        elif key == "winner_wickets":
            out["winner_wickets"] = val
        elif key == "outcome":
            out["outcome"] = val
    return out


def fetch(name: str, url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HDRS, timeout=180)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    infos, balls = [], []
    for fn in z.namelist():
        if not fn.endswith(".csv"):
            continue
        # Cricsheet ships an all_matches.csv aggregate alongside the per-match
        # files. Treating it as a match produced a single "innings" of 27,244
        # runs and dragged the mean innings total from 139 to 287.
        if Path(fn).stem in ("all_matches", "README"):
            continue
        if fn.endswith("_info.csv"):
            d = parse_info(z.read(fn).decode("utf-8", "ignore"))
            d["match_id"] = fn.replace("_info.csv", "")
            infos.append(d)
        else:
            try:
                b = pd.read_csv(io.BytesIO(z.read(fn)))
                b["match_id"] = fn.replace(".csv", "")
                balls.append(b)
            except Exception:
                continue

    if not infos:
        print(f"  {name}: no match info found", file=sys.stderr)
        return pd.DataFrame()

    mi = pd.DataFrame(infos)
    mi["date"] = pd.to_datetime(mi.get("date"), errors="coerce")
    mi["home_team"] = mi["teams"].str[0]
    mi["away_team"] = mi["teams"].str[1]
    mi = mi.drop(columns=["teams"])

    if balls:
        bb = pd.concat(balls, ignore_index=True)
        # Innings totals per match, for a scoring model.
        tot = (bb.groupby(["match_id", "innings"])
                 .agg(runs=("runs_off_bat", "sum"),
                      extras=("extras", "sum"),
                      balls=("ball", "size"))
                 .reset_index())
        tot["total"] = tot["runs"] + tot["extras"]
        tot.to_parquet(RAW / f"cricket_{name}_innings.parquet", index=False)
        print(f"  {name}: {len(bb):,} deliveries, "
              f"{tot['match_id'].nunique()} matches with innings totals")

    mi.to_parquet(RAW / f"cricket_{name}_matches.parquet", index=False)
    return mi


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        try:
            mi = fetch(name, url)
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:90]}")
            continue
        if mi.empty:
            continue
        print(f"{name}: {len(mi)} matches, "
              f"{mi['date'].min().date()} .. {mi['date'].max().date()}")
        print(f"  teams: {sorted(set(mi['home_team']) | set(mi['away_team']))}")
        if "winner" in mi:
            dec = mi["winner"].notna().sum()
            print(f"  with a winner: {dec} / {len(mi)} "
                  f"({len(mi) - dec} no-result / tie)")
        print(f"  seasons: {sorted(mi['season'].dropna().unique())}")
        print()


if __name__ == "__main__":
    main()
