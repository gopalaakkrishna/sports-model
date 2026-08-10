"""Download T20 international data from Cricsheet.

Why T20Is rather than The Hundred: the two things that killed the Hundred model
are both fixed here.

  sample     6,981 match files against 189
  stability  national squads persist; franchise squads are re-drafted yearly

There is also far more spread in team strength — full members against associate
nations — so there is genuinely more to predict than in an eight-team league
designed for parity.

Only matches from `since` onward are parsed. Cricket has changed enough since
2010 that older games mostly add noise, and parsing 7,000 files is slow.
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
URL = "https://cricsheet.org/downloads/t20s_{gender}_csv2.zip"


def parse_info(text: str) -> dict:
    out = {"teams": [], "players": []}
    for line in text.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 3 or p[0] != "info":
            continue
        k, v = p[1], p[2]
        if k == "team":
            out["teams"].append(v)
        elif k == "player" and len(p) >= 4:
            out["players"].append((v, p[3]))
        elif k in ("date", "venue", "city", "season", "winner", "toss_winner",
                   "toss_decision", "event", "match_type"):
            out.setdefault(k, v)
        elif k == "outcome":
            out["outcome"] = v
    return out


def main(gender: str = "male", since: str = "2015-01-01") -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"downloading T20I {gender} ...")
    r = requests.get(URL.format(gender=gender), headers=HDRS, timeout=600)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print(f"  {len(z.namelist())} files")

    cutoff = pd.Timestamp(since)
    matches, squads, balls = [], [], []
    kept = skipped = 0
    for fn in z.namelist():
        stem = Path(fn).stem
        if not fn.endswith("_info.csv") or stem in ("all_matches", "README"):
            continue
        info = parse_info(z.read(fn).decode("utf-8", "ignore"))
        d = pd.to_datetime(info.get("date"), errors="coerce")
        if pd.isna(d) or d < cutoff or len(info["teams"]) != 2:
            skipped += 1
            continue
        mid = stem.replace("_info", "")
        matches.append({
            "match_id": mid, "date": d,
            "team_a": info["teams"][0], "team_b": info["teams"][1],
            "winner": info.get("winner"), "venue": info.get("venue"),
            "season": info.get("season"), "toss_winner": info.get("toss_winner"),
            "toss_decision": info.get("toss_decision"),
            "outcome": info.get("outcome"),
        })
        for team, player in info["players"]:
            squads.append({"match_id": mid, "date": d, "team": team,
                           "player": player})
        try:
            b = pd.read_csv(io.BytesIO(z.read(f"{mid}.csv")), low_memory=False)
            b["match_id"] = mid
            b["date"] = d
            balls.append(b)
            kept += 1
        except Exception:
            continue

    if not matches:
        print("nothing parsed", file=sys.stderr)
        sys.exit(1)

    mt = pd.DataFrame(matches).sort_values("date")
    sq = pd.DataFrame(squads)
    bb = pd.concat(balls, ignore_index=True)

    # T20I seasons appear as both 2019 and "2019/20", so the column arrives with
    # mixed int/str types and parquet refuses it. Force every object column to
    # string rather than guessing which ones are affected.
    for df in (mt, sq, bb):
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype("string")

    mt.to_parquet(RAW / f"t20i_{gender}_matches.parquet", index=False)
    sq.to_parquet(RAW / f"t20i_{gender}_squads.parquet", index=False)
    bb.to_parquet(RAW / f"t20i_{gender}_balls.parquet", index=False)

    dec = mt["winner"].notna().sum()
    teams = pd.concat([mt["team_a"], mt["team_b"]]).value_counts()
    print(f"\nkept {kept:,} matches (skipped {skipped:,} pre-{since} or malformed)")
    print(f"  {mt['date'].min().date()} .. {mt['date'].max().date()}")
    print(f"  with a winner: {dec:,} / {len(mt):,}")
    print(f"  deliveries: {len(bb):,}")
    print(f"  players: {sq['player'].nunique():,}")
    print(f"  teams: {len(teams)}  (top: {list(teams.head(8).index)})")
    print(f"  teams with 20+ matches: {(teams >= 20).sum()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "male")
