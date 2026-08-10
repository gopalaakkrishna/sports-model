"""Download per-inning MLB line scores, so in-play states can be reconstructed.

Full play-by-play is ~590KB per game — far more than an in-play win-probability
model needs. The per-inning line score gives the score after every half-inning,
which is 18 states per game and enough to both fit and validate.

The schedule endpoint hydrates line scores for a whole date range at once, so a
season costs about a dozen requests rather than thousands.
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


def fetch_month(year: int, month: int) -> list[dict]:
    last = pd.Period(f"{year}-{month:02d}").days_in_month
    params = {
        "sportId": 1,
        "startDate": f"{year}-{month:02d}-01",
        "endDate": f"{year}-{month:02d}-{last}",
        "hydrate": "linescore,team",
    }
    for attempt in range(4):
        try:
            r = requests.get(f"{STATS}/schedule", params=params, timeout=120)
            if r.status_code == 200:
                return r.json().get("dates", [])
            time.sleep(2 * (attempt + 1))
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    print(f"  {year}-{month:02d}: failed", file=sys.stderr)
    return []


def parse(dates: list[dict]) -> list[dict]:
    rows = []
    for d in dates:
        for g in d.get("games", []):
            if g.get("gameType") != "R":
                continue
            if (g.get("status") or {}).get("detailedState") != "Final":
                continue
            ls = g.get("linescore") or {}
            innings = ls.get("innings") or []
            if len(innings) < 8:
                continue
            t = g.get("teams") or {}
            home = (t.get("home") or {}).get("team", {}).get("name")
            away = (t.get("away") or {}).get("team", {}).get("name")
            hs = (t.get("home") or {}).get("score")
            as_ = (t.get("away") or {}).get("score")
            if not home or not away or hs is None or as_ is None:
                continue
            seq = []
            for i in innings:
                seq.append((
                    i.get("num"),
                    (i.get("away") or {}).get("runs"),
                    (i.get("home") or {}).get("runs"),
                ))
            rows.append({
                "gamePk": g.get("gamePk"), "date": d.get("date"),
                "home_team": home, "away_team": away,
                "home_score": hs, "away_score": as_,
                "n_innings": len(innings),
                "innings": seq,
            })
    return rows


def main(start_year: int = 2021, end_year: int = 2026) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    allrows = []
    for yr in range(start_year, end_year + 1):
        got = 0
        for mo in range(3, 11):
            rows = parse(fetch_month(yr, mo))
            allrows.extend(rows)
            got += len(rows)
            time.sleep(0.2)
        print(f"{yr}: {got:,} finals with inning detail")

    if not allrows:
        print("nothing downloaded", file=sys.stderr)
        sys.exit(1)

    # Explode to one row per half-inning STATE (the score going into it).
    states = []
    for g in allrows:
        ah = 0
        hh = 0
        for (num, ar, hr) in g["innings"]:
            ar = ar or 0
            hr = hr or 0
            # State at the start of the top of this inning.
            states.append({
                "gamePk": g["gamePk"], "date": g["date"],
                "home_team": g["home_team"], "away_team": g["away_team"],
                "inning": num, "half": "top",
                "home_runs_so_far": hh, "away_runs_so_far": ah,
                "diff": hh - ah,
                "home_won": int(g["home_score"] > g["away_score"]),
            })
            ah += ar
            # State at the start of the bottom.
            states.append({
                "gamePk": g["gamePk"], "date": g["date"],
                "home_team": g["home_team"], "away_team": g["away_team"],
                "inning": num, "half": "bottom",
                "home_runs_so_far": hh, "away_runs_so_far": ah,
                "diff": hh - ah,
                "home_won": int(g["home_score"] > g["away_score"]),
            })
            hh += hr

    games = pd.DataFrame([{k: v for k, v in g.items() if k != "innings"}
                          for g in allrows])
    st = pd.DataFrame(states)
    games.to_parquet(RAW / "mlb_linescores.parquet", index=False)
    st.to_parquet(RAW / "mlb_inplay_states.parquet", index=False)

    print(f"\nSaved {len(games):,} games -> mlb_linescores.parquet")
    print(f"Saved {len(st):,} half-inning states -> mlb_inplay_states.parquet")
    print(f"  {games['date'].min()} .. {games['date'].max()}")
    print(f"  home win rate {games.eval('home_score > away_score').mean():.3%}")
    print(f"  mean innings {games['n_innings'].mean():.2f}")


if __name__ == "__main__":
    main()
