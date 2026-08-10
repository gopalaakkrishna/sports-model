"""Player-level ratings for The Hundred — the upgrade the format actually needs.

Why player-level rather than team-level:

Franchise identity barely persists. Year-over-year correlation of team win rate
is +0.142, against 0.5-0.7 in football and baseball. Squads are re-drafted
annually, overseas players rotate, and three of eight franchises were renamed
for 2026. A team-strength model built on that scored 0.6932 against a 0.6924
base rate — worse than guessing.

Players do persist. 270 of them across the competition, median 11 matches, 144
with 10 or more, 128 appearing in three seasons or more. "Trent Rockets" is not
a stable entity; the batter who scores at 1.5 runs a ball is.

Ratings, from ball-by-ball data:

    batting  runs off the bat per ball faced, versus league average
    bowling  runs conceded per ball bowled, versus league average

Both are shrunk toward the league mean in proportion to balls involved, so a
player with 20 balls sits near average and only a substantial workload moves
him. Team strength is then the aggregate over the eleven who actually played.

Honest limitation: predicting a FUTURE match needs the XI, and Cricsheet only
records it after the fact. The last-known XI is used as a proxy, which is
reasonable across a three-week tournament but wrong whenever a side rotates.
"""

from __future__ import annotations

import io
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HDRS = {"User-Agent": "Mozilla/5.0 (sports model research)"}
URL = "https://cricsheet.org/downloads/hnd_{gender}_csv2.zip"


def download(gender: str = "male") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ball-by-ball deliveries plus per-match squads."""
    r = requests.get(URL.format(gender=gender), headers=HDRS, timeout=240)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))

    balls, squads = [], []
    for fn in z.namelist():
        stem = Path(fn).stem
        if not fn.endswith(".csv") or stem in ("all_matches", "README"):
            continue
        if fn.endswith("_info.csv"):
            mid = stem.replace("_info", "")
            season = date = None
            for line in z.read(fn).decode("utf-8", "ignore").splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) >= 3 and p[1] == "season":
                    season = p[2]
                elif len(p) >= 3 and p[1] == "date":
                    date = p[2]
                elif len(p) >= 4 and p[1] == "player":
                    squads.append({"match_id": mid, "team": p[2],
                                   "player": p[3], "season": season,
                                   "date": date})
        else:
            try:
                b = pd.read_csv(io.BytesIO(z.read(fn)), low_memory=False)
                b["match_id"] = stem
                balls.append(b)
            except Exception:
                continue

    bb = pd.concat(balls, ignore_index=True)
    sq = pd.DataFrame(squads)
    sq["date"] = pd.to_datetime(sq["date"], errors="coerce")
    # Attach dates to deliveries.
    dates = sq.groupby("match_id")["date"].first()
    bb["date"] = bb["match_id"].map(dates)
    return bb, sq


@dataclass
class PlayerFit:
    bat: dict[str, float]      # runs per ball above average
    bowl: dict[str, float]     # runs conceded per ball below average (higher = better)
    bat_balls: dict[str, float]
    bowl_balls: dict[str, float]
    league_rpb: float


def fit_players(bb: pd.DataFrame, as_of: pd.Timestamp,
                prior_balls: float = 60.0, xi: float = 0.0010) -> PlayerFit:
    """Shrunk per-player rates from deliveries strictly before `as_of`."""
    h = bb[bb["date"] < as_of]
    if len(h) < 2000:
        raise ValueError(f"only {len(h)} deliveries before {as_of}")
    w = np.exp(-xi * (as_of - h["date"]).dt.days.to_numpy())
    runs = h["runs_off_bat"].to_numpy(float)
    league = float(np.average(runs, weights=w))

    def shrunk(keys):
        num = defaultdict(float)
        den = defaultdict(float)
        for k, r_, w_ in zip(keys, runs, w):
            if not isinstance(k, str):
                continue
            num[k] += r_ * w_
            den[k] += w_
        out, bal = {}, {}
        for k, d_ in den.items():
            # Shrink toward the league rate by effective ball count.
            out[k] = (num[k] + league * prior_balls) / (d_ + prior_balls) - league
            bal[k] = d_
        return out, bal

    bat, bat_b = shrunk(h["striker"].tolist())
    conc, bowl_b = shrunk(h["bowler"].tolist())
    # A bowler conceding BELOW league average is good, so flip the sign.
    bowl = {k: -v for k, v in conc.items()}
    return PlayerFit(bat, bowl, bat_b, bowl_b, league)


def team_strength(f: PlayerFit, xi_players: list[str]) -> dict:
    """Aggregate an XI into batting and bowling strength, in runs per 100 balls."""
    bat = [f.bat.get(p, 0.0) for p in xi_players]
    bowl = [f.bowl.get(p, 0.0) for p in xi_players]
    # A Hundred innings is 100 balls; the top order faces most of them, so the
    # best batters are weighted more heavily than a flat mean.
    # Squad lists vary in length — T20I entries sometimes include impact
    # substitutes — so the weights are generated rather than hard-coded to 11.
    bat_sorted = sorted(bat, reverse=True)
    n = len(bat_sorted)
    if n == 0:
        return {"bat": 0.0, "bowl": 0.0, "net": 0.0, "n_rated": 0}
    wts = np.exp(-0.28 * np.arange(n))   # top order faces most of the balls
    wts = wts / wts.sum()
    bat_score = float(np.dot(bat_sorted, wts)) * 100.0

    # Bowling: only five bowlers deliver the 100 balls, so take the best five.
    bowl_sorted = sorted(bowl, reverse=True)[:5]
    bowl_score = float(np.mean(bowl_sorted)) * 100.0 if bowl_sorted else 0.0
    return {"bat": bat_score, "bowl": bowl_score,
            "net": bat_score + bowl_score,
            "n_rated": sum(1 for p in xi_players if p in f.bat)}


def predict(f: PlayerFit, xi_a: list[str], xi_b: list[str],
            sigma: float = 26.0) -> dict:
    from scipy.stats import norm
    a = team_strength(f, xi_a)
    b = team_strength(f, xi_b)
    margin = (a["bat"] - b["bowl"]) - (b["bat"] - a["bowl"])
    p_a = float(norm.cdf(margin / sigma))
    return {"p_a": p_a, "p_b": 1 - p_a, "margin": margin,
            "a": a, "b": b}


if __name__ == "__main__":
    bb, sq = download("male")
    RAW.mkdir(parents=True, exist_ok=True)
    bb.to_parquet(RAW / "cricket_hundred_male_balls.parquet", index=False)
    sq.to_parquet(RAW / "cricket_hundred_male_squads.parquet", index=False)
    print(f"deliveries {len(bb):,}   squad rows {len(sq):,}")
    print(f"  players {sq['player'].nunique()}   matches {sq['match_id'].nunique()}")
    print(f"  dates {bb['date'].min().date()} .. {bb['date'].max().date()}")

    f = fit_players(bb, pd.Timestamp("2026-08-01"))
    print(f"\nleague runs/ball {f.league_rpb:.4f}")
    rated = [(k, v, f.bat_balls[k]) for k, v in f.bat.items()
             if f.bat_balls[k] >= 150]
    rated.sort(key=lambda x: -x[1])
    print(f"\n  best batters (150+ weighted balls):")
    for k, v, n in rated[:8]:
        print(f"    {k:<24}{v * 100:+7.1f} runs/100 balls   ({n:.0f} balls)")
    print(f"  worst:")
    for k, v, n in rated[-4:]:
        print(f"    {k:<24}{v * 100:+7.1f} runs/100 balls   ({n:.0f} balls)")

    rb = [(k, v, f.bowl_balls[k]) for k, v in f.bowl.items()
          if f.bowl_balls[k] >= 150]
    rb.sort(key=lambda x: -x[1])
    print(f"\n  best bowlers (150+ weighted balls):")
    for k, v, n in rb[:8]:
        print(f"    {k:<24}{v * 100:+7.1f} runs saved/100   ({n:.0f} balls)")
