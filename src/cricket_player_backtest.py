"""Does player-level rating beat team-level for The Hundred?

Team-level scored 0.6932 against a 0.6924 base rate — worse than guessing —
because franchise strength barely persists (year-over-year correlation +0.142).
Players persist far better, so this tests whether rating the eleven who actually
played does better.

An important caveat on interpretation: this uses the ACTUAL XI, which is known
only after the toss. It therefore measures whether player ratings carry signal at
all, not what could have been forecast a day ahead. If it fails even with the
true XI, no lineup-guessing scheme can rescue it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import cricket_model as CM
import cricket_players as CP

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-15


def ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main(sigma: float = 26.0):
    bb = pd.read_parquet(ROOT / "data" / "raw" / "cricket_hundred_male_balls.parquet")
    sq = pd.read_parquet(ROOT / "data" / "raw" / "cricket_hundred_male_squads.parquet")
    mt = pd.read_parquet(ROOT / "data" / "raw" / "cricket_hundred_male_matches.parquet")
    mt["date"] = pd.to_datetime(mt["date"], errors="coerce")
    mt["match_id"] = mt["match_id"].astype(str)
    sq["match_id"] = sq["match_id"].astype(str)
    bb["date"] = pd.to_datetime(bb["date"])

    mt["home_team"] = mt["home_team"].map(CM.canon)
    mt["away_team"] = mt["away_team"].map(CM.canon)
    mt["winner"] = mt["winner"].map(lambda x: CM.canon(x) if pd.notna(x) else x)
    sq["team"] = sq["team"].map(CM.canon)

    played = mt[mt["winner"].notna()].sort_values("date")
    rows = []
    for _, g in played.iterrows():
        squads = sq[sq["match_id"] == g["match_id"]]
        if squads["team"].nunique() != 2:
            continue
        try:
            f = CP.fit_players(bb, g["date"])
        except ValueError:
            continue
        a, b = g["home_team"], g["away_team"]
        xi_a = squads[squads["team"] == a]["player"].tolist()
        xi_b = squads[squads["team"] == b]["player"].tolist()
        if len(xi_a) < 8 or len(xi_b) < 8:
            continue
        p = CP.predict(f, xi_a, xi_b, sigma=sigma)
        rows.append({
            "date": g["date"], "season": g["season"],
            "a": a, "b": b, "p_a": p["p_a"],
            "won_a": int(g["winner"] == a),
            "margin": p["margin"],
            "rated_a": p["a"]["n_rated"], "rated_b": p["b"]["n_rated"],
        })

    d = pd.DataFrame(rows)
    if d.empty:
        print("nothing scored")
        return
    y, pp = d["won_a"].to_numpy(), d["p_a"].to_numpy()
    base = np.full(len(y), y.mean())
    print(f"PLAYER-LEVEL MODEL — {len(d)} matches "
          f"({d['date'].min().date()} .. {d['date'].max().date()})")
    print(f"  log loss model     {ll(pp, y):.5f}")
    print(f"  log loss base rate {ll(base, y):.5f}")
    print(f"  accuracy           {((pp > 0.5) == y).mean():.3%}")
    print(f"  prob spread sd     {pp.std():.4f} "
          f"(range {pp.min():.1%}-{pp.max():.1%})")
    print(f"\n  team-level model was 0.69478 vs base 0.69235 (worse than guessing)")
    gain = ll(base, y) - ll(pp, y)
    print(f"  player model vs base rate: {gain:+.5f} "
          f"({'BETTER' if gain > 0 else 'worse'})")

    rng = np.random.default_rng(0)
    diff = (-(y * np.log(np.clip(pp, EPS, 1 - EPS))
              + (1 - y) * np.log(np.clip(1 - pp, EPS, 1 - EPS)))
            + (y * np.log(np.clip(base, EPS, 1 - EPS))
               + (1 - y) * np.log(np.clip(1 - base, EPS, 1 - EPS))))
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                     for _ in range(5000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  95% CI on the gain [{lo:+.5f}, {hi:+.5f}]  "
          f"-> {'significant' if lo > 0 else 'not significant'}")

    print(f"\n  BY SEASON")
    for s, g in d.groupby("season"):
        if len(g) < 8:
            continue
        yy, qq = g["won_a"].to_numpy(), g["p_a"].to_numpy()
        print(f"    {s}  n={len(g):>3}  logloss {ll(qq, yy):.4f}  "
              f"acc {((qq > 0.5) == yy).mean():.1%}")

    print(f"\n  CALIBRATION")
    d["bucket"] = pd.cut(d["p_a"], [0, .35, .45, .55, .65, 1.0])
    print(d.groupby("bucket", observed=True).agg(
        n=("won_a", "size"), predicted=("p_a", "mean"),
        actual=("won_a", "mean")).round(3).to_string())


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 26.0)
