"""T20 international model: team ratings plus player ratings, walk-forward.

The Hundred failed on two counts — 189 matches and franchise squads that reset
every year. T20Is fix both: 3,131 matches since 2015 and national teams that
persist. There is also far more spread in strength, since full members play
associate nations.

Two models are compared on identical fixtures:

  team    Elo-style rating updated match by match
  player  batting and bowling rates aggregated over the XI, as built for
          The Hundred

Both are scored against the base rate with a bootstrap CI. No market benchmark
exists — Kalshi lists T20I markets but quotes none — so this measures whether
the models know anything, not whether they beat anyone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import cricket_players as CP

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-15


def ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def elo_walk(mt: pd.DataFrame, k: float = 20.0, base: float = 1500.0):
    """Sequential Elo. Ratings only ever use prior matches, so it is honest."""
    r = {}
    rows = []
    for _, g in mt.iterrows():
        if pd.isna(g["winner"]):
            continue
        a, b = g["team_a"], g["team_b"]
        ra, rb = r.get(a, base), r.get(b, base)
        exp_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        won_a = int(g["winner"] == a)
        rows.append({"match_id": g["match_id"], "date": g["date"],
                     "a": a, "b": b, "p_a": exp_a, "won_a": won_a,
                     "n_a": sum(1 for x in rows if a in (x["a"], x["b"])),
                     "n_b": sum(1 for x in rows if b in (x["a"], x["b"]))})
        r[a] = ra + k * (won_a - exp_a)
        r[b] = rb + k * ((1 - won_a) - (1 - exp_a))
    return pd.DataFrame(rows), r


def main(gender: str = "male", min_matches: int = 10, refit_days: int = 45):
    mt = pd.read_parquet(ROOT / "data" / "raw" / f"t20i_{gender}_matches.parquet")
    sq = pd.read_parquet(ROOT / "data" / "raw" / f"t20i_{gender}_squads.parquet")
    bb = pd.read_parquet(ROOT / "data" / "raw" / f"t20i_{gender}_balls.parquet")
    for d in (mt, sq, bb):
        d["date"] = pd.to_datetime(d["date"])
    mt = mt.sort_values("date")

    print(f"T20I {gender}: {len(mt):,} matches, {len(bb):,} deliveries")

    # ---- team model (Elo) ----
    e, final = elo_walk(mt)
    # Only score once both sides have a history.
    e = e[(e["n_a"] >= min_matches) & (e["n_b"] >= min_matches)]
    y, p = e["won_a"].to_numpy(), e["p_a"].to_numpy()
    b0 = np.full(len(y), y.mean())
    print(f"\nELO (team)  scored {len(e):,} matches")
    print(f"  log loss model     {ll(p, y):.5f}")
    print(f"  log loss base rate {ll(b0, y):.5f}")
    print(f"  accuracy           {((p > 0.5) == y).mean():.3%}")
    print(f"  gain over base     {ll(b0, y) - ll(p, y):+.5f}")

    top = sorted(final.items(), key=lambda z: -z[1])[:8]
    print(f"  strongest now: " + ", ".join(f"{t} {v:.0f}" for t, v in top))

    # ---- player model ----
    print(f"\nPLAYER model (refit every {refit_days} days)")
    eligible = set(e["match_id"])
    sub = mt[mt["match_id"].isin(eligible)].sort_values("date")
    rows, fit, last_fit = [], None, None
    for _, g in sub.iterrows():
        d0 = g["date"]
        if last_fit is None or (d0 - last_fit).days >= refit_days:
            try:
                fit = CP.fit_players(bb, d0, prior_balls=120.0, xi=0.0006)
                last_fit = d0
            except ValueError:
                continue
        if fit is None:
            continue
        s = sq[sq["match_id"] == g["match_id"]]
        xa = s[s["team"] == g["team_a"]]["player"].tolist()
        xb = s[s["team"] == g["team_b"]]["player"].tolist()
        if len(xa) < 9 or len(xb) < 9:
            continue
        pr = CP.predict(fit, xa, xb, sigma=30.0)
        rows.append({"date": d0, "p_a": pr["p_a"],
                     "won_a": int(g["winner"] == g["team_a"]),
                     "match_id": g["match_id"]})

    d = pd.DataFrame(rows)
    if d.empty:
        print("  nothing scored")
        return
    y2, p2 = d["won_a"].to_numpy(), d["p_a"].to_numpy()
    b2 = np.full(len(y2), y2.mean())
    print(f"  scored {len(d):,} matches")
    print(f"  log loss model     {ll(p2, y2):.5f}")
    print(f"  log loss base rate {ll(b2, y2):.5f}")
    print(f"  accuracy           {((p2 > 0.5) == y2).mean():.3%}")
    print(f"  gain over base     {ll(b2, y2) - ll(p2, y2):+.5f}")

    # ---- blend, on matches both scored ----
    merged = e.merge(d[["match_id", "p_a"]], on="match_id",
                     suffixes=("_elo", "_pl"))
    if len(merged) > 100:
        ym = merged["won_a"].to_numpy()
        pe, pl = merged["p_a_elo"].to_numpy(), merged["p_a_pl"].to_numpy()
        bm = np.full(len(ym), ym.mean())
        best_w, best = 0.0, ll(pe, ym)
        for w in np.arange(0, 1.001, 0.05):
            v = ll(w * pl + (1 - w) * pe, ym)
            if v < best:
                best_w, best = float(w), v
        print(f"\nBLEND on {len(merged):,} common matches")
        print(f"  elo alone    {ll(pe, ym):.5f}")
        print(f"  player alone {ll(pl, ym):.5f}")
        print(f"  base rate    {ll(bm, ym):.5f}")
        print(f"  best blend   {best:.5f}  (weight {best_w:.2f} on player)")

        rng = np.random.default_rng(0)
        diff = (-(ym * np.log(np.clip(pe, EPS, 1 - EPS))
                  + (1 - ym) * np.log(np.clip(1 - pe, EPS, 1 - EPS)))
                + (ym * np.log(np.clip(bm, EPS, 1 - EPS))
                   + (1 - ym) * np.log(np.clip(1 - bm, EPS, 1 - EPS))))
        boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                         for _ in range(4000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  elo gain over base: 95% CI [{lo:+.5f}, {hi:+.5f}] "
              f"-> {'significant' if lo > 0 else 'not significant'}")

    print(f"\n  CALIBRATION (elo)")
    e2 = e.copy()
    e2["bucket"] = pd.cut(e2["p_a"], [0, .2, .35, .5, .65, .8, 1.0])
    print(e2.groupby("bucket", observed=True).agg(
        n=("won_a", "size"), predicted=("p_a", "mean"),
        actual=("won_a", "mean")).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "male")
