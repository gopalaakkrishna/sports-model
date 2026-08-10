"""Score the logged in-play snapshots against what actually happened.

This settles the only question that matters for the in-play model: does it beat
Kalshi's LIVE price, or merely approximate it? Beating a base rate proved it
knows more than nothing. Matching a live market to within a point, as the first
snapshots did, is consistent with knowing nothing the market does not.

Results are joined from MLB StatsAPI by gamePk, so nothing depends on my own
bookkeeping being right.

Snapshots are heavily autocorrelated — consecutive states in one game are nearly
the same bet — so the bootstrap resamples by GAME rather than by row. Resampling
rows would treat 40 snapshots of one blowout as 40 independent observations and
produce a confidence interval several times too narrow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "processed" / "inplay_live_log.csv"
STATS = "https://statsapi.mlb.com/api/v1"
EPS = 1e-15


def ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def outcomes(game_pks) -> dict[int, int]:
    """gamePk -> 1 if the home team won."""
    out = {}
    for pk in game_pks:
        try:
            r = requests.get(f"{STATS}/game/{int(pk)}/linescore", timeout=30)
            if r.status_code != 200:
                continue
            j = r.json()
            t = j.get("teams") or {}
            hs = (t.get("home") or {}).get("runs")
            as_ = (t.get("away") or {}).get("runs")
            if hs is None or as_ is None:
                continue
            # Only count finished games.
            if j.get("currentInning") and not j.get("inningState") == "End":
                pass
            out[int(pk)] = int(hs > as_)
        except requests.RequestException:
            continue
    return out


def main():
    if not LOG.exists():
        print(f"no log yet at {LOG}")
        return
    d = pd.read_csv(LOG)
    d = d.dropna(subset=["mkt_home", "model_home"])
    if d.empty:
        print("no snapshots with a market price")
        return

    print(f"{len(d):,} snapshots across {d['gamePk'].nunique()} games")
    res = outcomes(d["gamePk"].unique())
    d["home_won"] = d["gamePk"].map(res)
    d = d.dropna(subset=["home_won"])
    d["home_won"] = d["home_won"].astype(int)

    # Drop games still in progress: a game is only usable once settled.
    print(f"  {d['gamePk'].nunique()} games with a final result, "
          f"{len(d):,} usable snapshots")
    if len(d) < 20:
        print("  too few to score yet — let the watcher run longer")
        return

    m, k, y = (d["model_home"].to_numpy(), d["mkt_home"].to_numpy(),
               d["home_won"].to_numpy())
    print(f"\n  {'variant':<22}{'log loss':>11}")
    print(f"  {'model (in-play)':<22}{ll(m, y):>11.5f}")
    print(f"  {'Kalshi live price':<22}{ll(k, y):>11.5f}")
    gap = ll(m, y) - ll(k, y)
    print(f"  {'gap':<22}{gap:>+11.5f}   "
          f"({'model better' if gap < 0 else 'MARKET BETTER'})")

    # Bootstrap by GAME, not by snapshot — consecutive states are near-duplicates.
    games = d["gamePk"].unique()
    per_game = {g: d[d["gamePk"] == g] for g in games}
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(3000):
        pick = rng.choice(games, len(games), replace=True)
        sub = pd.concat([per_game[g] for g in pick], ignore_index=True)
        diffs.append(ll(sub["model_home"], sub["home_won"])
                     - ll(sub["mkt_home"], sub["home_won"]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  95% CI on the gap (resampled by game) [{lo:+.5f}, {hi:+.5f}]")
    if hi < 0:
        print("  -> the model beats the live market")
    elif lo > 0:
        print("  -> the live market is better")
    else:
        print("  -> indistinguishable at this sample size")

    print(f"\n  BY DISAGREEMENT SIZE")
    d["dis"] = (d["model_home"] - d["mkt_home"]).abs()
    qs = np.quantile(d["dis"], [0, .5, .8, .95, 1.0])
    print(f"    {'|model-mkt|':<16}{'n':>7}{'model':>10}{'market':>10}{'gap':>10}")
    for i in range(len(qs) - 1):
        sel = (d["dis"] >= qs[i]) & (
            (d["dis"] < qs[i + 1]) if i < len(qs) - 2 else (d["dis"] <= qs[i + 1]))
        if sel.sum() < 15:
            continue
        s = d[sel]
        a_, b_ = ll(s["model_home"], s["home_won"]), ll(s["mkt_home"], s["home_won"])
        print(f"    {f'{qs[i]:.3f}-{qs[i+1]:.3f}':<16}{int(sel.sum()):>7}"
              f"{a_:>10.4f}{b_:>10.4f}{a_ - b_:>+10.4f}")

    print(f"\n  BY GAME STAGE")
    d["stage"] = pd.cut(d["inning"], [0, 3, 6, 9, 99],
                        labels=["1-3", "4-6", "7-9", "extras"])
    for st, g in d.groupby("stage", observed=True):
        if len(g) < 15:
            continue
        a_, b_ = ll(g["model_home"], g["home_won"]), ll(g["mkt_home"], g["home_won"])
        print(f"    {str(st):<10}{len(g):>6}  model {a_:.4f}  market {b_:.4f}  "
              f"gap {a_ - b_:+.4f}")

    print(f"\n  NOTE: snapshots within a game are highly correlated. {d['gamePk'].nunique()}")
    print("  settled games is the real sample size, not the snapshot count.")


if __name__ == "__main__":
    main()
