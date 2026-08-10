"""Does the starting-quarterback term earn its place in the NFL model?

Same discipline as every other feature in this project: run the identical
walk-forward backtest with and without it, on exactly the same games, and put a
bootstrap confidence interval on the paired difference.

The prior is genuinely favourable here — starting QB is widely held to be the
largest single factor in an NFL game, and the equivalent term (starting pitcher)
is real in the MLB model. But "widely held" is how altitude and per-team home
advantage got into this project, and both were wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import margin_model as MM

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-15


def ll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def run(g: pd.DataFrame, start, end, xi, reg, reg_qb, step_days=7):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    rows, cursor = [], s
    while cursor <= e:
        nxt = cursor + pd.Timedelta(days=step_days)
        wk = g[(g["date"] >= cursor) & (g["date"] < nxt)]
        if wk.empty:
            cursor = nxt
            continue
        try:
            f = MM.fit(g, cursor, xi=xi, reg=reg, reg_qb=reg_qb)
        except ValueError:
            cursor = nxt
            continue
        for _, m in wk.iterrows():
            sp = m.get("spread_line")
            p = MM.predict(f, m["home_team"], m["away_team"],
                           spread=float(sp) if pd.notna(sp) else None,
                           home_qb=m.get("home_qb_name"),
                           away_qb=m.get("away_qb_name"))
            if p is None:
                continue
            rows.append({
                "key": f"{m['date']}|{m['home_team']}|{m['away_team']}",
                "home_win": int(m["margin"] > 0),
                "margin": m["margin"], "spread_line": sp,
                "p_home": p["p_home"], "exp_margin": p["exp_margin"],
                "p_home_covers": p.get("p_home_covers"),
                "qb_h": p["qb_home_effect"], "qb_a": p["qb_away_effect"],
            })
        cursor = nxt
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-09-01")
    ap.add_argument("--end", default="2026-08-01")
    ap.add_argument("--variants", default="0,4,12,30",
                    help="reg_qb values; 0 = QB term off")
    args = ap.parse_args()

    g = pd.read_parquet(ROOT / "data" / "raw" / "nfl_games.parquet")
    g = g[g["played"]].copy()
    g["date"] = pd.to_datetime(g["date"])
    print(f"NFL QB ablation, {args.start} .. {args.end}")
    print(f"  QB names present on {g['home_qb_name'].notna().sum():,} games\n")

    runs = {}
    for v in [float(x) for x in args.variants.split(",")]:
        label = "off" if v == 0 else f"reg_qb={v:g}"
        d = run(g, args.start, args.end, 0.0025, 8.0, v)
        runs[label] = d
        print(f"  {label:<14} {len(d):,} predictions")

    common = None
    for d in runs.values():
        common = set(d["key"]) if common is None else common & set(d["key"])
    aligned = {k: d[d["key"].isin(common)].sort_values("key").reset_index(drop=True)
               for k, d in runs.items()}
    base = aligned["off"]
    y = base["home_win"].to_numpy()
    print(f"\n  common games: {len(base):,}")

    print(f"\n  {'variant':<14}{'ML logloss':>12}{'vs off':>10}"
          f"{'spread %':>11}{'margin MAE':>12}")
    per = {}
    for k, d in aligned.items():
        p = d["p_home"].to_numpy()
        v = ll(p, y)
        per[k] = -np.log(np.clip(np.where(y == 1, p, 1 - p), EPS, 1))
        sp = d.dropna(subset=["spread_line", "p_home_covers"])
        cov = (sp["margin"] - sp["spread_line"] > 0).astype(int)
        live = (sp["margin"] - sp["spread_line"]) != 0
        acc = ((sp.loc[live, "p_home_covers"] > 0.5).astype(int)
               == cov[live]).mean()
        mae = np.abs(d["margin"] - d["exp_margin"]).mean()
        delta = "" if k == "off" else f"{v - ll(base['p_home'].to_numpy(), y):+.5f}"
        print(f"  {k:<14}{v:>12.5f}{delta:>10}{acc:>10.2%}{mae:>12.2f}")

    rng = np.random.default_rng(0)
    for k in aligned:
        if k == "off":
            continue
        diff = per[k] - per["off"]
        boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                         for _ in range(4000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        verdict = "HELPS" if hi < 0 else "HURTS" if lo > 0 else "no effect"
        print(f"\n  {k} vs off: mean {diff.mean():+.5f}, "
              f"95% CI [{lo:+.5f}, {hi:+.5f}]  -> {verdict}")

    best = min(aligned, key=lambda k: ll(aligned[k]["p_home"].to_numpy(), y))
    if best != "off":
        d = aligned[best]
        spread = (d["qb_h"] - d["qb_a"])
        print(f"\n  under {best}, QB differential spans "
              f"{spread.min():+.2f} to {spread.max():+.2f} points")


if __name__ == "__main__":
    main()
