"""Evaluate against the market-average closing line as well as Pinnacle.

football-data.co.uk stopped publishing Pinnacle odds during the 2025/26 season
(coverage decays from Nov 2025, zero from Feb 2026), so the Pinnacle benchmark
cannot cover recent matches and will not be available for future fixtures at
all. The average across books (AvgH/D/A) has complete coverage.

The average line is a slightly weaker forecast than Pinnacle - it blends sharp
and soft books, and carries a larger overround - so the model should look
comparatively better against it. If the model still loses to the AVERAGE line,
the conclusion is robust to which benchmark is used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B

ROOT = Path(__file__).resolve().parents[1]


def prep(d, odds_cols):
    d = d.dropna(subset=odds_cols).copy()
    d["y"] = d["FTR"].map({"H": 0, "D": 1, "A": 2})
    d = d.dropna(subset=["y"])
    d["y"] = d["y"].astype(int)
    return d


def report(d, odds_cols, label):
    mp = d[["m_home", "m_draw", "m_away"]].to_numpy(float)
    mp = mp / mp.sum(axis=1, keepdims=True)
    kp = B.devig_proportional(d[odds_cols].to_numpy(float))
    y = d["y"].to_numpy(int)

    ll, mk = B.log_loss(mp, y), B.log_loss(kp, y)
    best_w, best = 0.0, mk
    for w in np.arange(0, 1.0001, 0.01):
        v = B.log_loss(w * mp + (1 - w) * kp, y)
        if v < best - 1e-12:
            best_w, best = float(w), v

    over = 100 * (1 / d[odds_cols].to_numpy(float)).sum(axis=1).mean() - 100
    print(f"\n  {label}")
    print(f"    matches           {len(d):,}  "
          f"({d['Date'].min().date()} .. {d['Date'].max().date()})")
    print(f"    overround         {over:.2f}%")
    print(f"    model log loss    {ll:.5f}")
    print(f"    market log loss   {mk:.5f}")
    print(f"    gap               {ll - mk:+.5f}"
          f"  ({'model better' if ll < mk else 'market better'})")
    print(f"    blend weight      {best_w:.2f}")
    return {"label": label, "n": len(d), "model": ll, "market": mk,
            "gap": ll - mk, "w": best_w}


def main():
    d = pd.read_parquet(ROOT / "data" / "processed" / "backtest_holdout.parquet")
    print("Holdout window 2022-08-01 .. 2026-08-03, tuned params (xi=0.0025, reg=1.0)")

    ps = prep(d, ["PSH", "PSD", "PSA"])
    report(ps, ["PSH", "PSD", "PSA"], "vs PINNACLE closing (coverage ends 2026-01-15)")

    avg = prep(d, ["AvgH", "AvgD", "AvgA"])
    report(avg, ["AvgH", "AvgD", "AvgA"], "vs MARKET AVERAGE closing (full coverage)")

    # Same matches, both benchmarks, so the two are directly comparable.
    both = prep(d, ["PSH", "PSD", "PSA", "AvgH", "AvgD", "AvgA"])
    print("\n  --- restricted to matches where BOTH are available ---")
    report(both, ["PSH", "PSD", "PSA"], "vs Pinnacle (matched sample)")
    report(both, ["AvgH", "AvgD", "AvgA"], "vs market average (matched sample)")

    # The period Pinnacle no longer covers.
    recent = avg[avg["Date"] > "2026-01-15"]
    if len(recent) > 200:
        print("\n  --- period with no Pinnacle data at all ---")
        report(recent, ["AvgH", "AvgD", "AvgA"], "vs market average, after 2026-01-15")


if __name__ == "__main__":
    main()
