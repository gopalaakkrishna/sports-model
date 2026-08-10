"""Score the MLB model against Kalshi's pre-game closing line.

This is the test the MLB model has been missing. Everything before it measured
the model against the base rate, which says whether the model knows anything at
all — not whether it knows anything the market doesn't.

The model probabilities come from the walk-forward backtest (fitted only on
prior data). The Platt calibration was fitted on games through 2024, and the
Kalshi closes only exist from mid-2026, so applying it here is out of sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-15


def ll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    preds = pd.read_parquet(ROOT / "data" / "processed" / "mlb_backtest.parquet")
    closes = pd.read_parquet(ROOT / "data" / "raw" / "kalshi_mlb_closes.parquet")

    preds["date"] = pd.to_datetime(preds["date"]).dt.date.astype(str)
    closes["date"] = closes["date"].astype(str)

    d = preds.merge(closes, on=["date", "home", "away"],
                    how="inner", suffixes=("", "_k"))
    if d.empty:
        print("no overlap between backtest predictions and Kalshi closes")
        return
    # Sanity: the two sources must agree on who won.
    mism = int((d["home_win"] != d["home_win_k"]).sum()) if "home_win_k" in d else 0
    print(f"matched {len(d):,} games "
          f"({d['date'].min()} .. {d['date'].max()})")
    if mism:
        print(f"  !! {mism} games where MLB result and Kalshi settlement disagree "
              f"— dropping them")
        d = d[d["home_win"] == d["home_win_k"]]

    cal = json.loads((ROOT / "data" / "processed" / "mlb_calibration.json").read_text())
    a, b = cal["a"], cal["b"]

    def logit(p):
        p = np.clip(p, EPS, 1 - EPS)
        return np.log(p / (1 - p))

    raw = d["p_home"].to_numpy()
    model = 1 / (1 + np.exp(-(a * logit(raw) + b)))
    # Normalise the two closing prices to remove Kalshi's small spread.
    mkt = (d["close_home"] / (d["close_home"] + d["close_away"])).to_numpy()
    y = d["home_win"].to_numpy()
    base = np.full(len(y), y.mean())

    print(f"\n  mean Kalshi book sum {d['book_sum'].mean():.4f} "
          f"(spread {100 * (d['book_sum'].mean() - 1):.2f}%)")
    print(f"  home win rate {y.mean():.3%}, market mean {mkt.mean():.3%}, "
          f"model mean {model.mean():.3%}")

    rows = [
        ("base rate", ll(base, y)),
        ("model (raw)", ll(raw, y)),
        (f"model (calibrated a={a:.3f})", ll(model, y)),
        ("Kalshi pre-game close", ll(mkt, y)),
    ]
    print(f"\n  LOG LOSS\n  {'variant':<30}{'log loss':>10}{'vs market':>12}")
    mk = ll(mkt, y)
    for n, v in rows:
        print(f"  {n:<30}{v:>10.5f}{v - mk:>+12.5f}")

    best_w, best = 0.0, mk
    for w in np.arange(0, 1.0001, 0.01):
        v = ll(w * model + (1 - w) * mkt, y)
        if v < best - 1e-12:
            best_w, best = float(w), v
    print(f"\n  optimal blend weight on model: {best_w:.2f} -> {best:.5f}")

    # Bootstrap the model-minus-market difference to see if it is distinguishable
    # from noise at this sample size.
    li_m = -np.log(np.clip(np.where(y == 1, model, 1 - model), EPS, 1))
    li_k = -np.log(np.clip(np.where(y == 1, mkt, 1 - mkt), EPS, 1))
    diff = li_m - li_k
    rng = np.random.default_rng(0)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                     for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  mean per-game difference {diff.mean():+.5f}, 95% CI [{lo:+.5f}, {hi:+.5f}]")
    if lo < 0 < hi:
        print("  -> not distinguishable from the market at this sample size")
    elif diff.mean() > 0:
        print("  -> the market is significantly better")
    else:
        print("  -> the MODEL is significantly better")

    print(f"\n  DISAGREEMENT BUCKETS")
    dis = np.abs(model - mkt)
    qs = np.quantile(dis, [0, 0.5, 0.8, 0.95, 1.0])
    print(f"    {'|model-market|':<18}{'n':>6}{'model':>10}{'market':>10}{'gap':>10}")
    for i in range(len(qs) - 1):
        m = (dis >= qs[i]) & ((dis < qs[i + 1]) if i < len(qs) - 2 else (dis <= qs[i + 1]))
        if m.sum() < 20:
            continue
        print(f"    {f'{qs[i]:.3f}-{qs[i+1]:.3f}':<18}{m.sum():>6}"
              f"{ll(model[m], y[m]):>10.4f}{ll(mkt[m], y[m]):>10.4f}"
              f"{ll(model[m], y[m]) - ll(mkt[m], y[m]):>+10.4f}")

    print(f"\n  NOTE: {len(d)} games is a small sample. Detecting a genuine 0.01")
    print("  log-loss edge reliably needs thousands. Read the CI, not the point estimate.")


if __name__ == "__main__":
    main()
