"""Is the MLB blend weight real, or fitted noise?

The in-sample optimal blend put 0.33 weight on the model against Kalshi's
closing line — the first non-zero weight anywhere in this project. But that
weight was chosen on the same 907 games it was scored on, which is exactly how a
spurious edge gets manufactured.

Two checks here:
  1. Fit the blend weight on an earlier slice, apply it to a later one.
  2. Bootstrap the out-of-sample improvement to get a confidence interval.

A weight that survives both is worth something. One that does not is noise, and
saying so is the whole point of running this.
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


def best_weight(model, mkt, y):
    bw, bl = 0.0, ll(mkt, y)
    for w in np.arange(0, 1.0001, 0.01):
        v = ll(w * model + (1 - w) * mkt, y)
        if v < bl - 1e-12:
            bw, bl = float(w), v
    return bw, bl


def main():
    preds = pd.read_parquet(ROOT / "data" / "processed" / "mlb_backtest.parquet")
    closes = pd.read_parquet(ROOT / "data" / "raw" / "kalshi_mlb_closes.parquet")
    preds["date"] = pd.to_datetime(preds["date"]).dt.date.astype(str)
    closes["date"] = closes["date"].astype(str)
    d = preds.merge(closes, on=["date", "home", "away"], how="inner",
                    suffixes=("", "_k"))
    d = d[d["home_win"] == d["home_win_k"]].sort_values("date").reset_index(drop=True)

    cal = json.loads((ROOT / "data" / "processed" / "mlb_calibration.json").read_text())
    a, b = cal["a"], cal["b"]

    def logit(p):
        p = np.clip(p, EPS, 1 - EPS)
        return np.log(p / (1 - p))

    d["model"] = 1 / (1 + np.exp(-(a * logit(d["p_home"].to_numpy()) + b)))
    d["mkt"] = d["close_home"] / (d["close_home"] + d["close_away"])

    print(f"{len(d)} games, {d['date'].min()} .. {d['date'].max()}")

    # --- 1. temporal split ---
    k = len(d) // 2
    tr, te = d.iloc[:k], d.iloc[k:]
    w_tr, ll_tr = best_weight(tr["model"].to_numpy(), tr["mkt"].to_numpy(),
                              tr["home_win"].to_numpy())
    m_te, k_te, y_te = (te["model"].to_numpy(), te["mkt"].to_numpy(),
                        te["home_win"].to_numpy())
    ll_mkt_te = ll(k_te, y_te)
    ll_blend_te = ll(w_tr * m_te + (1 - w_tr) * k_te, y_te)
    w_te, _ = best_weight(m_te, k_te, y_te)

    print(f"\n  TEMPORAL SPLIT")
    print(f"    fit   {tr['date'].min()} .. {tr['date'].max()}  n={len(tr)}  "
          f"best w={w_tr:.2f}")
    print(f"    test  {te['date'].min()} .. {te['date'].max()}  n={len(te)}  "
          f"(its own best w would be {w_te:.2f})")
    print(f"    market alone on test      {ll_mkt_te:.5f}")
    print(f"    blend at w={w_tr:.2f} on test  {ll_blend_te:.5f}")
    print(f"    out-of-sample improvement {ll_mkt_te - ll_blend_te:+.5f}"
          f"  ({'better' if ll_blend_te < ll_mkt_te else 'WORSE'})")

    # --- 2. bootstrap the OOS improvement ---
    li_b = -np.log(np.clip(np.where(y_te == 1, w_tr * m_te + (1 - w_tr) * k_te,
                                    1 - (w_tr * m_te + (1 - w_tr) * k_te)), EPS, 1))
    li_k = -np.log(np.clip(np.where(y_te == 1, k_te, 1 - k_te), EPS, 1))
    diff = li_k - li_b   # positive = blend better
    rng = np.random.default_rng(0)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                     for _ in range(5000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"    mean per-game gain {diff.mean():+.5f}, 95% CI [{lo:+.5f}, {hi:+.5f}]")
    verdict = ("blend genuinely helps" if lo > 0 else
               "blend genuinely hurts" if hi < 0 else
               "indistinguishable from noise")
    print(f"    -> {verdict}")

    # --- 3. how stable is the weight across halves and months? ---
    print(f"\n  WEIGHT STABILITY")
    d["month"] = d["date"].str[:7]
    for mo, g in d.groupby("month"):
        if len(g) < 60:
            continue
        w, _ = best_weight(g["model"].to_numpy(), g["mkt"].to_numpy(),
                           g["home_win"].to_numpy())
        gap = ll(g["model"].to_numpy(), g["home_win"].to_numpy()) - \
              ll(g["mkt"].to_numpy(), g["home_win"].to_numpy())
        print(f"    {mo}  n={len(g):>4}  best w={w:.2f}  model-market {gap:+.5f}")
    print("\n  A weight that swings wildly month to month is being fitted to noise.")


if __name__ == "__main__":
    main()
