"""Where, if anywhere, does the model compete with the market?

The pooled result says the closing line wins. That is the expected answer for
top divisions, which are heavily bet and sharply priced. This script breaks the
backtest down to look for pockets where the model holds up:

  * by division   - lower/less-followed leagues get less bookmaker attention
  * by era        - has the market got sharper over time?
  * by confidence - does the model do better when it disagrees mildly vs wildly?
  * over/under    - totals markets are priced differently from 1X2

A positive result here is only interesting if the sample is large and the blend
weight is meaningfully above zero. Small-sample division-level wins are exactly
what overfitting looks like, so n is reported everywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-15


def prep(d):
    d = d.dropna(subset=["PSH", "PSD", "PSA"]).copy()
    d["y"] = d["FTR"].map({"H": 0, "D": 1, "A": 2})
    d = d.dropna(subset=["y"])
    d["y"] = d["y"].astype(int)
    return d


def parts(d):
    mp = d[["m_home", "m_draw", "m_away"]].to_numpy(float)
    mp = mp / mp.sum(axis=1, keepdims=True)
    kp = B.devig_proportional(d[["PSH", "PSD", "PSA"]].to_numpy(float))
    return mp, kp, d["y"].to_numpy(int)


def best_blend(mp, kp, y):
    best_w, best = 0.0, B.log_loss(kp, y)
    for w in np.arange(0, 1.0001, 0.01):
        ll = B.log_loss(w * mp + (1 - w) * kp, y)
        if ll < best - 1e-12:
            best_w, best = float(w), ll
    return best_w, best


def section(title):
    print(f"\n{title}\n{'=' * len(title)}")


def main():
    d = prep(pd.read_parquet(ROOT / "data" / "processed" / "backtest_preds.parquet"))

    section("BY DIVISION (sorted by how close the model gets to the market)")
    print(f"  {'div':<6}{'n':>7}{'model':>9}{'market':>9}{'gap':>9}{'blend w':>9}")
    rows = []
    for div, g in d.groupby("Div"):
        if len(g) < 500:
            continue
        mp, kp, y = parts(g)
        ll, mk = B.log_loss(mp, y), B.log_loss(kp, y)
        w, _ = best_blend(mp, kp, y)
        rows.append((div, len(g), ll, mk, ll - mk, w))
    for div, n, ll, mk, gap, w in sorted(rows, key=lambda r: r[4]):
        star = "  <-- model competitive" if gap < 0.002 else ""
        print(f"  {div:<6}{n:>7,}{ll:>9.4f}{mk:>9.4f}{gap:>+9.4f}{w:>9.2f}{star}")

    section("BY ERA")
    print(f"  {'period':<12}{'n':>8}{'model':>9}{'market':>9}{'gap':>9}")
    d["yr"] = d["Date"].dt.year
    for lo, hi in [(2015, 2017), (2018, 2020), (2021, 2023), (2024, 2026)]:
        g = d[(d["yr"] >= lo) & (d["yr"] <= hi)]
        if len(g) < 500:
            continue
        mp, kp, y = parts(g)
        ll, mk = B.log_loss(mp, y), B.log_loss(kp, y)
        print(f"  {f'{lo}-{hi}':<12}{len(g):>8,}{ll:>9.4f}{mk:>9.4f}{ll - mk:>+9.4f}")

    section("BY SIZE OF DISAGREEMENT WITH THE MARKET")
    mp, kp, y = parts(d)
    disagree = np.abs(mp - kp).max(axis=1)
    qs = np.quantile(disagree, [0, 0.25, 0.5, 0.75, 0.9, 1.0])
    print(f"  {'disagreement':<16}{'n':>8}{'model':>9}{'market':>9}{'gap':>9}")
    for i in range(len(qs) - 1):
        m = (disagree >= qs[i]) & (disagree < qs[i + 1] if i < len(qs) - 2 else True)
        if m.sum() < 200:
            continue
        ll, mk = B.log_loss(mp[m], y[m]), B.log_loss(kp[m], y[m])
        print(f"  {f'{qs[i]:.3f}-{qs[i+1]:.3f}':<16}{m.sum():>8,}{ll:>9.4f}{mk:>9.4f}{ll - mk:>+9.4f}")

    section("FLAT-STAKE SIMULATION AT BEST AVAILABLE ODDS")
    print("  Betting every outcome where model prob * best odds - 1 > threshold.")
    print("  Uses MaxH/D/A (best price across books) — the most generous")
    print("  assumption possible, and still the realistic ceiling is lower")
    print("  because best prices get limited or moved.\n")
    mx = d[["MaxH", "MaxD", "MaxA"]].to_numpy(float)
    ok = np.isfinite(mx).all(axis=1)
    mp_o, y_o, mx_o = mp[ok], y[ok], mx[ok]
    onehot = np.zeros_like(mp_o)
    onehot[np.arange(len(y_o)), y_o] = 1.0
    ev = mp_o * mx_o - 1.0
    print(f"  {'threshold':<12}{'bets':>9}{'staked':>10}{'profit':>10}{'ROI':>9}")
    for thr in [0.0, 0.02, 0.05, 0.10, 0.20]:
        sel = ev > thr
        n = int(sel.sum())
        if n == 0:
            print(f"  {f'>{thr:.0%}':<12}{0:>9}{'-':>10}{'-':>10}{'-':>9}")
            continue
        ret = (onehot[sel] * mx_o[sel]).sum() - n
        print(f"  {f'>{thr:.0%}':<12}{n:>9,}{n:>10,}{ret:>+10.0f}{ret / n:>+9.2%}")

    section("OVER/UNDER 2.5 GOALS")
    tot = d["FTHG"] + d["FTAG"]
    over = (tot > 2.5).to_numpy().astype(int)
    p_over = d["m_over25"].to_numpy(float)
    p2 = np.column_stack([1 - p_over, p_over])
    ll = B.log_loss(p2, over)
    base = np.tile([1 - over.mean(), over.mean()], (len(over), 1))
    print(f"  n = {len(over):,}")
    print(f"  model log loss     {ll:.5f}")
    print(f"  base rate log loss {B.log_loss(base, over):.5f}  (over rate {over.mean():.1%})")
    print("  (no closing totals odds stored in the backtest, so no market column here)")


if __name__ == "__main__":
    main()
