"""Does the NFL totals market under-adjust for wind?

The raw pattern is suggestive: in games with 13+ mph wind the total lands about
a point BELOW the closing line, while in calm games it lands over. If real and
large enough, betting unders in wind would beat the number.

Reasons to be suspicious before believing it:

  * Wind is recorded only for outdoor games (72% coverage), so any comparison
    against all games conflates wind with stadium type.
  * The market clearly does move for wind already (line correlation -0.089), so
    the question is only whether it moves ENOUGH.
  * This pattern was found by looking at the data. Betting on it means betting
    that a bookmaker with far more information has left a simple, visible edge
    on the table for a decade.

Tested here properly: an out-of-sample split, a bootstrap CI on the hit rate,
and the 52.4% threshold that -110 juice demands.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def summarise(d: pd.DataFrame, label: str, thresh: float) -> dict | None:
    sel = d[d["wind"] >= thresh]
    live = sel[sel["total_points"] != sel["total_line"]]
    if len(live) < 60:
        return None
    under = (live["total_points"] < live["total_line"]).mean()
    n = len(live)
    se = np.sqrt(under * (1 - under) / n)
    return {"label": label, "thresh": thresh, "n": n, "under_rate": under,
            "lo": under - 1.96 * se, "hi": under + 1.96 * se,
            "roi": under * (100 / 110) - (1 - under)}


def main():
    d = pd.read_parquet(ROOT / "data" / "raw" / "nfl_games.parquet")
    d = d[d["played"] & d["wind"].notna() & d["total_line"].notna()].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    print(f"{len(d):,} outdoor games with wind and a closing total")
    print(f"  {d['date'].min().date()} .. {d['date'].max().date()}\n")

    print(f"  {'wind >=':<10}{'n':>7}{'under%':>9}{'95% CI':>20}{'ROI@-110':>11}")
    rows = []
    for t in (0, 8, 10, 12, 13, 15, 18, 20):
        r = summarise(d, "all", t)
        if r:
            rows.append(r)
            print(f"  {t:<10}{r['n']:>7,}{r['under_rate']:>9.2%}"
                  f"{f'[{r[chr(108)+chr(111)]:.1%}, {r[chr(104)+chr(105)]:.1%}]':>20}"
                  f"{r['roi']:>+11.2%}")
    print(f"\n  break-even at -110 juice is 52.38%")

    # Out-of-sample: pick the threshold on the first half, test on the second.
    k = len(d) // 2
    tr, te = d.iloc[:k], d.iloc[k:]
    print(f"\n  OUT-OF-SAMPLE SPLIT")
    print(f"    fit  {tr['date'].min().date()} .. {tr['date'].max().date()}  n={len(tr):,}")
    print(f"    test {te['date'].min().date()} .. {te['date'].max().date()}  n={len(te):,}")

    best_t, best_rate = None, 0.0
    for t in (8, 10, 12, 13, 15, 18, 20):
        r = summarise(tr, "train", t)
        if r and r["under_rate"] > best_rate:
            best_t, best_rate = t, r["under_rate"]
    if best_t is None:
        print("    no usable threshold in the training half")
        return
    print(f"    best threshold on the fit half: wind >= {best_t} "
          f"({best_rate:.2%} under)")

    r = summarise(te, "test", best_t)
    if not r:
        print("    too few test games at that threshold")
        return
    print(f"    applied to the test half: {r['n']:,} games, "
          f"{r['under_rate']:.2%} under")
    print(f"      95% CI [{r['lo']:.1%}, {r['hi']:.1%}]   "
          f"ROI at -110: {r['roi']:+.2%}")

    rng = np.random.default_rng(0)
    sel = te[(te["wind"] >= best_t) &
             (te["total_points"] != te["total_line"])]
    y = (sel["total_points"] < sel["total_line"]).to_numpy().astype(float)
    boot = np.array([y[rng.integers(0, len(y), len(y))].mean()
                     for _ in range(5000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"      bootstrap [{lo:.1%}, {hi:.1%}]")
    if lo > 0.5238:
        print("      -> beats the juice at 95% confidence")
    elif hi < 0.5238:
        print("      -> significantly BELOW break-even")
    else:
        print("      -> indistinguishable from break-even")

    print(f"\n  residual after the line, by wind bucket (test half):")
    te2 = te.copy()
    te2["resid"] = te2["total_points"] - te2["total_line"]
    te2["bin"] = pd.cut(te2["wind"], [-1, 5, 10, 15, 100])
    print(te2.groupby("bin", observed=True).agg(
        n=("resid", "size"), mean_resid=("resid", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
