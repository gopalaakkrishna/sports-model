"""Does a proposed model feature actually help? Test it, do not assume it.

Runs the same walk-forward backtest with a feature on and off, on identical
matches, and reports the paired difference with a bootstrap confidence interval.
A feature that cannot clear this bar does not go into the model, however
plausible it sounds.

Every "improvement" in this project so far that was adopted on plausibility
alone turned out to be noise: the tuned hyperparameters did not replicate, and
the altitude story behind per-team home advantage ran the wrong way in the data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import backtest as B

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-15


def score(res: pd.DataFrame):
    d = res.dropna(subset=["AvgH", "AvgD", "AvgA"]).copy()
    d["y"] = d["FTR"].map({"H": 0, "D": 1, "A": 2})
    d = d.dropna(subset=["y"])
    d["y"] = d["y"].astype(int)
    d["key"] = (d["Date"].astype(str) + "|" + d["HomeTeam"] + "|" + d["AwayTeam"])
    return d


def probs(d):
    p = d[["m_home", "m_draw", "m_away"]].to_numpy(float)
    return p / p.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-08-01")
    ap.add_argument("--end", default="2026-08-04")
    ap.add_argument("--countries", default="England,Spain,Germany,Italy,France")
    ap.add_argument("--xi", type=float, default=0.0018)
    ap.add_argument("--reg", type=float, default=2.0)
    ap.add_argument("--variants", default="0,4,8",
                    help="values for the feature under test; 0 = feature off")
    ap.add_argument("--feature", default="reg_home",
                    choices=["reg_home", "shot_weight"],
                    help="which feature the variants apply to")
    args = ap.parse_args()

    only = args.countries.split(",")
    runs = {}
    for v in [float(x) for x in args.variants.split(",")]:
        label = "off" if v == 0 else f"{args.feature}={v:g}"
        print(f"\nrunning variant {label} ...")
        kw = {args.feature: v}
        res = B.run(args.start, args.end, args.xi, args.reg,
                    verbose=False, only_countries=only, **kw)
        runs[label] = score(res)
        print(f"  {len(runs[label]):,} scored matches")

    # Restrict every variant to exactly the same matches.
    common = None
    for d in runs.values():
        common = set(d["key"]) if common is None else (common & set(d["key"]))
    print(f"\ncommon matches across variants: {len(common):,}")

    aligned = {}
    for k, d in runs.items():
        dd = d[d["key"].isin(common)].sort_values("key").reset_index(drop=True)
        aligned[k] = dd
    base_key = "off"
    y = aligned[base_key]["y"].to_numpy()
    mkt = B.devig_proportional(
        aligned[base_key][["AvgH", "AvgD", "AvgA"]].to_numpy(float))
    ll_mkt = B.log_loss(mkt, y)

    print(f"\n  {'variant':<16}{'log loss':>10}{'vs market':>12}{'vs baseline':>14}")
    base_ll = None
    per_match = {}
    for k, d in aligned.items():
        p = probs(d)
        v = B.log_loss(p, y)
        li = -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))
        per_match[k] = li
        if k == base_key:
            base_ll = v
        delta = "" if k == base_key else f"{v - base_ll:+.5f}"
        print(f"  {k:<16}{v:>10.5f}{v - ll_mkt:>+12.5f}{delta:>14}")
    print(f"  {'market':<16}{ll_mkt:>10.5f}")

    rng = np.random.default_rng(0)
    for k in aligned:
        if k == base_key:
            continue
        diff = per_match[k] - per_match[base_key]
        boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                         for _ in range(4000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        verdict = ("HELPS" if hi < 0 else "HURTS" if lo > 0 else "no effect")
        print(f"\n  {k} vs baseline: mean {diff.mean():+.5f}, "
              f"95% CI [{lo:+.5f}, {hi:+.5f}]  -> {verdict}")


if __name__ == "__main__":
    main()
