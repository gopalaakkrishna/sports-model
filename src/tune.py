"""Grid-search the time-decay and regularisation hyperparameters.

Tuning happens on an EARLY window only. The recent seasons are never touched
here, so the final backtest over them stays genuinely out-of-sample. Tuning and
reporting on the same period is the most common way a sports model ends up
looking better than it is.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B

ROOT = Path(__file__).resolve().parents[1]

# Half-life in days for reference: ln(2)/xi
XI_GRID = [0.0008, 0.0015, 0.0025, 0.0040]
REG_GRID = [1.0, 2.0, 5.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-08-01")
    ap.add_argument("--end", default="2022-06-30", help="tuning window end (exclusive of holdout)")
    args = ap.parse_args()

    print(f"Tuning on {args.start} .. {args.end}")
    print(f"(seasons after {args.end} are held out for final evaluation)\n")

    results = []
    for xi, reg in itertools.product(XI_GRID, REG_GRID):
        t0 = time.time()
        res = B.run(args.start, args.end, xi, reg, verbose=False)
        d = res.dropna(subset=["PSH", "PSD", "PSA"]).copy()
        y = d["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        keep = ~pd.isna(y)
        d, y = d[keep], y[keep].astype(int)
        mp = d[["m_home", "m_draw", "m_away"]].to_numpy(float)
        mp = mp / mp.sum(axis=1, keepdims=True)
        kp = B.devig_proportional(d[["PSH", "PSD", "PSA"]].to_numpy(float))
        ll = B.log_loss(mp, y)
        mk = B.log_loss(kp, y)
        results.append({
            "xi": xi, "half_life_days": round(np.log(2) / xi),
            "reg": reg, "n": len(d),
            "logloss": ll, "market_logloss": mk, "gap": ll - mk,
        })
        print(f"  xi={xi:.4f} (hl {np.log(2)/xi:>4.0f}d) reg={reg:<4} "
              f"logloss {ll:.5f}  vs market {ll - mk:+.5f}  [{time.time() - t0:.0f}s]")

    df = pd.DataFrame(results).sort_values("logloss")
    outp = ROOT / "reports" / "tuning_results.csv"
    outp.parent.mkdir(exist_ok=True)
    df.to_csv(outp, index=False)

    best = df.iloc[0]
    print(f"\nBest: xi={best['xi']} (half-life {best['half_life_days']:.0f} days), "
          f"reg={best['reg']}, log loss {best['logloss']:.5f}")
    (ROOT / "data" / "processed" / "best_params.json").write_text(
        json.dumps({"xi": float(best["xi"]), "reg": float(best["reg"])}, indent=2)
    )
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
