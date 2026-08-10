"""What the tracked record can and cannot yet tell us about the models.

Read this before believing any claim that the system is "learning". Right now
there is NO feedback loop from the ledger into any model. Nothing in
model.py, mlb_model.py, margin_model.py or cricket_model.py reads
ledger.jsonl. The models refit on HISTORICAL results when the fetch scripts
re-run — that is the only sense in which they update — and our own tracked
picks influence nothing.

That is the correct design at this sample size, and this script exists to say
when it stops being correct.

Why not tune on our own record yet: with a handful of settled predictions, any
adjustment fitted to them is fitting noise. The standard error on a win rate
after n bets is about 0.5/sqrt(n) — at n=9 that is +/-17 points, so a run of
4-5 is statistically indistinguishable from a 65% model having a bad week or a
35% model having a good one. Reacting to it would be superstition with extra
steps.

What this reports:
  * the gap to the market, with a bootstrap interval, so its width is visible
  * calibration by confidence band — of the calls made at N%, how many landed
  * how many more settled picks are needed before the gap could be called real
  * per-sport breakdown, since a soccer problem should not hide behind baseball

    python src/learning.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "processed" / "ledger.jsonl"
EPS = 1e-15


def ll(p, y) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> None:
    rows = [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    df = pd.DataFrame(rows)
    live = df[df["outcome"].notna()
              & ~df.get("voided", pd.Series(False, index=df.index)).fillna(False)]
    print(f"settled predictions: {len(live)}")
    if len(live) < 2:
        print("too few to say anything at all.")
        return

    y = live["won"].astype(int).to_numpy()
    p = live["model_prob"].to_numpy(float)
    print(f"win rate           : {y.mean():.1%}  ({y.sum()}-{len(y) - y.sum()})")
    print(f"model log loss     : {ll(p, y):.4f}")

    have = live[live["market_prob"].notna()]
    if len(have) >= 2:
        ys = have["won"].astype(int).to_numpy()
        pm = have["model_prob"].to_numpy(float)
        qm = have["market_prob"].to_numpy(float)
        gap = ll(pm, ys) - ll(qm, ys)
        # Per-prediction difference, bootstrapped. The interval is the point:
        # a gap whose interval spans zero is not evidence of anything.
        diff = (-(ys * np.log(np.clip(pm, EPS, 1 - EPS))
                  + (1 - ys) * np.log(np.clip(1 - pm, EPS, 1 - EPS)))
                + (ys * np.log(np.clip(qm, EPS, 1 - EPS))
                   + (1 - ys) * np.log(np.clip(1 - qm, EPS, 1 - EPS))))
        rng = np.random.default_rng(0)
        boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                         for _ in range(5000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"market log loss    : {ll(qm, ys):.4f}   (on the {len(have)} priced)")
        print(f"vs market          : {gap:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
        verdict = ("model is significantly WORSE" if lo > 0 else
                   "model is significantly better" if hi < 0 else
                   "indistinguishable from the market — the interval spans zero")
        print(f"verdict            : {verdict}")

        # How much more data before a gap this size could be called real?
        sd = float(diff.std(ddof=1))
        if sd > 0 and abs(diff.mean()) > 1e-9:
            need = int(np.ceil((1.96 * sd / abs(diff.mean())) ** 2))
            print(f"\nsample size        : {len(have)} priced; a gap THIS LARGE "
                  f"needs ~{need} to detect")
            if need <= len(have):
                print("  The gap is big enough to have cleared the noise floor "
                      "already.")
                print("  That is not good news — it is a large deficit, not a "
                      "small one\n  measured precisely.")
            else:
                print("  Not yet enough. Treat the number above as bookkeeping.")
            # A 2-point edge is the realistic thing to hunt for, and it needs
            # vastly more data than a deficit this size does.
            need2 = int(np.ceil((1.96 * sd / 0.02) ** 2))
            print(f"  Detecting a genuine 2-point EDGE would need ~{need2:,}.")

    print("\ncalibration — of the calls at each confidence, how many landed")
    band = pd.cut(live["model_prob"], [0, .4, .55, .7, .85, 1.0],
                  labels=["<40%", "40-55%", "55-70%", "70-85%", ">85%"])
    for k, g in live.groupby(band, observed=True):
        act = g["won"].astype(int).mean()
        say = g["model_prob"].mean()
        n = len(g)
        se = 0.5 / np.sqrt(n)
        note = "" if abs(act - say) < 2 * se else "   <-- outside 2 SE, but n is tiny"
        print(f"  {str(k):<8} n={n:<3} said {say:.0%}  actual {act:.0%}{note}")

    print("\nby sport")
    # The ledger holds both "mlb" and "baseball" for the same sport, depending
    # on which path wrote the row. Grouping raw splits one record into two.
    live = live.assign(sport=live["sport"].str.lower()
                       .replace({"mlb": "baseball", "wnba": "basketball"}))
    for sp, g in live.groupby("sport"):
        ys = g["won"].astype(int).to_numpy()
        print(f"  {str(sp):<11} n={len(g):<3} {ys.sum()}-{len(ys) - ys.sum()}"
              f"   log loss {ll(g['model_prob'].to_numpy(float), ys):.4f}")

    print("\nWHAT WOULD ACTUALLY CHANGE A MODEL")
    print("  Not this file. Models change when a feature survives an ablation")
    print("  test on THOUSANDS of historical matches with a paired confidence")
    print("  interval — that is how the shots hybrid (-0.00086) and the NFL QB")
    print("  adjustment (-0.00830) got in, and how tuned hyperparameters,")
    print("  altitude and per-team home advantage got rejected. The ledger's")
    print("  job is to catch the case where a model that backtested well is")
    print("  failing live, which needs the sample sizes printed above.")


if __name__ == "__main__":
    main()
