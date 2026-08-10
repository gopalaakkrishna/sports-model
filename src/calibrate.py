"""Fit and validate calibration on top of the raw model output.

Raw Dixon-Coles probabilities are typically over-confident: a stated 75% does
not happen 75% of the time. Two corrections are fitted here, both on a strict
time split (earlier data fits, later data validates) so the reported numbers are
out-of-sample:

  temperature   p -> p**t, renormalised. t < 1 softens, t > 1 sharpens.
  market blend  w * model + (1 - w) * market

The blend weight is the honest measure of whether this model knows anything the
closing line does not. w = 0 means it does not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-15


def _norm(p):
    return p / p.sum(axis=1, keepdims=True)


def _ll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0)).mean())


def _devig(odds):
    inv = 1.0 / odds
    return inv / inv.sum(axis=1, keepdims=True)


def load(path: Path):
    d = pd.read_parquet(path)
    d = d.dropna(subset=["PSH", "PSD", "PSA"]).copy()
    d["y"] = d["FTR"].map({"H": 0, "D": 1, "A": 2})
    d = d.dropna(subset=["y"]).sort_values("Date").reset_index(drop=True)
    return d


def split(d, frac=0.7):
    k = int(len(d) * frac)
    return d.iloc[:k], d.iloc[k:]


def fit_temperature(model_p, y):
    def obj(t):
        return _ll(_norm(np.clip(model_p, EPS, 1) ** t), y)
    r = minimize_scalar(obj, bounds=(0.2, 2.5), method="bounded")
    return float(r.x)


def fit_blend(model_p, market_p, y):
    def obj(w):
        return _ll(w * model_p + (1 - w) * market_p, y)
    r = minimize_scalar(obj, bounds=(0.0, 1.0), method="bounded")
    return float(r.x)


def main(preds="backtest_preds.parquet"):
    path = ROOT / "data" / "processed" / preds
    d = load(path)
    tr, te = split(d)
    print(f"loaded {len(d):,} scored predictions")
    print(f"  fit   {tr['Date'].min().date()} .. {tr['Date'].max().date()}  n={len(tr):,}")
    print(f"  valid {te['Date'].min().date()} .. {te['Date'].max().date()}  n={len(te):,}")

    def cols(x):
        mp = _norm(x[["m_home", "m_draw", "m_away"]].to_numpy(float))
        kp = _devig(x[["PSH", "PSD", "PSA"]].to_numpy(float))
        return mp, kp, x["y"].to_numpy(int)

    tr_m, tr_k, tr_y = cols(tr)
    te_m, te_k, te_y = cols(te)

    t = fit_temperature(tr_m, tr_y)
    tr_mt = _norm(np.clip(tr_m, EPS, 1) ** t)
    te_mt = _norm(np.clip(te_m, EPS, 1) ** t)
    w = fit_blend(tr_mt, tr_k, tr_y)

    rows = [
        ("raw model",            _ll(te_m, te_y)),
        (f"model, temp t={t:.3f}", _ll(te_mt, te_y)),
        ("market (Pinnacle)",    _ll(te_k, te_y)),
        (f"blend w={w:.3f}",     _ll(w * te_mt + (1 - w) * te_k, te_y)),
    ]
    print("\n  OUT-OF-SAMPLE LOG LOSS (validation period)")
    print(f"  {'variant':<26}{'log loss':>10}{'vs market':>12}")
    mk = _ll(te_k, te_y)
    for name, v in rows:
        print(f"  {name:<26}{v:>10.5f}{v - mk:>+12.5f}")

    params = {
        "temperature": t,
        "blend_weight_on_model": w,
        "fit_rows": len(tr),
        "valid_rows": len(te),
        "valid_logloss": {n: v for n, v in rows},
        "market_logloss": mk,
        "fit_period": [str(tr["Date"].min().date()), str(tr["Date"].max().date())],
        "valid_period": [str(te["Date"].min().date()), str(te["Date"].max().date())],
    }
    outp = ROOT / "data" / "processed" / "calibration.json"
    outp.write_text(json.dumps(params, indent=2))
    print(f"\nsaved -> {outp}")

    if w < 0.05:
        print("\n  VERDICT: the model adds essentially nothing to the closing line.")
    elif w < 0.3:
        print(f"\n  VERDICT: model carries some independent signal (weight {w:.2f}),")
        print("  but the market dominates. Not a betting edge on its own.")
    else:
        print(f"\n  VERDICT: model carries substantial independent signal (weight {w:.2f}).")
    return params


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="backtest_preds.parquet")
    main(ap.parse_args().preds)
