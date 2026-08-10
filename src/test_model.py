"""Sanity checks: analytic gradient vs numerical, and a smoke fit."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import check_grad, approx_fprime

sys.path.insert(0, str(Path(__file__).parent))
import model as M

ROOT = Path(__file__).resolve().parents[1]


def test_gradient():
    rng = np.random.default_rng(0)
    n, n_teams, n_div = 500, 12, 2
    hi = rng.integers(0, n_teams, n)
    ai = rng.integers(0, n_teams, n)
    di = rng.integers(0, n_div, n)
    x = rng.poisson(1.4, n).astype(float)
    y = rng.poisson(1.1, n).astype(float)
    w = rng.uniform(0.3, 1.0, n)
    p = np.concatenate([
        rng.normal(0, 0.3, n_teams),
        rng.normal(0, 0.3, n_teams),
        rng.normal(0.25, 0.05, n_div),
        [-0.04],
    ])
    args = (hi, ai, di, x, y, w, n_teams, n_div, 2.0)

    f = lambda q: M._neg_ll(q, *args)[0]
    g = lambda q: M._neg_ll(q, *args)[1]
    err = check_grad(f, g, p, epsilon=1e-7)
    scale = np.linalg.norm(approx_fprime(p, f, 1e-7))
    rel = err / scale
    print(f"gradient check (no per-team home): abs {err:.3e}, relative {rel:.3e}")
    assert rel < 1e-5, "analytic gradient disagrees with numerical"

    # Same check with the per-team home-advantage block enabled.
    p2 = np.concatenate([p, rng.normal(0, 0.1, n_teams)])
    args2 = (hi, ai, di, x, y, w, n_teams, n_div, 2.0, 6.0)
    f2 = lambda q: M._neg_ll(q, *args2)[0]
    g2 = lambda q: M._neg_ll(q, *args2)[1]
    err2 = check_grad(f2, g2, p2, epsilon=1e-7)
    rel2 = err2 / np.linalg.norm(approx_fprime(p2, f2, 1e-7))
    print(f"gradient check (per-team home):    abs {err2:.3e}, relative {rel2:.3e}")
    assert rel2 < 1e-5, "per-team home gradient disagrees with numerical"
    print("  OK")


def test_fit():
    df = pd.read_parquet(ROOT / "data" / "raw" / "football_data_raw.parquet")
    eng = df[df["Div"].isin(M.COUNTRY_GROUPS["England"])].copy()
    as_of = pd.Timestamp("2025-08-01")
    t0 = time.time()
    fr = M.fit(eng, as_of)
    dt = time.time() - t0
    print(f"\nEngland fit @ {as_of.date()}: {fr.n_matches} matches, "
          f"{len(fr.teams)} teams, eff_n={fr.eff_n:.0f}, {dt:.2f}s")
    print(f"  rho = {fr.rho:+.4f}   (Dixon-Coles low-score correction)")
    for d, h in zip(fr.divisions, fr.home_adv):
        print(f"  home advantage {d}: {h:+.4f}")

    order = np.argsort(-(fr.attack - fr.defence))
    print("\n  strongest (attack - defence):")
    for k in order[:8]:
        print(f"    {fr.teams[k]:<18} atk {fr.attack[k]:+.3f}  def {fr.defence[k]:+.3f}")

    p = M.predict(fr, "Liverpool", "Everton", "E0")
    print(f"\n  Liverpool vs Everton: H {p['p_home']:.1%}  D {p['p_draw']:.1%}  "
          f"A {p['p_away']:.1%}  (xG {p['lambda_home']:.2f}-{p['lambda_away']:.2f})")
    tot = p["p_home"] + p["p_draw"] + p["p_away"]
    assert abs(tot - 1) < 1e-6, f"probabilities sum to {tot}"
    print(f"  probabilities sum to {tot:.10f}  OK")


if __name__ == "__main__":
    test_gradient()
    test_fit()
