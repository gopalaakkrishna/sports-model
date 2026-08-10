"""Gradient check and smoke test for the MLB model."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import check_grad, approx_fprime

sys.path.insert(0, str(Path(__file__).parent))
import mlb_model as MM

ROOT = Path(__file__).resolve().parents[1]


def test_gradient():
    rng = np.random.default_rng(0)
    n, n_t, n_p = 400, 10, 25
    hi = rng.integers(0, n_t, n); ai = rng.integers(0, n_t, n)
    sph = rng.integers(0, n_p, n); spa = rng.integers(0, n_p, n)
    x = rng.poisson(4.5, n).astype(float)
    y = rng.poisson(4.2, n).astype(float)
    w = rng.uniform(0.3, 1.0, n)
    p = np.concatenate([rng.normal(0.75, .1, n_t), rng.normal(0.75, .1, n_t),
                        rng.normal(0, .1, n_p), [0.03]])
    args = (hi, ai, sph, spa, x, y, w, n_t, n_p, 2.0, 8.0)
    f = lambda q: MM._neg_ll(q, *args)[0]
    g = lambda q: MM._neg_ll(q, *args)[1]
    err = check_grad(f, g, p, epsilon=1e-7)
    rel = err / np.linalg.norm(approx_fprime(p, f, 1e-7))
    print(f"gradient check: abs {err:.3e}, relative {rel:.3e}")
    assert rel < 1e-5, "analytic gradient disagrees with numerical"
    print("  OK")


def test_fit():
    g = pd.read_parquet(ROOT / "data" / "raw" / "mlb_games.parquet")
    as_of = pd.Timestamp("2026-08-04")
    t0 = time.time()
    f = MM.fit(g, as_of)
    print(f"\nfit @ {as_of.date()}: {f.n_games:,} games, {len(f.teams)} teams, "
          f"{len(f.pitchers)} pitchers, eff_n={f.eff_n:.0f}, {time.time()-t0:.1f}s")
    print(f"  home advantage {f.home_adv:+.4f} "
          f"(x{np.exp(f.home_adv):.3f} on home runs)")

    net = f.off - f.dfn
    order = np.argsort(-net)
    print("\n  strongest teams (offense - defense):")
    for k in order[:6]:
        print(f"    {f.teams[k]:<24} off {f.off[k]:+.3f}  def {f.dfn[k]:+.3f}")
    print("  weakest:")
    for k in order[-4:]:
        print(f"    {f.teams[k]:<24} off {f.off[k]:+.3f}  def {f.dfn[k]:+.3f}")

    # Pitchers: most negative sp = best run suppression. Require real workload.
    busy = f.pitcher_eff_n >= 20
    idx = np.where(busy)[0]
    best = idx[np.argsort(f.sp[idx])][:8]
    print("\n  best starters (sp < 0 suppresses runs, eff_n >= 20):")
    for k in best:
        print(f"    {f.pitchers[k]:<26} sp {f.sp[k]:+.3f}  eff_n {f.pitcher_eff_n[k]:.0f}")

    p = MM.predict(f, "Los Angeles Dodgers", "Colorado Rockies",
                   home_sp=None, away_sp=None)
    tot = p["p_home"] + p["p_away"]
    print(f"\n  Dodgers vs Rockies (no starters): home {p['p_home']:.1%} "
          f"away {p['p_away']:.1%}  runs {p['lambda_home']:.2f}-{p['lambda_away']:.2f}")
    print(f"  probabilities sum to {tot:.10f}")
    assert abs(tot - 1) < 1e-9, f"probabilities sum to {tot}"

    # League-average runs per team per game should land near reality (~4.4).
    hist = g[g["final"] & (g["date"] >= "2025-01-01")]
    actual = (hist["home_runs"].mean() + hist["away_runs"].mean()) / 2
    print(f"  league actual runs/team/game 2025+: {actual:.2f}")
    print(f"  model implied for an average matchup: "
          f"{(p['lambda_home'] + p['lambda_away']) / 2:.2f} (Dodgers/Rockies, not average)")
    print("  OK")


if __name__ == "__main__":
    test_gradient()
    test_fit()
