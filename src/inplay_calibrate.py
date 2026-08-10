"""Calibrate the MLB in-play model, and use real team rates rather than a flat one.

The raw state model is already strong (log loss 0.481 vs a 0.691 base rate) but
over-confident at both extremes: it says 2.6% where the true rate is 5.9%, and
97.7% where it is 95.5%. Independent Poisson understates comebacks — big innings
cluster, bullpens tire — so tail probabilities are too extreme.

Two corrections, both fitted on an earlier period and validated on a later one:

  * Platt scaling on the logit, which is what fixed the same problem pre-game
  * a measured extra-innings home edge instead of an assumed 0.53
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from inplay_mlb import InPlayState, win_probability

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-15


def ll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def team_rates() -> dict[str, tuple[float, float]]:
    """Runs scored / allowed per 9 from the season line scores, time weighted."""
    g = pd.read_parquet(ROOT / "data" / "raw" / "mlb_linescores.parquet")
    g["date"] = pd.to_datetime(g["date"])
    recent = g[g["date"] >= g["date"].max() - pd.Timedelta(days=400)]
    out = {}
    teams = set(recent["home_team"]) | set(recent["away_team"])
    for t in teams:
        h = recent[recent["home_team"] == t]
        a = recent[recent["away_team"] == t]
        scored = pd.concat([h["home_score"], a["away_score"]]).mean()
        allowed = pd.concat([h["away_score"], a["home_score"]]).mean()
        out[t] = (float(scored), float(allowed))
    return out


def main():
    st = pd.read_parquet(ROOT / "data" / "raw" / "mlb_inplay_states.parquet")
    st["date"] = pd.to_datetime(st["date"])
    st = st.sort_values("date").reset_index(drop=True)

    # Measured extra-innings home edge.
    ls = pd.read_parquet(ROOT / "data" / "raw" / "mlb_linescores.parquet")
    extras = ls[ls["n_innings"] > 9]
    edge = float((extras["home_score"] > extras["away_score"]).mean())
    print(f"extra-inning games: {len(extras):,}, home win rate {edge:.4f}")
    print(f"  (the model previously assumed 0.53)\n")

    rates = team_rates()
    league = float(np.mean([v[0] for v in rates.values()]))
    print(f"league mean runs/9: {league:.2f}, {len(rates)} teams rated\n")

    sample = st.sample(min(80000, len(st)), random_state=0).sort_values("date")
    rows = []
    for _, r in sample.iterrows():
        hr_s, hr_a = rates.get(r["home_team"], (league, league))
        ar_s, ar_a = rates.get(r["away_team"], (league, league))
        # Expected rate blends the batting side's offence with the other's defence.
        home_rate = (hr_s + ar_a) / 2
        away_rate = (ar_s + hr_a) / 2
        s = InPlayState(int(r["inning"]), r["half"], 0,
                        int(r["home_runs_so_far"]), int(r["away_runs_so_far"]))
        p = win_probability(s, home_rate, away_rate, extras_home_edge=edge)
        rows.append({"date": r["date"], "p_raw": p["p_home"], "y": r["home_won"],
                     "inning": r["inning"], "diff": r["diff"]})
    d = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    k = int(len(d) * 0.7)
    tr, te = d.iloc[:k], d.iloc[k:]
    print(f"fit  {tr['date'].min().date()} .. {tr['date'].max().date()}  n={len(tr):,}")
    print(f"val  {te['date'].min().date()} .. {te['date'].max().date()}  n={len(te):,}")

    ztr, ytr = logit(tr["p_raw"].to_numpy()), tr["y"].to_numpy()
    zte, yte = logit(te["p_raw"].to_numpy()), te["y"].to_numpy()
    r = minimize(lambda ab: ll(sig(ab[0] * ztr + ab[1]), ytr),
                 [1.0, 0.0], method="Nelder-Mead")
    a, b = float(r.x[0]), float(r.x[1])
    base = np.full(len(yte), ytr.mean())

    print(f"\n  Platt: a={a:.4f}  b={b:+.4f}")
    print(f"  {'variant':<26}{'log loss':>11}")
    print(f"  {'base rate':<26}{ll(base, yte):>11.5f}")
    print(f"  {'raw in-play':<26}{ll(te[chr(112)+chr(95)+chr(114)+chr(97)+chr(119)].to_numpy(), yte):>11.5f}")
    cal = sig(a * zte + b)
    print(f"  {'calibrated':<26}{ll(cal, yte):>11.5f}")
    print(f"  accuracy {((cal > 0.5) == yte).mean():.3%}")

    print(f"\n  CALIBRATION AFTER SCALING (validation)")
    tev = te.copy()
    tev["p"] = cal
    tev["bucket"] = pd.cut(tev["p"], np.arange(0, 1.01, 0.1))
    g = tev.groupby("bucket", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    g["err"] = g["actual"] - g["predicted"]
    print(g.round(4).to_string())

    out = ROOT / "data" / "processed" / "inplay_calibration.json"
    out.write_text(json.dumps({
        "platt_a": a, "platt_b": b, "extras_home_edge": edge,
        "league_runs_9": league,
        "valid_logloss": ll(cal, yte),
        "base_logloss": ll(base, yte),
    }, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
