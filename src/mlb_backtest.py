"""Walk-forward backtest for the MLB model.

Refits every `step_days` on data strictly prior, predicts the games in the
window, and scores against the base rate. Also surfaces the two structural
issues a run-based baseball model is prone to:

* **The home team often does not bat in the 9th.** When leading after the top of
  the ninth the game ends, so home teams score fewer runs than their true rate.
  A model fitted on runs therefore understates home advantage in WIN terms.
* **Park effects.** Coors Field inflates runs for both sides. With no park term,
  Colorado's offence looks strong and its defence terrible, and every game
  played there is mispriced.

Both are checked explicitly rather than assumed away.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import mlb_model as MM

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-15


def log_loss_binary(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def run(start: str, end: str, xi: float, reg_team: float, reg_sp: float,
        step_days: int = 7, verbose: bool = True) -> pd.DataFrame:
    g = pd.read_parquet(ROOT / "data" / "raw" / "mlb_games.parquet")
    g = g[g["final"] & g["home_runs"].notna() & g["away_runs"].notna()].copy()

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows, t0, cursor, n_fits = [], time.time(), start_ts, 0

    while cursor <= end_ts:
        nxt = cursor + pd.Timedelta(days=step_days)
        wk = g[(g["date"] >= cursor) & (g["date"] < nxt)]
        if wk.empty:
            cursor = nxt
            continue
        try:
            f = MM.fit(g, cursor, xi=xi, reg_team=reg_team, reg_sp=reg_sp)
        except ValueError:
            cursor = nxt
            continue
        n_fits += 1
        for _, m in wk.iterrows():
            p = MM.predict(f, m["home_team"], m["away_team"],
                           m["home_sp"], m["away_sp"], m["venue"])
            if p is None:
                continue
            rows.append({
                "date": m["date"], "home": m["home_team"], "away": m["away_team"],
                "venue": m["venue"],
                "home_runs": m["home_runs"], "away_runs": m["away_runs"],
                "home_win": int(m["home_runs"] > m["away_runs"]),
                "p_home": p["p_home"],
                "lam_h": p["lambda_home"], "lam_a": p["lambda_away"],
                "exp_total": p["exp_total"],
                # Total-runs probabilities off the same run matrix. The model
                # has always produced these; nothing consumed them, so the
                # totals market could not be scored on the same walk-forward
                # as the winner market.
                "p_over_8_5": p["p_over_8_5"],
                "p_over_9_5": p["p_over_9_5"],
                "actual_total": m["home_runs"] + m["away_runs"],
                "eff_n_sp_home": p["eff_n_sp_home"],
                "eff_n_sp_away": p["eff_n_sp_away"],
            })
        if verbose and n_fits % 25 == 0:
            print(f"  {cursor.date()}: {n_fits} fits, {len(rows):,} preds "
                  f"({time.time() - t0:.0f}s)")
        cursor = nxt

    return pd.DataFrame(rows)


def evaluate(d: pd.DataFrame) -> None:
    y = d["home_win"].to_numpy()
    p = d["p_home"].to_numpy()
    base = np.full(len(y), y.mean())

    print(f"\n  RESULTS\n  {'-' * 46}")
    print(f"  games scored            {len(d):,}")
    print(f"  actual home win rate    {y.mean():.3%}")
    print(f"  mean predicted          {p.mean():.3%}")
    print(f"  log loss  model         {log_loss_binary(p, y):.5f}")
    print(f"  log loss  base rate     {log_loss_binary(base, y):.5f}")
    print(f"  brier     model         {float(((p - y) ** 2).mean()):.5f}")
    print(f"  accuracy                {float(((p > 0.5) == y).mean()):.3%}")

    bias = p.mean() - y.mean()
    print(f"\n  HOME-ADVANTAGE CHECK")
    print(f"    predicted minus actual home win rate: {bias:+.3%}")
    if abs(bias) > 0.005:
        print("    !!! systematic bias. A run-based model understates home")
        print("        advantage because the home team skips the bottom of the")
        print("        9th when already ahead.")

    print(f"\n  CALIBRATION")
    edges = np.array([0, .35, .42, .48, .52, .58, .65, 1.0])
    idx = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    print(f"    {'bucket':<14}{'n':>7}{'predicted':>12}{'actual':>10}{'diff':>9}")
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 50:
            continue
        print(f"    {f'{edges[b]:.2f}-{edges[b+1]:.2f}':<14}{m.sum():>7,}"
              f"{p[m].mean():>11.1%}{y[m].mean():>10.1%}{y[m].mean() - p[m].mean():>+9.1%}")

    print(f"\n  RUN TOTALS")
    print(f"    mean predicted total {d['exp_total'].mean():.2f}")
    print(f"    mean actual total    {d['actual_total'].mean():.2f}")
    print(f"    correlation          {d['exp_total'].corr(d['actual_total']):.3f}")

    print(f"\n  PARK EFFECT CHECK (largest total-runs errors by venue, n>=100)")
    v = d.groupby("venue").agg(n=("actual_total", "size"),
                               pred=("exp_total", "mean"),
                               act=("actual_total", "mean"))
    v = v[v["n"] >= 100]
    v["err"] = v["act"] - v["pred"]
    for name, r in v.reindex(v["err"].abs().sort_values(ascending=False).index).head(6).iterrows():
        print(f"    {str(name)[:34]:<36}{r['n']:>6,.0f}  pred {r['pred']:.2f} "
              f"act {r['act']:.2f}  err {r['err']:+.2f}")
    worst = v["err"].abs().max()
    if worst > 0.5:
        print("    ! park effects are material and unmodelled.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-04-01")
    ap.add_argument("--end", default="2026-08-04")
    ap.add_argument("--xi", type=float, default=0.0025)
    ap.add_argument("--reg-team", type=float, default=2.0, dest="reg_team")
    ap.add_argument("--reg-sp", type=float, default=8.0, dest="reg_sp")
    ap.add_argument("--out", default="mlb_backtest.parquet")
    args = ap.parse_args()

    print(f"MLB backtest {args.start} .. {args.end} "
          f"(xi={args.xi}, reg_team={args.reg_team}, reg_sp={args.reg_sp})")
    d = run(args.start, args.end, args.xi, args.reg_team, args.reg_sp)
    outp = ROOT / "data" / "processed" / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(outp, index=False)
    print(f"\nsaved {len(d):,} predictions -> {outp}")
    evaluate(d)
    fit_calibration(d)


def fit_calibration(d: pd.DataFrame, frac: float = 0.7) -> None:
    """Fit Platt scaling on early data, report on held-out later data."""
    import json
    from scipy.optimize import minimize as _min

    d = d.sort_values("date").reset_index(drop=True)
    k = int(len(d) * frac)
    tr, te = d.iloc[:k], d.iloc[k:]

    def logit(p):
        p = np.clip(p, EPS, 1 - EPS)
        return np.log(p / (1 - p))

    def sig(z):
        return 1 / (1 + np.exp(-z))

    ztr, ytr = logit(tr["p_home"].to_numpy()), tr["home_win"].to_numpy()
    zte, yte = logit(te["p_home"].to_numpy()), te["home_win"].to_numpy()
    r = _min(lambda ab: log_loss_binary(sig(ab[0] * ztr + ab[1]), ytr),
             [1.0, 0.0], method="Nelder-Mead")
    a, b = float(r.x[0]), float(r.x[1])
    base = np.full(len(yte), ytr.mean())

    print(f"\n  CALIBRATION (fit on first {frac:.0%}, validated on the rest)")
    print(f"    shrinkage a={a:.3f}  intercept b={b:+.3f}")
    print(f"    {'variant':<24}{'log loss':>10}")
    print(f"    {'raw model':<24}{log_loss_binary(te['p_home'].to_numpy(), yte):>10.5f}")
    print(f"    {'base rate':<24}{log_loss_binary(base, yte):>10.5f}")
    cal = log_loss_binary(sig(a * zte + b), yte)
    print(f"    {'calibrated':<24}{cal:>10.5f}")
    print(f"    calibrated beats base rate by "
          f"{log_loss_binary(base, yte) - cal:+.5f}")

    outp = ROOT / "data" / "processed" / "mlb_calibration.json"
    outp.write_text(json.dumps({"a": a, "b": b,
                                "fit_rows": len(tr), "valid_rows": len(te),
                                "valid_logloss": cal,
                                "base_logloss": log_loss_binary(base, yte)}, indent=2))
    print(f"    saved -> {outp}")


if __name__ == "__main__":
    main()
