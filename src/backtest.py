"""Walk-forward backtest against the Pinnacle line.

NOTE ON WHICH LINE. PSH/PSD/PSA from football-data are PRE-CLOSING (opening)
odds; closing carries an extra "C" (PSCH/PSCD/PSCA). This module described
them as "closing" from the start, so every market comparison it has produced
was against the OPENING line. The direction of every conclusion survives —
closing is sharper still, so the model's deficit is larger, not smaller — but
the label was wrong. Both are recorded now; measured 2026-09-04, the gap
between them is +0.00074 RPS.

The only honest test of a football model is: predict matches you have not seen,
using only data available beforehand, then compare to the closing line. Closing
odds at a sharp book are the strongest public forecast available, so they are
the benchmark to beat — not a coin flip, and not the bookmaker's opening price.

Protocol: walk forward one week at a time. At each step refit on everything
strictly before that date, predict the coming week, record the result. No future
information touches any prediction.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import model as M

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-15


def devig_proportional(odds: np.ndarray) -> np.ndarray:
    """Normalise decimal odds to probabilities by removing the overround."""
    inv = 1.0 / odds
    return inv / inv.sum(axis=1, keepdims=True)


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass log loss. probs is (n,3) for H/D/A, outcomes is 0/1/2."""
    p = np.clip(probs[np.arange(len(outcomes)), outcomes], EPS, 1.0)
    return float(-np.log(p).mean())


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def run(
    start: str,
    end: str,
    xi: float,
    reg: float,
    step_days: int = 7,
    verbose: bool = True,
    include_new: bool = True,
    only_countries: list[str] | None = None,
    reg_home: float = 0.0,
    shot_weight: float = 0.0,
) -> pd.DataFrame:
    import data as D
    import model_shots as MS

    df = D.load_history(include_new=include_new)
    df = df[df["FTHG"].notna() & df["FTAG"].notna()].copy()
    groups = D.country_groups(df)
    if only_countries:
        groups = {c: v for c, v in groups.items() if c in only_countries}

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    t0 = time.time()

    for country, divs in groups.items():
        sub = df[df["Div"].isin(divs)].copy()
        if sub.empty:
            continue
        # Predict only matches in the backtest window; fit on all prior history.
        targets = sub[(sub["Date"] >= start_ts) & (sub["Date"] <= end_ts)]
        if targets.empty:
            continue

        n_fits = 0
        cursor = start_ts
        while cursor <= end_ts:
            nxt = cursor + pd.Timedelta(days=step_days)
            week = targets[(targets["Date"] >= cursor) & (targets["Date"] < nxt)]
            if week.empty:
                cursor = nxt
                continue
            try:
                if shot_weight > 0:
                    fr = MS.fit(sub, cursor, xi=xi, reg=reg, reg_home=reg_home,
                                weight=shot_weight)
                else:
                    fr = M.fit(sub, cursor, xi=xi, reg=reg, reg_home=reg_home)
            except ValueError:
                cursor = nxt
                continue
            n_fits += 1
            predict_fn = MS.predict if shot_weight > 0 else M.predict
            for _, m in week.iterrows():
                pred = predict_fn(fr, m["HomeTeam"], m["AwayTeam"], m["Div"])
                if pred is None:
                    continue  # team never seen before (e.g. first promotion)
                rows.append(
                    {
                        "Date": m["Date"],
                        "country": country,
                        "Div": m["Div"],
                        "HomeTeam": m["HomeTeam"],
                        "AwayTeam": m["AwayTeam"],
                        "FTHG": m["FTHG"],
                        "FTAG": m["FTAG"],
                        "FTR": m["FTR"],
                        "m_home": pred["p_home"],
                        "m_draw": pred["p_draw"],
                        "m_away": pred["p_away"],
                        "m_over25": pred["p_over25"],
                        # Recorded so the totals/BTTS markets can be scored on
                        # the same walk-forward run as the winner market. The
                        # model already computes these; nothing consumed them.
                        "m_btts": pred["p_btts"],
                        "lam_h": pred["lambda_home"],
                        "lam_a": pred["lambda_away"],
                        # PSH/PSD/PSA are PRE-CLOSING (opening). football-data
                        # marks closing with an extra "C" — PSCH/PSCD/PSCA.
                        # This file called them "closing" for months and every
                        # market benchmark computed here was really against the
                        # OPENING line, which is the weaker one. Both are now
                        # recorded so the distinction can never be lost again.
                        "PSH": m["PSH"], "PSD": m["PSD"], "PSA": m["PSA"],
                        "PSCH": m.get("PSCH"), "PSCD": m.get("PSCD"),
                        "PSCA": m.get("PSCA"),
                        "AvgH": m["AvgH"], "AvgD": m["AvgD"], "AvgA": m["AvgA"],
                        "MaxH": m["MaxH"], "MaxD": m["MaxD"], "MaxA": m["MaxA"],
                    }
                )
            cursor = nxt
        if verbose:
            print(f"  {country:<12} {n_fits:>4} fits, {len(rows):>6} preds so far "
                  f"({time.time() - t0:.0f}s)")

    return pd.DataFrame(rows)


def evaluate(res: pd.DataFrame, label: str = "") -> dict:
    """Score the model against the market on matches where both are available."""
    d = res.dropna(subset=["PSH", "PSD", "PSA"]).copy()
    if d.empty:
        print("no rows with Pinnacle odds")
        return {}

    outcomes = d["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    keep = ~pd.isna(outcomes)
    d, outcomes = d[keep], outcomes[keep].astype(int)

    model_p = d[["m_home", "m_draw", "m_away"]].to_numpy(float)
    model_p = model_p / model_p.sum(axis=1, keepdims=True)
    market_p = devig_proportional(d[["PSH", "PSD", "PSA"]].to_numpy(float))

    # A naive baseline: the unconditional H/D/A base rate over the sample.
    base = np.bincount(outcomes, minlength=3) / len(outcomes)
    base_p = np.tile(base, (len(outcomes), 1))

    out = {
        "n": len(d),
        "model_logloss": log_loss(model_p, outcomes),
        "market_logloss": log_loss(market_p, outcomes),
        "base_logloss": log_loss(base_p, outcomes),
        "model_brier": brier(model_p, outcomes),
        "market_brier": brier(market_p, outcomes),
        "model_acc": float((model_p.argmax(1) == outcomes).mean()),
        "market_acc": float((market_p.argmax(1) == outcomes).mean()),
    }

    # How much weight does the optimal blend put on the model? If the model
    # carries information the market lacks, the best blend is not w=0.
    best_w, best_ll = 0.0, out["market_logloss"]
    for w in np.arange(0, 1.001, 0.02):
        ll = log_loss(w * model_p + (1 - w) * market_p, outcomes)
        if ll < best_ll:
            best_w, best_ll = float(w), ll
    out["blend_best_w"] = best_w
    out["blend_logloss"] = best_ll

    hdr = f"  RESULTS {label}".rstrip()
    print(f"\n{hdr}\n  {'-' * max(len(hdr) - 2, 40)}")
    print(f"  matches scored              {out['n']:,}")
    print(f"  log loss  model             {out['model_logloss']:.5f}")
    print(f"  log loss  market (Pinnacle) {out['market_logloss']:.5f}")
    print(f"  log loss  base rate         {out['base_logloss']:.5f}")
    gap = out["model_logloss"] - out["market_logloss"]
    print(f"  model - market              {gap:+.5f}"
          f"   ({'model better' if gap < 0 else 'MARKET BETTER'})")
    print(f"  brier     model / market    {out['model_brier']:.5f} / {out['market_brier']:.5f}")
    print(f"  accuracy  model / market    {out['model_acc']:.3%} / {out['market_acc']:.3%}")
    print(f"  best blend weight on model  {out['blend_best_w']:.2f}"
          f"  -> log loss {out['blend_logloss']:.5f}")
    return out


def calibration_table(res: pd.DataFrame, bins: int = 10) -> None:
    """Does a stated 70% actually happen 70% of the time?"""
    d = res.dropna(subset=["PSH"]).copy()
    outcomes = d["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    p = d[["m_home", "m_draw", "m_away"]].to_numpy(float)
    p = p / p.sum(axis=1, keepdims=True)

    flat_p = p.reshape(-1)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(outcomes)), outcomes] = 1
    flat_y = onehot.reshape(-1)

    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(flat_p, edges) - 1, 0, bins - 1)
    print("\n  CALIBRATION (all H/D/A predictions pooled)")
    print(f"  {'bucket':<14}{'n':>8}{'predicted':>12}{'actual':>10}{'diff':>9}")
    for b in range(bins):
        m = idx == b
        if m.sum() < 30:
            continue
        pr, ac = flat_p[m].mean(), flat_y[m].mean()
        print(f"  {edges[b]:.1f}-{edges[b+1]:.1f}      {m.sum():>8,}"
              f"{pr:>11.1%}{ac:>10.1%}{ac - pr:>+9.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-08-01")
    ap.add_argument("--end", default="2026-08-03")
    ap.add_argument("--xi", type=float, default=0.0018)
    ap.add_argument("--reg", type=float, default=2.0)
    ap.add_argument("--out", default="backtest_preds.parquet")
    ap.add_argument("--countries", default=None,
                    help="comma-separated subset, e.g. 'USA,Argentina,Brazil'")
    args = ap.parse_args()

    only = args.countries.split(",") if args.countries else None
    print(f"Backtest {args.start} .. {args.end}  (xi={args.xi}, reg={args.reg})")
    res = run(args.start, args.end, args.xi, args.reg, only_countries=only)
    outp = ROOT / "data" / "processed" / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(outp, index=False)
    print(f"\nsaved {len(res):,} predictions -> {outp}")

    evaluate(res, f"({args.start} .. {args.end})")
    calibration_table(res)


if __name__ == "__main__":
    main()
