"""Can the model call goal totals and both-teams-to-score?

WHY THIS EXISTS

Every measured attempt to beat the market on GAME WINNERS has failed: 68k
matches at blend weight 0.00, a live record indistinguishable from the line,
and a simultaneous-price experiment that found Kalshi and the bookmakers
identical to within 0.8c. That question is closed.

This asks a different one. Dixon-Coles does not actually model "who wins" —
it models two goal rates and a low-score correlation, and the winner
probability is a DERIVED quantity obtained by summing one triangle of the
score matrix. Totals and BTTS are read off the same matrix and are closer to
what the model actually estimates. `predict()` has always returned p_over25
and p_btts; nothing has ever consumed them.

Kalshi lists ~450 open TOTAL and BTTS markets across the leagues we already
price, all at 1c spreads. If these probabilities are calibrated, that is a
large increase in pick volume from a model we have already fitted, with no
new data source.

WHAT WOULD MAKE IT REAL, AND WHAT WOULD KILL IT

Calibration is the gate, and it is a harsher test than accuracy. A model can
be accurate on average and still be systematically overconfident in the tails
where the picks would actually be taken. Reported here:

  * log loss and Brier against a base rate that always predicts the mean
  * a calibration table: predicted probability vs realised frequency by decile
  * the same, restricted to the >=60% conviction band the floor actually uses

If the model cannot beat its own base rate, there is nothing here — quoting
"58% of matches go over 2.5" is not a forecast. If it beats the base rate but
the calibration table is skewed at the top end, the floor picks would be
systematically overpriced and it must not ship.

This does NOT establish an edge over the market. Historical over/under odds
are not in our data (fetch_data.KEEP never captured them), so no market
comparison is possible retrospectively. That question is deliberately left to
live paper-tracking against Kalshi, exactly as the winner market was handled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

EPS = 1e-15


def ll(p, y) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def calib_table(p, y, label: str, bins: int = 10) -> None:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    print(f"\n  calibration — {label}")
    print(f"    {'band':<14}{'n':>7}{'predicted':>12}{'actual':>10}{'diff':>9}")
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() < 30:
            continue
        pr, ac = p[m].mean(), y[m].mean()
        flag = "  <-- off" if abs(pr - ac) > 0.05 else ""
        print(f"    {lo:.0%}-{hi:.0%}{'':<7}{m.sum():>7}{pr:>12.1%}{ac:>10.1%}"
              f"{ac - pr:>+9.1%}{flag}")


def score(res: pd.DataFrame, pcol: str, ycol: str, name: str) -> dict:
    d = res[res[pcol].notna() & res[ycol].notna()]
    if len(d) < 100:
        print(f"\n{name}: only {len(d)} rows — skipping")
        return {}
    p, y = d[pcol].to_numpy(float), d[ycol].to_numpy(float)
    base = float(y.mean())
    m_ll, b_ll = ll(p, y), ll(np.full_like(p, base), y)
    print(f"\n{'=' * 66}\n{name}   (n={len(d):,})")
    print(f"  base rate           {base:.1%}  (always predict this)")
    print(f"  model log loss      {m_ll:.4f}")
    print(f"  base  log loss      {b_ll:.4f}")
    print(f"  improvement         {b_ll - m_ll:+.4f}  "
          f"{'MODEL BEATS BASE RATE' if m_ll < b_ll else 'NO SKILL — model is worse than a constant'}")
    print(f"  model Brier         {brier(p, y):.4f}   base {brier(np.full_like(p, base), y):.4f}")
    calib_table(p, y, name)

    # The band that would actually be picked. A floor pick is taken on
    # whichever side clears the threshold, so both tails matter.
    print(f"\n  the >=60% conviction band (what would actually be picked):")
    for side, mask, yy in (("YES", p >= 0.60, y), ("NO", p <= 0.40, 1 - y)):
        if mask.sum() < 30:
            print(f"    {side:<4} too few ({mask.sum()})")
            continue
        conf = p[mask] if side == "YES" else 1 - p[mask]
        hit = yy[mask].mean()
        print(f"    {side:<4} n={mask.sum():>5}  stated {conf.mean():.1%}  "
              f"actual {hit:.1%}  {'OK' if abs(conf.mean() - hit) < 0.04 else '<-- MISCALIBRATED'}")
    return {"n": len(d), "ll": m_ll, "base_ll": b_ll, "edge": b_ll - m_ll}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-08-01")
    ap.add_argument("--end", default="2026-08-01")
    ap.add_argument("--xi", type=float, default=None)
    ap.add_argument("--reg", type=float, default=None)
    ap.add_argument("--cache", default="reports/totals_backtest.parquet")
    ap.add_argument("--refit", action="store_true",
                    help="ignore the cached walk-forward and rerun it")
    args = ap.parse_args()

    cache = ROOT / args.cache
    if cache.exists() and not args.refit:
        res = pd.read_parquet(cache)
        print(f"loaded cached walk-forward: {len(res):,} matches "
              f"({res['Date'].min().date()} .. {res['Date'].max().date()})")
    else:
        import backtest as B
        import json
        xi, reg = args.xi, args.reg
        bp = ROOT / "data" / "processed" / "best_params.json"
        if (xi is None or reg is None) and bp.exists():
            j = json.loads(bp.read_text())
            xi = xi if xi is not None else j.get("xi")
            reg = reg if reg is not None else j.get("reg")
        xi = xi if xi is not None else 0.0018
        reg = reg if reg is not None else 0.02
        print(f"walk-forward {args.start} .. {args.end}  (xi={xi}, reg={reg})")
        res = B.run(args.start, args.end, xi=xi, reg=reg)
        cache.parent.mkdir(parents=True, exist_ok=True)
        res.to_parquet(cache, index=False)
        print(f"cached -> {cache}")

    if res.empty:
        print("no predictions produced")
        return 1

    res = res[res["FTHG"].notna() & res["FTAG"].notna()].copy()
    res["over25"] = ((res["FTHG"] + res["FTAG"]) > 2.5).astype(int)
    res["btts"] = ((res["FTHG"] > 0) & (res["FTAG"] > 0)).astype(int)

    score(res, "m_over25", "over25", "OVER 2.5 GOALS")
    if "m_btts" in res.columns:
        score(res, "m_btts", "btts", "BOTH TEAMS TO SCORE")
    else:
        print("\n(no m_btts column in cache — rerun with --refit to score BTTS)")

    vs_market(res)

    print(f"\n{'=' * 66}")
    print("Beating the base rate is necessary, not sufficient. It shows the")
    print("model carries real information about goals; it says nothing about")
    print("whether that information is already in Kalshi's price. Historical")
    print("over/under odds are absent from our data, so the market comparison")
    print("has to be earned live, the same way the winner market was.")
    return 0




# The board prices majors only (user decision, reaffirmed 2026-08-19), so the
# research must be scoped the same way. Measuring totals across Danish 2nd tier
# would answer a question about leagues we would never pick from, and pooling
# them in would let cheap edge in obscure divisions mask its absence in the
# ones we actually trade.
MAJOR_DIVS = {
    "E0",           # England Premier League
    "SP1",          # Spain La Liga
    "I1",           # Italy Serie A
    "D1",           # Germany Bundesliga
    "F1",           # France Ligue 1
    "USA:MLS",      # MLS
    "MEX:Liga MX",  # Liga MX
}


def only_majors(df: pd.DataFrame) -> pd.DataFrame:
    if "Div" not in df.columns:
        return df
    return df[df["Div"].isin(MAJOR_DIVS)]


def vs_market(res: pd.DataFrame) -> None:
    """The test that actually matters: model vs the closing over/under line.

    Beating a base rate only proves the model knows something about goals.
    The market knows it too. This runs the same comparison that closed the
    winner market at blend weight 0.00, so the two answers are directly
    comparable. The blend weight is the most honest single summary: it asks
    how much of the model a perfectly informed bettor would mix into the
    market price, and returns 0.00 when the model adds nothing.
    """
    hist = pd.read_parquet(ROOT / "data" / "raw" / "football_data_raw.parquet")
    cols = [c for c in ("AvgC>2.5", "AvgC<2.5", "PC>2.5", "PC<2.5")
            if c in hist.columns]
    if len(cols) < 2:
        print("\nno closing over/under odds in the data — cannot compare")
        return
    hist = hist[["Div", "Date", "HomeTeam", "AwayTeam"] + cols].copy()
    hist["Date"] = pd.to_datetime(hist["Date"])
    res = res.copy()
    res["Date"] = pd.to_datetime(res["Date"])
    m = res.merge(hist, on=["Div", "Date", "HomeTeam", "AwayTeam"], how="inner")

    over = under = None
    if "PC>2.5" in m and "PC<2.5" in m:
        over, under = m["PC>2.5"].copy(), m["PC<2.5"].copy()
    if "AvgC>2.5" in m and "AvgC<2.5" in m:
        over = m["AvgC>2.5"] if over is None else over.fillna(m["AvgC>2.5"])
        under = m["AvgC<2.5"] if under is None else under.fillna(m["AvgC<2.5"])
    m["_o"], m["_u"] = over, under
    m = m[m["_o"].notna() & m["_u"].notna() & (m["_o"] > 1) & (m["_u"] > 1)]

    for label, sub in (("MAJORS ONLY", only_majors(m)), ("all divisions", m)):
        if len(sub) < 500:
            print(f"\n{label}: only {len(sub)} rows with odds — too few")
            continue
        _compare(sub, label)


def _compare(m: pd.DataFrame, label: str) -> None:
    # De-vig proportionally, the same treatment the winner market received.
    inv_o, inv_u = 1 / m["_o"], 1 / m["_u"]
    q = (inv_o / (inv_o + inv_u)).to_numpy(float)
    p = m["m_over25"].to_numpy(float)
    y = m["over25"].to_numpy(float)

    print("\n" + "=" * 66)
    print(f"OVER 2.5 — MODEL vs CLOSING LINE  [{label}]   (n={len(m):,})")
    m_ll, q_ll = ll(p, y), ll(q, y)
    print(f"  model log loss      {m_ll:.5f}")
    print(f"  market log loss     {q_ll:.5f}")
    verdict = "model worse" if m_ll > q_ll else "MODEL BETTER"
    print(f"  difference          {m_ll - q_ll:+.5f}   ({verdict})")

    ws = np.linspace(0, 1, 101)
    lls = [ll(w * p + (1 - w) * q, y) for w in ws]
    w_star = float(ws[int(np.argmin(lls))])
    print(f"  optimal blend weight on the model: {w_star:.2f}")
    msg = ("model adds nothing the line does not already carry"
           if w_star < 0.05 else "model carries information beyond the line")
    print(f"    -> {msg}")

    d = (-(y * np.log(np.clip(p, EPS, 1)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1)))
         + (y * np.log(np.clip(q, EPS, 1)) + (1 - y) * np.log(np.clip(1 - q, EPS, 1))))
    rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  95% CI on the gap   [{lo:+.5f}, {hi:+.5f}]")
    concl = ("market significantly sharper" if lo > 0 else
             "MODEL significantly sharper" if hi < 0 else
             "not distinguishable")
    print(f"    -> {concl}")

    print("\n  where the model disagrees with the line:")
    gap = p - q
    for lo_g, hi_g in ((0.03, 0.06), (0.06, 0.10), (0.10, 1.0)):
        sel = (np.abs(gap) >= lo_g) & (np.abs(gap) < hi_g)
        if sel.sum() < 100:
            continue
        mll, qll = ll(p[sel], y[sel]), ll(q[sel], y[sel])
        print(f"    |gap| {lo_g:.0%}-{hi_g:.0%}  n={sel.sum():>6}  "
              f"model {mll:.4f}   market {qll:.4f}   "
              f"{'model' if mll < qll else 'MARKET'} wins")


if __name__ == "__main__":
    raise SystemExit(main())
