"""Walk-forward backtest for the NFL and WNBA margin model.

The NFL data ships with closing spreads, totals and moneylines, so for once the
market benchmark needs no scraping — and all three markets can be scored:

    moneyline  model win probability vs the de-vigged closing moneyline
    spread     did the model pick the right side of the closing spread?
    total      did it pick the right side of the closing total?

Spread and total are the honest tests. Beating a closing spread requires
predicting the margin better than the market did, which is a far higher bar than
merely picking winners.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import margin_model as MM

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-15


def ll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def american_to_prob(m):
    m = np.asarray(m, float)
    return np.where(m < 0, -m / (-m + 100.0), 100.0 / (m + 100.0))


def run(sport: str, start: str, end: str, xi: float, reg: float,
        step_days: int = 7) -> pd.DataFrame:
    path = ROOT / "data" / "raw" / f"{sport}_games.parquet"
    g = pd.read_parquet(path)
    g = g[g["played"]].copy()
    g["date"] = pd.to_datetime(g["date"])

    s, e = pd.Timestamp(start), pd.Timestamp(end)
    rows, cursor, fits = [], s, 0
    while cursor <= e:
        nxt = cursor + pd.Timedelta(days=step_days)
        wk = g[(g["date"] >= cursor) & (g["date"] < nxt)]
        if wk.empty:
            cursor = nxt
            continue
        try:
            f = MM.fit(g, cursor, xi=xi, reg=reg)
        except ValueError:
            cursor = nxt
            continue
        fits += 1
        for _, m in wk.iterrows():
            sp = m.get("spread_line")
            tl = m.get("total_line")
            p = MM.predict(f, m["home_team"], m["away_team"],
                           spread=float(sp) if pd.notna(sp) else None,
                           total_line=float(tl) if pd.notna(tl) else None)
            if p is None:
                continue
            rows.append({
                "date": m["date"], "home": m["home_team"], "away": m["away_team"],
                "home_score": m["home_score"], "away_score": m["away_score"],
                "margin": m["margin"], "total_points": m["total_points"],
                "home_win": int(m["margin"] > 0),
                "p_home": p["p_home"], "exp_margin": p["exp_margin"],
                "exp_total": p["exp_total"],
                "p_home_covers": p.get("p_home_covers"),
                "p_over": p.get("p_over"),
                "spread_line": sp, "total_line": tl,
                "home_ml": m.get("home_moneyline"), "away_ml": m.get("away_moneyline"),
                "eff_n_min": p["eff_n_min"],
            })
        cursor = nxt
    print(f"  {fits} fits, {len(rows):,} predictions")
    return pd.DataFrame(rows)


def evaluate(d: pd.DataFrame, sport: str) -> None:
    print(f"\n{'=' * 70}\n{sport.upper()} — {len(d):,} games "
          f"({d['date'].min().date()} .. {d['date'].max().date()})\n{'=' * 70}")
    y = d["home_win"].to_numpy()
    p = d["p_home"].to_numpy()
    base = np.full(len(y), y.mean())
    print(f"  MONEYLINE")
    print(f"    home win rate      {y.mean():.3%}   model mean {p.mean():.3%}")
    print(f"    log loss model     {ll(p, y):.5f}")
    print(f"    log loss base rate {ll(base, y):.5f}")
    print(f"    accuracy           {((p > 0.5) == y).mean():.3%}")

    mk = d.dropna(subset=["home_ml", "away_ml"])
    if len(mk) > 100:
        hp = american_to_prob(mk["home_ml"].to_numpy())
        ap = american_to_prob(mk["away_ml"].to_numpy())
        mkt = hp / (hp + ap)
        ym = mk["home_win"].to_numpy()
        pm = mk["p_home"].to_numpy()
        print(f"    -- vs closing moneyline ({len(mk):,} games) --")
        print(f"    model  {ll(pm, ym):.5f}")
        print(f"    market {ll(mkt, ym):.5f}")
        gap = ll(pm, ym) - ll(mkt, ym)
        print(f"    gap    {gap:+.5f}  "
              f"({'model better' if gap < 0 else 'MARKET BETTER'})")

    sp = d.dropna(subset=["spread_line", "p_home_covers"])
    if len(sp) > 100:
        # POSITIVE spread_line = home favoured, so home covers when the margin
        # exceeds it. Verified empirically: this rule yields ~47.6% home covers,
        # the inverted one 58%.
        covered = (sp["margin"] - sp["spread_line"] > 0).astype(int)
        push = (sp["margin"] - sp["spread_line"] == 0)
        live = ~push
        pc = sp.loc[live, "p_home_covers"].to_numpy()
        yc = covered[live].to_numpy()
        picks = (pc > 0.5).astype(int)
        print(f"\n  SPREAD ({live.sum():,} non-push)")
        print(f"    model log loss {ll(pc, yc):.5f}   (0.69315 = coin flip)")
        print(f"    pick accuracy  {(picks == yc).mean():.3%}   "
              f"(52.4% needed to beat -110 juice)")
        strong = np.abs(pc - 0.5) > 0.05
        if strong.sum() > 50:
            print(f"    when model is confident (>55/45): "
                  f"{(picks[strong] == yc[strong]).mean():.3%} on {strong.sum():,}")

    to = d.dropna(subset=["total_line", "p_over"])
    if len(to) > 100:
        over = (to["total_points"] > to["total_line"]).astype(int)
        push = (to["total_points"] == to["total_line"])
        live = ~push
        po = to.loc[live, "p_over"].to_numpy()
        yo = over[live].to_numpy()
        picks = (po > 0.5).astype(int)
        print(f"\n  TOTAL ({live.sum():,} non-push)")
        print(f"    model log loss {ll(po, yo):.5f}")
        print(f"    pick accuracy  {(picks == yo).mean():.3%}")
        print(f"    mean predicted total {to['exp_total'].mean():.1f} "
              f"vs line {to['total_line'].mean():.1f} "
              f"vs actual {to['total_points'].mean():.1f}")

    print(f"\n  MARGIN ACCURACY")
    print(f"    MAE  model {np.abs(d['margin'] - d['exp_margin']).mean():.2f}")
    if d["spread_line"].notna().sum() > 100:
        s = d.dropna(subset=["spread_line"])
        print(f"    MAE  market spread {np.abs(s['margin'] - s['spread_line']).mean():.2f}")
    print(f"    correlation model vs actual margin "
          f"{d['exp_margin'].corr(d['margin']):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default="nfl", choices=["nfl", "wnba"])
    ap.add_argument("--start", default="2015-09-01")
    ap.add_argument("--end", default="2026-08-01")
    ap.add_argument("--xi", type=float, default=0.0025)
    ap.add_argument("--reg", type=float, default=8.0)
    args = ap.parse_args()

    print(f"{args.sport} backtest {args.start} .. {args.end} "
          f"(xi={args.xi}, reg={args.reg})")
    d = run(args.sport, args.start, args.end, args.xi, args.reg)
    if d.empty:
        print("no predictions")
        return
    out = ROOT / "data" / "processed" / f"{args.sport}_backtest.parquet"
    d.to_parquet(out, index=False)
    evaluate(d, args.sport)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
