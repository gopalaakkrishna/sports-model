"""Walk-forward backtest for The Hundred.

No market benchmark exists — Kalshi lists 62 cricket series and quotes none of
them — so this can only measure the model against a base rate. That answers
"does it know anything", not "does it beat anyone".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import cricket_model as CM

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-15


def ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main(gender: str = "male"):
    m = pd.read_parquet(ROOT / "data" / "raw" / f"cricket_hundred_{gender}_matches.parquet")
    inn = pd.read_parquet(ROOT / "data" / "raw" / f"cricket_hundred_{gender}_innings.parquet")
    panel = CM.build(m, inn)
    print(f"{gender}: {panel['match_id'].nunique()} matches, {len(panel)} innings")
    print(f"  home innings identified: {panel['is_home'].sum()} / {len(panel)}")

    # One row per match for scoring.
    games = (panel[panel["innings_order"] == 1]
             .rename(columns={"team": "bat_first", "opponent": "bat_second"}))
    games = games.sort_values("date")

    rows = []
    dates = sorted(games["date"].dropna().unique())
    for dt in dates:
        day = games[games["date"] == dt]
        try:
            f = CM.fit(panel, pd.Timestamp(dt))
        except (ValueError, np.linalg.LinAlgError):
            continue
        for _, g in day.iterrows():
            if pd.isna(g["winner"]):
                continue   # no result / tie
            home = None
            hp = panel[(panel["match_id"] == g["match_id"]) & (panel["is_home"] == 1)]
            if len(hp):
                home = hp.iloc[0]["team"]
            p = CM.predict(f, g["bat_first"], g["bat_second"], home=home)
            if p is None:
                continue
            rows.append({
                "date": dt, "a": p["team_a"], "b": p["team_b"],
                "p_a": p["p_a"], "won_a": int(g["winner"] == p["team_a"]),
                "eff_n": p["eff_n_min"], "season": g["season"],
            })

    d = pd.DataFrame(rows)
    if d.empty:
        print("no scored matches")
        return
    y, p = d["won_a"].to_numpy(), d["p_a"].to_numpy()
    base = np.full(len(y), y.mean())
    print(f"\n  scored {len(d)} matches "
          f"({d['date'].min().date()} .. {d['date'].max().date()})")
    print(f"  log loss model     {ll(p, y):.5f}")
    print(f"  log loss base rate {ll(base, y):.5f}")
    print(f"  accuracy           {((p > 0.5) == y).mean():.3%}")
    print(f"  model prob spread  sd {p.std():.4f} "
          f"(range {p.min():.1%}-{p.max():.1%})")

    print(f"\n  BY SEASON")
    for s, g in d.groupby("season"):
        if len(g) < 8:
            continue
        yy, pp = g["won_a"].to_numpy(), g["p_a"].to_numpy()
        print(f"    {s}  n={len(g):>3}  logloss {ll(pp, yy):.4f}  "
              f"acc {((pp > 0.5) == yy).mean():.1%}")

    print(f"\n  CALIBRATION")
    d["bucket"] = pd.cut(d["p_a"], [0, .35, .45, .55, .65, 1.0])
    print(d.groupby("bucket", observed=True).agg(
        n=("won_a", "size"), predicted=("p_a", "mean"),
        actual=("won_a", "mean")).round(3).to_string())

    print(f"\n  {len(d)} matches is a very small sample. A 3-point difference in")
    print("  log loss here is well inside the noise, and there is no market to")
    print("  compare against — Kalshi quotes no cricket at all.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "male")


