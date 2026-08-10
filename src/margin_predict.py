"""Price upcoming NFL and WNBA games, and flag the wind signal for forward testing.

NFL fixtures come from nflverse (the same file as the history — future games are
present with lines but no score). WNBA fixtures come from the Basketball
Reference schedule.

The wind flag exists to be tested, not traded. Wind >= 12 mph unders hit 56-58%
historically against a 52.4% break-even, but the threshold was suggested by the
data and the effect is unstable at 15 mph. Only forward results on games nobody
has seen can settle it, so those games are logged as predictions and scored like
everything else.

One caveat that cannot be engineered away: nflverse records the wind that
ACTUALLY occurred. At bet time you have a forecast. The backtest is therefore
mildly optimistic about what is knowable in advance.
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
WIND_THRESHOLD = 12.0


def upcoming(sport: str, days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = pd.read_parquet(ROOT / "data" / "raw" / f"{sport}_games.parquet")
    g["date"] = pd.to_datetime(g["date"])
    today = pd.Timestamp.now().normalize()
    fut = g[(~g["played"]) & (g["date"] >= today) &
            (g["date"] <= today + pd.Timedelta(days=days))]
    return g[g["played"]], fut.sort_values("date")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default="nfl", choices=["nfl", "wnba"])
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--min-eff-n", type=float, default=12.0)
    args = ap.parse_args()

    hist, fut = upcoming(args.sport, args.days)
    if fut.empty:
        nxt = pd.read_parquet(ROOT / "data" / "raw" / f"{args.sport}_games.parquet")
        nxt["date"] = pd.to_datetime(nxt["date"])
        pending = nxt[~nxt["played"]]
        if pending.empty:
            print(f"{args.sport}: no scheduled games at all")
        else:
            print(f"{args.sport}: no games in the next {args.days} days. "
                  f"Next is {pending['date'].min().date()}.")
        return

    today = pd.Timestamp.now().normalize()
    f = MM.fit(hist, today)
    print(f"{args.sport.upper()}: fitted {f.n_games:,} games, {len(f.teams)} teams")
    print(f"  league level {f.intercept:.2f} pts/team, "
          f"home advantage {f.home_adv:+.2f}")
    print(f"  sigma margin {f.sigma_margin:.2f}, sigma total {f.sigma_total:.2f}")
    if f.qb is not None:
        busy = f.qb_eff_n >= 8
        if busy.sum() > 4:
            idx = np.where(busy)[0]
            top = idx[np.argsort(-f.qb[idx])][:4]
            print("  best-rated starting QBs: " +
                  ", ".join(f"{f.qbs[k]} ({f.qb[k]:+.1f})" for k in top))

    rows = []
    for _, m in fut.iterrows():
        sp = m.get("spread_line")
        tl = m.get("total_line")
        p = MM.predict(f, m["home_team"], m["away_team"],
                       spread=float(sp) if pd.notna(sp) else None,
                       total_line=float(tl) if pd.notna(tl) else None,
                       home_qb=m.get("home_qb_name"),
                       away_qb=m.get("away_qb_name"))
        if p is None:
            continue
        wind = m.get("wind")
        rows.append({
            "date": m["date"].date(), "away": m["away_team"], "home": m["home_team"],
            "exp_margin": p["exp_margin"], "exp_total": p["exp_total"],
            "p_home": p["p_home"],
            "spread_line": sp, "total_line": tl,
            "p_home_covers": p.get("p_home_covers"), "p_over": p.get("p_over"),
            "wind": wind, "roof": m.get("roof"),
            "eff_n_min": p["eff_n_min"],
            "thin": p["eff_n_min"] < args.min_eff_n,
        })

    if not rows:
        print("  nothing predictable")
        return
    d = pd.DataFrame(rows)
    out = ROOT / "reports" / f"{args.sport}_upcoming_{today.date()}.csv"
    d.to_csv(out, index=False)

    print(f"\n{'date':<12}{'match':<34}{'p(home)':>9}{'margin':>9}"
          f"{'total':>8}{'line':>7}{'wind':>6}")
    for _, r in d.iterrows():
        match = f"{r['away']} @ {r['home']}"
        line = f"{r['total_line']:.1f}" if pd.notna(r["total_line"]) else "-"
        wind = f"{r['wind']:.0f}" if pd.notna(r["wind"]) else "-"
        print(f"{str(r['date']):<12}{match[:33]:<34}{r['p_home']:>9.1%}"
              f"{r['exp_margin']:>+9.1f}{r['exp_total']:>8.1f}{line:>7}{wind:>6}")

    if args.sport == "nfl":
        windy = d[d["wind"].notna() & (d["wind"] >= WIND_THRESHOLD)]
        print(f"\n  WIND WATCH (>= {WIND_THRESHOLD:.0f} mph) — forward test, not a pick")
        if windy.empty:
            print("    no games flagged")
        else:
            for _, r in windy.iterrows():
                print(f"    {r['away']} @ {r['home']}  wind {r['wind']:.0f}mph  "
                      f"total line {r['total_line']}  -> UNDER")
            print(f"    Historical: 56-58% under vs 52.4% break-even, but the")
            print(f"    threshold came from the data and 15mph does not hold up.")
            print(f"    Logging these is how we find out; ~200 live games needed.")

    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
