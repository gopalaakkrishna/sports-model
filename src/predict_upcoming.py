"""Predict upcoming fixtures and compare against the current market.

Fits each country's model as of today, forecasts every scheduled fixture, and
reports the model's probabilities next to the bookmakers' — including expected
value at the best available price and a fractional-Kelly stake.

The calibration produced by calibrate.py is applied by default. If that file
says the model has no edge over the closing line, the EV column is exactly what
it looks like: noise. Read reports/README before betting anything on it.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
import model as M
from team_names import TeamResolver

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
NEW_FIXTURES_URL = "https://www.football-data.co.uk/new_leagues_fixtures.csv"
EPS = 1e-15


def _read_csv(url: str) -> pd.DataFrame | None:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    if not r.content.strip():
        return None
    df = pd.read_csv(io.BytesIO(r.content), encoding="latin-1", on_bad_lines="skip")
    df.columns = [str(c).replace("﻿", "").replace("ï»¿", "").strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
    return df


def fetch_fixtures() -> pd.DataFrame:
    """European divisions plus the year-round leagues, in one frame."""
    frames = []

    euro = _read_csv(FIXTURES_URL)
    if euro is not None and "HomeTeam" in euro.columns:
        frames.append(euro.dropna(subset=["Date", "HomeTeam", "AwayTeam", "Div"]))

    new = _read_csv(NEW_FIXTURES_URL)
    if new is not None and "Home" in new.columns:
        # Map Country/League onto the "CODE:League" division ids used in history.
        name_to_code = {v: k for k, v in M.NEW_LEAGUE_COUNTRIES.items()}
        new = new.dropna(subset=["Date", "Home", "Away", "Country", "League"]).copy()
        new["Div"] = [
            f"{name_to_code.get(str(c).strip(), str(c).strip())}:{str(lg).strip()}"
            for c, lg in zip(new["Country"], new["League"])
        ]
        new = new.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam"})
        frames.append(new)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def devig(odds: np.ndarray) -> np.ndarray:
    inv = 1.0 / odds
    return inv / inv.sum(axis=1, keepdims=True)


def load_calibration(use: bool) -> dict:
    path = ROOT / "data" / "processed" / "calibration.json"
    if not use or not path.exists():
        return {"temperature": 1.0, "blend_weight_on_model": 1.0}
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xi", type=float, default=0.0018)
    ap.add_argument("--reg", type=float, default=2.0)
    ap.add_argument("--raw", action="store_true",
                    help="skip calibration and market blending (raw model output)")
    ap.add_argument("--kelly-fraction", type=float, default=0.25)
    ap.add_argument("--min-edge", type=float, default=0.02,
                    help="only flag bets where model prob exceeds market by this")
    args = ap.parse_args()

    cal = load_calibration(not args.raw)
    temp = cal.get("temperature", 1.0)
    w = cal.get("blend_weight_on_model", 1.0)

    import data as D

    hist = D.load_history()
    hist = hist[hist["FTHG"].notna()].copy()
    groups = D.country_groups(hist)
    fx = fetch_fixtures()
    today = pd.Timestamp.now().normalize()

    print(f"model run {today.date()}   history to {hist['Date'].max().date()}")
    print(f"calibration: temperature {temp:.3f}, weight on model {w:.3f}"
          f"{'  (RAW, uncalibrated)' if args.raw else ''}")
    print(f"{len(fx)} scheduled fixtures\n")

    out = []
    for country, divs in groups.items():
        games = fx[fx["Div"].isin(divs)]
        if games.empty:
            continue
        sub = hist[hist["Div"].isin(divs)]
        n_skipped = 0

        # The feed sometimes still lists matches that have already been played.
        # Fit as of each fixture's own date, never later, or the model would be
        # predicting matches that are already in its training data.
        for fix_date, games_d in games.groupby(games["Date"].dt.normalize()):
            as_of = min(fix_date, today) if fix_date <= today else fix_date
            try:
                fr = M.fit(sub, as_of, xi=args.xi, reg=args.reg)
            except ValueError as e:
                print(f"  {country} {fix_date.date()}: skipped ({e})")
                continue

            # The fixtures feed names teams differently from the season files.
            resolver = TeamResolver(fr.teams)
            for _, g in games_d.iterrows():
                home = resolver.resolve(g["HomeTeam"])
                away = resolver.resolve(g["AwayTeam"])
                if home is None or away is None:
                    n_skipped += 1
                    continue
                p = M.predict(fr, home, away, g["Div"])
                if p is None:
                    n_skipped += 1
                    continue
                # The model's own view, temperature-calibrated. This column
                # always stays the model — blending it with the market here
                # would make the "model vs market" comparison circular.
                mp = np.array([p["p_home"], p["p_draw"], p["p_away"]])
                mp = mp**temp
                mp = mp / mp.sum()

                # Market consensus, and the best price actually available.
                avg = np.array([g.get("AvgH"), g.get("AvgD"), g.get("AvgA")], float)
                mx = np.array([g.get("MaxH"), g.get("MaxD"), g.get("MaxA")], float)
                if np.isfinite(avg).all():
                    kp = devig(avg.reshape(1, 3))[0]
                    # Best available forecast per the backtest: blend at weight
                    # w. w came out 0, so this is the market — that is the
                    # finding, not a bug.
                    fused = w * mp + (1 - w) * kp
                else:
                    kp = np.full(3, np.nan)
                    fused = mp

                best = mx if np.isfinite(mx).all() else avg
                fin = np.isfinite(best).all()
                # EV the naive way (trusting the model outright) and the honest
                # way (trusting the backtest-optimal blend).
                ev_model = mp * best - 1.0 if fin else np.full(3, np.nan)
                ev_fused = fused * best - 1.0 if fin else np.full(3, np.nan)
                edge = mp - kp

                out.append({
                    "Date": g["Date"], "Time": g.get("Time", ""),
                    "played": bool(g["Date"].normalize() < today),
                    "Div": g["Div"], "Home": g["HomeTeam"], "Away": g["AwayTeam"],
                    "xg_h": p["lambda_home"], "xg_a": p["lambda_away"],
                    "p_H": mp[0], "p_D": mp[1], "p_A": mp[2],
                    "mkt_H": kp[0], "mkt_D": kp[1], "mkt_A": kp[2],
                    "fused_H": fused[0], "fused_D": fused[1], "fused_A": fused[2],
                    "edge_H": edge[0], "edge_D": edge[1], "edge_A": edge[2],
                    "ev_model_H": ev_model[0], "ev_model_D": ev_model[1],
                    "ev_model_A": ev_model[2],
                    "ev_fused_H": ev_fused[0], "ev_fused_D": ev_fused[1],
                    "ev_fused_A": ev_fused[2],
                    "best_H": best[0], "best_D": best[1], "best_A": best[2],
                    "p_over25": p["p_over25"], "p_btts": p["p_btts"],
                    "top_score": f"{p['top_scorelines'][0][0]}-{p['top_scorelines'][0][1]}",
                    "top_score_p": p["top_scorelines"][0][2],
                })

            if resolver.fuzzy_log or resolver.unresolved:
                print(f"  {country} {fix_date.date()}:")
                resolver.report()
        if n_skipped:
            print(f"  {country}: {n_skipped} fixture(s) dropped")

    if not out:
        print("No fixtures could be predicted.")
        return

    res = pd.DataFrame(out).sort_values(["Date", "Time"])
    outp = ROOT / "reports" / f"predictions_{today.date()}.csv"
    outp.parent.mkdir(exist_ok=True)
    res.to_csv(outp, index=False)

    n_future = int((~res["played"]).sum())
    print(f"{'DATE':<11}{'':<2}{'MATCH':<32}{'MODEL H/D/A':<17}{'MARKET H/D/A':<17}"
          f"{'xG':>11}{'SCORE':>7}{'O2.5':>7}")
    print("-" * 105)
    for _, r in res.iterrows():
        match = f"{r['Home']} v {r['Away']}"
        model_s = f"{r['p_H']:.0%}/{r['p_D']:.0%}/{r['p_A']:.0%}"
        mkt_s = ("n/a" if not np.isfinite(r["mkt_H"])
                 else f"{r['mkt_H']:.0%}/{r['mkt_D']:.0%}/{r['mkt_A']:.0%}")
        xg = f"{r['xg_h']:.2f}-{r['xg_a']:.2f}"
        mark = "* " if r["played"] else "  "
        print(f"{r['Date'].date()!s:<11}{mark}{match[:31]:<32}{model_s:<17}{mkt_s:<17}"
              f"{xg:>11}{r['top_score']:>7}{r['p_over25']:>7.0%}")

    if n_future < len(res):
        print(f"\n  * = already played. Each of these was predicted with the model")
        print("    fitted as of its own kickoff date, so no result leaked in.")
    if n_future == 0:
        print("\n  No FUTURE fixtures in the feed right now. Most European leagues")
        print("  restart mid-August; the feed fills in about a week ahead.")

    # Where the model most disagrees with the market, for matches not yet played.
    flags = []
    for _, r in res[~res["played"]].iterrows():
        for lbl, e, v, o in [
            ("H", r["edge_H"], r["ev_model_H"], r["best_H"]),
            ("D", r["edge_D"], r["ev_model_D"], r["best_D"]),
            ("A", r["edge_A"], r["ev_model_A"], r["best_A"]),
        ]:
            if np.isfinite(e) and e >= args.min_edge and np.isfinite(v) and v > 0:
                k = max(0.0, (v / (o - 1.0))) * args.kelly_fraction
                flags.append((r["Home"], r["Away"], lbl, e, v, o, k))

    print(f"\nLargest disagreements with the market (edge >= {args.min_edge:.0%}):")
    if not flags:
        print("  none")
    else:
        flags.sort(key=lambda z: -z[4])
        print(f"  {'MATCH':<40}{'PICK':<6}{'EDGE':>7}{'EV':>8}{'ODDS':>7}{'KELLY':>8}")
        for h, a, lbl, e, v, o, k in flags[:15]:
            print(f"  {(h + ' v ' + a)[:39]:<40}{lbl:<6}{e:>+6.1%}{v:>+8.1%}{o:>7.2f}{k:>7.2%}")

    print(f"\nsaved -> {outp}")
    if not args.raw and w < 0.3:
        print(f"\n  NOTE: the backtest put {w:.0%} weight on this model against the")
        print("  closing line, over 68,304 matches. The EDGE and EV columns above are")
        print("  the model disagreeing with the market, and the backtest says that")
        print("  disagreement is the model's error: betting it lost 4.6% flat-stake,")
        print("  and 7.4% when restricted to the largest edges. Read them as")
        print("  'where my model is probably wrong', not as picks.")


if __name__ == "__main__":
    main()
