"""When Kalshi and the bookmakers disagree, which one is right?

THE POINT

Beating the market at forecasting is a closed question here: 68,304 matches,
blend weight 0.00, the model loses to the closing line everywhere it has been
measured. Improving the model means out-forecasting professionals using public
data, and every test says we do not.

But profit does not require a better forecast. It requires a better PRICE. We
already possess a forecast that provably beats ours — the bookmaker closing
line. If Kalshi, a thin retail venue, prices a fixture differently from that
line, the trade is to take Kalshi's side of the disagreement and let the
sharper line be right. No model improvement needed at all.

This measures whether that gap exists and, when it does, who wins.

WHAT WOULD MAKE IT REAL

Three things must all hold, and any one of them failing kills it:

  1. The prices must actually diverge. If Kalshi tracks the bookmakers, there
     is nothing to trade.
  2. The bookmaker line must be the more accurate of the two when they differ.
     Divergence alone is not edge — Kalshi could be the sharper one.
  3. The gap must exceed the cost of crossing it. Kalshi's spread runs from 1c
     to 29c depending on the market, plus a 0.07*p*(1-p) fee. A 3-point
     disagreement behind a 10c spread is a losing trade.

Reports all three rather than stopping at the first encouraging number.

RESULT (2026-08-12, 182 matched fixtures / 546 legs)

The direction is real and consistent. Bookmakers are significantly sharper
than Kalshi: gap +0.0056, 95% CI [+0.0015, +0.0100]. Both sides confirm it —
legs where Kalshi sits ABOVE the book line won only 8% while priced at 54%.
A naive simulation of buying Kalshi when it is >3% below the book returns
+14.4% ROI over 31 bets.

DO NOT TRADE THIS YET. The test contains lookahead and the apparent edge is
probably it.

The Kalshi price used here is sampled ~8h before the event (see
fetch_kalshi_closes.py — Kalshi exposes no reliable start time, so the sample
is anchored well back from close). The bookmaker figure is the CLOSING line.
So "Kalshi is 5% below the book" partly means "the market moved over the
following 8 hours" — information not available when the bet would be placed.
A closing line beating an 8-hour-old price is expected and untradeable.

To answer this properly both prices must be sampled at the SAME moment. That
needs bookmaker odds captured live alongside the Kalshi snapshot;
football-data.co.uk publishes odds only for completed matches, so the history
cannot answer it retrospectively. Collecting simultaneous pairs going forward
is the concrete next step, and until that exists the number above is a
hypothesis, not an edge.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from team_names import TeamResolver

EPS = 1e-15

# Kalshi soccer series -> the division code used by football-data / new_leagues
SERIES_DIV = {
    "KXMLSGAME": "USA:MLS",
    "KXLIGAMXGAME": "MEX:Liga MX",
    "KXALLSVENSKANGAME": "SWE:Allsvenskan",
    "KXELITESERIENGAME": "NOR:Eliteserien",
    "KXJLEAGUEGAME": "JPN:J1 League",
    "KXSCOTTISHPREMGAME": "SC0",
    "KXBUNDESLIGA2GAME": "D2",
}


def ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def load_book() -> pd.DataFrame:
    frames = []
    for f in ("football_data_raw.parquet", "new_leagues_raw.parquet"):
        p = ROOT / "data" / "raw" / f
        if p.exists():
            frames.append(pd.read_parquet(p))
    d = pd.concat(frames, ignore_index=True)
    d = d[d["FTR"].notna()]
    # Average closing odds, de-vigged. Not Pinnacle — football-data stopped
    # publishing it in 2025/26 — but FINDINGS showed the conclusion holds
    # against the market average too.
    for c in ("AvgH", "AvgD", "AvgA"):
        if c not in d.columns:
            return pd.DataFrame()
    d = d[d["AvgH"].notna() & d["AvgD"].notna() & d["AvgA"].notna()].copy()
    inv = 1 / d["AvgH"] + 1 / d["AvgD"] + 1 / d["AvgA"]
    d["bH"] = (1 / d["AvgH"]) / inv
    d["bD"] = (1 / d["AvgD"]) / inv
    d["bA"] = (1 / d["AvgA"]) / inv
    d["Date"] = pd.to_datetime(d["Date"])
    return d


def main() -> int:
    kal = pd.read_parquet(ROOT / "data" / "raw" / "kalshi_closes.parquet")
    kal = kal[kal["series"].isin(SERIES_DIV)]
    book = load_book()
    if book.empty:
        print("no bookmaker odds available")
        return 1

    rows = []
    for series, g in kal.groupby("series"):
        div = SERIES_DIV[series]
        b = book[book["Div"] == div]
        if b.empty:
            continue
        names = sorted(set(b["HomeTeam"].dropna()) | set(b["AwayTeam"].dropna()))
        res = TeamResolver(names)

        for ev, e in g.groupby("event_ticker"):
            if e["result"].eq("yes").sum() != 1:
                continue
            title = str(e["title"].iloc[0]).replace(" Winner?", "").strip()
            if " vs " not in title:
                continue
            ha, aa = [x.strip() for x in title.split(" vs ", 1)]
            h, a = res.resolve(ha), res.resolve(aa)
            if not h or not a:
                continue
            d0 = pd.Timestamp(str(e["date"].iloc[0]))
            m = b[(b["HomeTeam"] == h) & (b["AwayTeam"] == a)
                  & (b["Date"] >= d0 - pd.Timedelta(days=2))
                  & (b["Date"] <= d0 + pd.Timedelta(days=2))]
            if len(m) != 1:
                continue
            m = m.iloc[0]

            s = e["close"].sum()
            if not (0.8 < s < 1.6):
                continue
            legs = {}
            for _, r in e.iterrows():
                leg = str(r["leg"]).strip().lower()
                key = ("D" if leg in ("tie", "draw")
                       else "H" if res.resolve(str(r["leg"])) == h
                       else "A" if res.resolve(str(r["leg"])) == a else None)
                if key:
                    legs[key] = (r["close"] / s, r["result"] == "yes")
            if set(legs) != {"H", "D", "A"}:
                continue
            for k, bk in (("H", "bH"), ("D", "bD"), ("A", "bA")):
                rows.append({"series": series, "outcome": k,
                             "kalshi": legs[k][0], "book": float(m[bk]),
                             "won": int(legs[k][1])})

    df = pd.DataFrame(rows)
    print(f"matched {df['won'].count() // 3 if len(df) else 0} fixtures "
          f"present on BOTH Kalshi and the bookmakers "
          f"({len(df)} outcome legs)\n")
    if len(df) < 30:
        print("Too few matched fixtures to conclude anything.")
        if len(df):
            print(df.groupby('series').size())
        return 0

    df["gap"] = df["kalshi"] - df["book"]

    print("1) DO THE PRICES DIVERGE?")
    print(f"   mean |Kalshi - book| : {df['gap'].abs().mean():.1%}")
    print(f"   median               : {df['gap'].abs().median():.1%}")
    print(f"   legs >5pts apart     : {(df['gap'].abs() > .05).mean():.0%}")
    print(f"   legs >10pts apart    : {(df['gap'].abs() > .10).mean():.0%}")

    print("\n2) WHEN THEY DIVERGE, WHO IS RIGHT?")
    print(f"   {'disagreement':<16}{'legs':>6}{'Kalshi LL':>11}{'book LL':>10}"
          f"{'winner':>10}")
    for lo, hi in ((0, .03), (.03, .06), (.06, .10), (.10, 1.0)):
        s = df[(df["gap"].abs() >= lo) & (df["gap"].abs() < hi)]
        if len(s) < 15:
            continue
        kl, bl = ll(s["kalshi"], s["won"]), ll(s["book"], s["won"])
        who = "book" if bl < kl else "KALSHI"
        print(f"   {lo:.0%}-{hi:.0%}{'':<10}{len(s):>6}{kl:>11.4f}{bl:>10.4f}"
              f"{who:>10}")

    kl, bl = ll(df["kalshi"], df["won"]), ll(df["book"], df["won"])
    print(f"\n   overall: Kalshi {kl:.4f}   book {bl:.4f}   "
          f"gap {kl - bl:+.4f}")
    diff = (-(df["won"] * np.log(np.clip(df["kalshi"], EPS, 1 - EPS))
              + (1 - df["won"]) * np.log(np.clip(1 - df["kalshi"], EPS, 1 - EPS)))
            + (df["won"] * np.log(np.clip(df["book"], EPS, 1 - EPS))
               + (1 - df["won"]) * np.log(np.clip(1 - df["book"], EPS, 1 - EPS))))
    rng = np.random.default_rng(0)
    boot = np.array([diff.sample(len(diff), replace=True, random_state=int(rng.integers(1e9))).mean()
                     for _ in range(2000)])
    lo_, hi_ = np.percentile(boot, [2.5, 97.5])
    print(f"   95% CI on that gap: [{lo_:+.4f}, {hi_:+.4f}]")
    print(f"   -> {'bookmakers significantly sharper' if lo_ > 0 else 'not distinguishable'}")

    print("\n3) IS THE GAP BIGGER THAN THE COST OF CROSSING IT?")
    big = df[df["gap"].abs() > .05]
    if len(big):
        print(f"   legs where Kalshi is >5pts off the book line: {len(big)}")
        print(f"   mean edge if the book line is truth: "
              f"{big['gap'].abs().mean():.1%}")
        print("   ...against a Kalshi spread of 1c (MLB/WNBA) to 29c (Liga MX)")
        print("   plus a 0.07*p*(1-p) fee. Compare per market before trading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
