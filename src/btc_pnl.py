"""True realised P&L on Kalshi, computed from settlements.

Fills alone cannot answer this: on Kalshi, binary positions are usually held to
expiry rather than closed out, so the fills show only money going out. The
settlements endpoint carries what was paid, what the market resolved to, and
what came back.

Per settlement:
    cost   = yes_total_cost_dollars + no_total_cost_dollars
    payout = (winning side's contract count) x $1
    net    = payout - cost - fee_cost
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import kalshi_auth as KA

ROOT = Path(__file__).resolve().parents[1]


def fetch(max_pages: int = 20) -> pd.DataFrame:
    rows, cursor = [], None
    for _ in range(max_pages):
        p = {"limit": 200}
        if cursor:
            p["cursor"] = cursor
        j = KA.get("/trade-api/v2/portfolio/settlements", p)
        b = j.get("settlements", [])
        rows.extend(b)
        cursor = j.get("cursor")
        if not cursor or not b:
            break
    return pd.DataFrame(rows)


def num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def main():
    df = fetch()
    if df.empty:
        print("no settlements")
        return

    df["settled_time"] = pd.to_datetime(df["settled_time"], errors="coerce", utc=True)
    df["date"] = df["settled_time"].dt.date
    for c in ("yes_count_fp", "no_count_fp", "yes_total_cost_dollars",
              "no_total_cost_dollars", "fee_cost", "value", "revenue"):
        if c in df:
            df[c] = num(df[c])

    df["cost"] = df["yes_total_cost_dollars"] + df["no_total_cost_dollars"]
    # The winning side pays $1 per contract; `value` is 100 or 0 in cents.
    won_yes = df["market_result"].astype(str).str.lower() == "yes"
    df["payout"] = np.where(won_yes, df["yes_count_fp"], df["no_count_fp"]) * 1.0
    df["net"] = df["payout"] - df["cost"] - df["fee_cost"]
    df["series"] = df["ticker"].astype(str).str.split("-").str[0]

    print(f"settlements: {len(df):,}")
    print(f"  {df['settled_time'].min()} .. {df['settled_time'].max()}")
    print(f"  distinct days: {df['date'].nunique()}")

    print(f"\n{'=' * 66}\nOVERALL\n{'=' * 66}")
    print(f"  total staked (cost)   ${df['cost'].sum():>12,.2f}")
    print(f"  total payout          ${df['payout'].sum():>12,.2f}")
    print(f"  total fees            ${df['fee_cost'].sum():>12,.2f}")
    print(f"  NET P&L               ${df['net'].sum():>+12,.2f}")
    if df["cost"].sum() > 0:
        print(f"  return on staked      {df['net'].sum() / df['cost'].sum():>12.2%}")
    print(f"\n  fees as % of net loss: ", end="")
    if df["net"].sum() < 0:
        print(f"{df['fee_cost'].sum() / abs(df['net'].sum()):.0%}")
    else:
        print("n/a (profitable)")

    print(f"\n{'=' * 66}\nBY SERIES\n{'=' * 66}")
    g = df.groupby("series").agg(n=("ticker", "size"), cost=("cost", "sum"),
                                 payout=("payout", "sum"), fees=("fee_cost", "sum"),
                                 net=("net", "sum"))
    g["roi"] = g["net"] / g["cost"].replace(0, np.nan)
    print(g.round(2).to_string())

    print(f"\n{'=' * 66}\nBY DAY\n{'=' * 66}")
    d = df.groupby("date").agg(n=("ticker", "size"), cost=("cost", "sum"),
                               fees=("fee_cost", "sum"), net=("net", "sum"))
    d["cum"] = d["net"].cumsum()
    print(d.round(2).tail(20).to_string())

    print(f"\n{'=' * 66}\nWIN RATE\n{'=' * 66}")
    traded = df[df["cost"] > 0]
    wins = (traded["net"] > 0).sum()
    print(f"  settled markets with money at risk: {len(traded):,}")
    print(f"  profitable: {wins:,} ({wins/max(len(traded),1):.1%})")
    print(f"  mean net per market: ${traded['net'].mean():+.3f}")
    print(f"  median: ${traded['net'].median():+.3f}")
    print(f"  best ${traded['net'].max():+,.2f}   worst ${traded['net'].min():+,.2f}")

    # Hedged books: both sides held into settlement guarantees a loss if the
    # combined price paid exceeded $1.
    both = traded[(traded["yes_count_fp"] > 0) & (traded["no_count_fp"] > 0)]
    if len(both):
        pair = np.minimum(both["yes_count_fp"], both["no_count_fp"])
        implied = both["cost"] / pair.replace(0, np.nan)
        print(f"\n  markets where BOTH sides were held to expiry: {len(both):,} "
              f"({len(both)/len(traded):.0%})")
        print(f"    mean combined price paid per matched pair: ${implied.mean():.3f}")
        print("    (above $1.00 means the pair was locked in at a loss before")
        print("     the market even resolved)")

    out = ROOT / "reports" / "kalshi_settlements.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
