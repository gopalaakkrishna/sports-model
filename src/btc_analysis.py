"""Analyse actual Kalshi BTC trading: cash flow, fees, and where money goes.

Kalshi fills give price, size, fee and whether the fill was a taker. For a fully
closed book (no open positions) realised P&L is simply net cash minus fees, so
this can be computed exactly rather than estimated.

The point of this file is to answer one question honestly: is the trading
profitable, and if not, what is taking the money?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load() -> pd.DataFrame:
    f = pd.read_csv(ROOT / "reports" / "kalshi_activity.csv")
    f["created_time"] = pd.to_datetime(f["created_time"], errors="coerce")
    f["date"] = f["created_time"].dt.date
    # Price paid/received depends on which side of the contract was traded.
    f["price"] = np.where(f["side"] == "yes",
                          f["yes_price_dollars"], f["no_price_dollars"])
    f["notional"] = f["price"] * f["count_fp"]
    # Buying costs cash, selling returns it.
    f["cash"] = np.where(f["action"] == "buy", -f["notional"], f["notional"])
    f["series"] = f["ticker"].astype(str).str.split("-").str[0]
    return f


def main():
    f = load()
    print(f"fills: {len(f)}   {f['created_time'].min()} .. {f['created_time'].max()}")
    days = f["date"].nunique()
    print(f"trading days: {days}   average {len(f)/days:.0f} fills/day")

    print(f"\n{'=' * 62}\nSIZE AND FEES\n{'=' * 62}")
    print(f"  total contracts traded   {f['count_fp'].sum():,.1f}")
    print(f"  mean contracts per fill  {f['count_fp'].mean():.2f}")
    print(f"  total notional traded    ${f['notional'].sum():,.2f}")
    print(f"  TOTAL FEES PAID          ${f['fee_cost'].sum():,.2f}")
    taker = f[f["is_taker"] == True]
    maker = f[f["is_taker"] != True]
    print(f"  taker fills {len(taker)} ({len(taker)/len(f):.0%}), fees ${taker['fee_cost'].sum():,.2f}")
    print(f"  maker fills {len(maker)} ({len(maker)/len(f):.0%}), fees ${maker['fee_cost'].sum():,.2f}")

    print(f"\n{'=' * 62}\nREALISED P&L\n{'=' * 62}")
    # Any market still holding a net position cannot be settled from fills alone.
    pos = f.groupby("ticker").apply(
        lambda g: (g.loc[g["side"] == "yes", "count_fp"] *
                   np.where(g.loc[g["side"] == "yes", "action"] == "buy", 1, -1)).sum(),
        include_groups=False).rename("net_yes")
    posn = f.groupby("ticker").apply(
        lambda g: (g.loc[g["side"] == "no", "count_fp"] *
                   np.where(g.loc[g["side"] == "no", "action"] == "buy", 1, -1)).sum(),
        include_groups=False).rename("net_no")
    cash = f.groupby("ticker")["cash"].sum().rename("cash")
    fees = f.groupby("ticker")["fee_cost"].sum().rename("fees")
    per = pd.concat([pos, posn, cash, fees], axis=1).fillna(0.0)
    per["flat"] = (per["net_yes"].abs() < 1e-6) & (per["net_no"].abs() < 1e-6)

    flat = per[per["flat"]]
    open_ = per[~per["flat"]]
    print(f"  markets traded            {len(per)}")
    print(f"  fully closed (flat)       {len(flat)}")
    print(f"  left with a position      {len(open_)}  "
          f"(settled at expiry; not computable from fills alone)")

    pnl_flat = float(flat["cash"].sum() - flat["fees"].sum())
    print(f"\n  On the {len(flat)} fully-closed markets:")
    print(f"    gross cash flow   ${flat['cash'].sum():+,.2f}")
    print(f"    fees              ${flat['fees'].sum():,.2f}")
    print(f"    NET REALISED P&L  ${pnl_flat:+,.2f}")
    if flat["fees"].sum() > 0:
        gross = flat["cash"].sum()
        print(f"    fees as a share of gross: "
              f"{abs(flat['fees'].sum() / gross):.0%}" if abs(gross) > 1e-9
              else "    (gross ~0: fees are the entire result)")

    print(f"\n{'=' * 62}\nBY SERIES\n{'=' * 62}")
    g = f.groupby("series").agg(fills=("fill_id", "size"),
                                contracts=("count_fp", "sum"),
                                notional=("notional", "sum"),
                                fees=("fee_cost", "sum"),
                                cash=("cash", "sum"))
    g["net"] = g["cash"] - g["fees"]
    print(g.round(2).to_string())

    print(f"\n{'=' * 62}\nBY DAY\n{'=' * 62}")
    d = f.groupby("date").agg(fills=("fill_id", "size"),
                              contracts=("count_fp", "sum"),
                              fees=("fee_cost", "sum"),
                              cash=("cash", "sum"))
    d["net"] = d["cash"] - d["fees"]
    print(d.round(2).to_string())

    print(f"\n  NOTE: cash flow on markets left open cannot be resolved without")
    print(f"  each market's settlement, so treat the flat-book figure as the")
    print(f"  reliable one and the rest as incomplete.")


if __name__ == "__main__":
    main()
