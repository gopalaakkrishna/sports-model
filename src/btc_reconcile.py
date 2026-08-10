"""Reconcile Kalshi P&L from first principles and resolve the both-sides puzzle.

Written to CHECK btc_pnl.py rather than reuse its assumptions. Three independent
views of the same account:

  1. SETTLEMENTS  - what btc_pnl.py uses. Cost/payout/fee per settled market.
  2. FILLS        - every execution, with taker/maker flag and per-fill fee.
  3. BALANCE      - the account's own number, as a cross-check on 1 and 2.

If (1) and (2) disagree, one of them is mismeasuring, and the write-up must say
which before any conclusion is drawn from it.

Read-only. No order placement anywhere in this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import kalshi_auth as KA

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def page(path: str, key: str, limit: int = 200, max_pages: int = 400) -> pd.DataFrame:
    rows, cursor, pages = [], None, 0
    while pages < max_pages:
        p = {"limit": limit}
        if cursor:
            p["cursor"] = cursor
        j = KA.get(path, p)
        batch = j.get(key, [])
        rows.extend(batch)
        cursor = j.get("cursor")
        pages += 1
        if not cursor or not batch:
            break
    print(f"  {path} -> {len(rows):,} rows ({pages} pages)")
    return pd.DataFrame(rows)


def num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def main():
    print("=" * 78)
    print("FETCHING (read-only)")
    print("=" * 78)
    st = page("/trade-api/v2/portfolio/settlements", "settlements")
    fl = page("/trade-api/v2/portfolio/fills", "fills")
    bal = KA.get("/trade-api/v2/portfolio/balance")
    balance = bal.get("balance", 0) / 100.0

    # ---------------------------------------------------------------- fields
    print("\n" + "=" * 78)
    print("RAW FIELDS (so nothing is assumed about the schema)")
    print("=" * 78)
    print("settlements columns:", sorted(st.columns.tolist()))
    print("fills columns      :", sorted(fl.columns.tolist()))

    # ------------------------------------------------------------ settlements
    for c in ("yes_count_fp", "no_count_fp", "yes_total_cost_dollars",
              "no_total_cost_dollars", "fee_cost", "value", "revenue"):
        if c in st:
            st[c] = num(st[c])
    st["settled_time"] = pd.to_datetime(st["settled_time"], errors="coerce", utc=True)
    st["series"] = st["ticker"].astype(str).str.split("-").str[0]
    st["cost"] = st["yes_total_cost_dollars"] + st["no_total_cost_dollars"]
    won_yes = st["market_result"].astype(str).str.lower() == "yes"
    st["payout"] = np.where(won_yes, st["yes_count_fp"], st["no_count_fp"]) * 1.0
    st["net"] = st["payout"] - st["cost"] - st["fee_cost"]

    print("\n" + "=" * 78)
    print("VIEW 1 - SETTLEMENTS (reproduces btc_pnl.py)")
    print("=" * 78)
    print(f"  n                {len(st):,}")
    print(f"  window           {st['settled_time'].min()} .. {st['settled_time'].max()}")
    print(f"  staked (cost)    ${st['cost'].sum():>12,.2f}")
    print(f"  payout           ${st['payout'].sum():>12,.2f}")
    print(f"  fees             ${st['fee_cost'].sum():>12,.2f}")
    print(f"  NET              ${st['net'].sum():>+12,.2f}")
    print(f"  ROI on staked    {st['net'].sum()/max(st['cost'].sum(),1e-9):>12.2%}")

    # ------------------------------------------------------ THE BOTH-SIDES PUZZLE
    print("\n" + "=" * 78)
    print("THE BOTH-SIDES PUZZLE")
    print("=" * 78)
    traded = st[st["cost"] > 0]
    both = traded[(traded["yes_count_fp"] > 0) & (traded["no_count_fp"] > 0)].copy()
    print(f"  markets with BOTH sides settling: {len(both):,} of {len(traded):,} "
          f"({len(both)/max(len(traded),1):.0%})")

    if len(both):
        # -- the ORIGINAL (suspect) calculation from btc_pnl.py -----------
        pair = np.minimum(both["yes_count_fp"], both["no_count_fp"])
        old = (both["cost"] / pair.replace(0, np.nan)).mean()
        print(f"\n  [A] btc_pnl.py method  : ${old:.3f} per 'matched pair'")
        print("      cost of ALL contracts / count of MATCHED pairs only.")

        # -- correct: mean price actually paid on each side ---------------
        both["yes_avg"] = both["yes_total_cost_dollars"] / both["yes_count_fp"].replace(0, np.nan)
        both["no_avg"] = both["no_total_cost_dollars"] / both["no_count_fp"].replace(0, np.nan)
        both["pair_price"] = both["yes_avg"] + both["no_avg"]
        print(f"\n  [B] correct method     : ${both['pair_price'].mean():.3f} per pair")
        print("      avg YES price + avg NO price. >$1.00 = real locked loss.")
        print(f"      median ${both['pair_price'].median():.3f}   "
              f"min ${both['pair_price'].min():.3f}   max ${both['pair_price'].max():.3f}")
        under = (both["pair_price"] < 1.0).mean()
        print(f"      pairs bought UNDER $1.00 (locked PROFIT): {under:.1%}")

        # -- how lopsided are these books? --------------------------------
        both["imbalance"] = (
            (both["yes_count_fp"] - both["no_count_fp"]).abs()
            / (both["yes_count_fp"] + both["no_count_fp"])
        )
        print(f"\n  size imbalance between the two sides:")
        print(f"      mean {both['imbalance'].mean():.1%}   median {both['imbalance'].median():.1%}")
        print(f"      truly balanced (<10% apart): {(both['imbalance']<0.10).mean():.1%}")
        print("      A lopsided book is a position that was partly exited by")
        print("      buying the other side, NOT a deliberate both-sides bet.")

        # -- do these markets actually lose money? ------------------------
        print(f"\n  ACTUAL net on both-sides markets: ${both['net'].sum():+,.2f} "
              f"(mean ${both['net'].mean():+.3f}/market)")
        one = traded[~traded.index.isin(both.index)]
        print(f"  ACTUAL net on one-side markets  : ${one['net'].sum():+,.2f} "
              f"(mean ${one['net'].mean():+.3f}/market)")
        print("  If [A] were real, both-sides markets would lose ~34c per pair.")

    # ------------------------------------------------------------- fills view
    print("\n" + "=" * 78)
    print("VIEW 2 - FILLS (independent of settlements)")
    print("=" * 78)
    if not fl.empty:
        print("  sample row:", {k: fl.iloc[0][k] for k in list(fl.columns)[:12]})
        for c in ("count", "yes_price", "no_price", "price"):
            if c in fl:
                fl[c] = num(fl[c])
        fl["created_time"] = pd.to_datetime(fl.get("created_time"), errors="coerce", utc=True)
        if "is_taker" in fl:
            print(f"\n  taker fills: {fl['is_taker'].sum():,} / {len(fl):,} "
                  f"({fl['is_taker'].mean():.1%})")
        # money out on buys
        if {"action", "side", "count"}.issubset(fl.columns):
            print("\n  action x side counts:")
            print(fl.groupby(["action", "side"]).agg(
                n=("count", "size"), contracts=("count", "sum")).to_string())

    # ---------------------------------------------------------------- balance
    print("\n" + "=" * 78)
    print("VIEW 3 - BALANCE (account's own number)")
    print("=" * 78)
    print(f"  current balance ${balance:,.2f}")
    print("  NOTE: balance alone cannot validate P&L without the deposit/withdrawal")
    print("  history. Treat it as a sanity bound, not a reconciliation.")

    st.to_csv(REPORTS / "reconcile_settlements.csv", index=False)
    if not fl.empty:
        fl.to_csv(REPORTS / "reconcile_fills.csv", index=False)
    print(f"\nsaved -> {REPORTS}")


if __name__ == "__main__":
    main()
