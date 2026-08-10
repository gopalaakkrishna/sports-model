"""Where the money actually goes: both-sides books, series split, taker/maker.

Follows btc_reconcile.py, which established that the -$12,467 total is real but
that btc_pnl.py's "$1.343 per pair" was an arithmetic artifact. This file asks
the question that finding raises: both-sides markets lose $15.4k while one-side
markets MAKE $2.9k, so what are the both-sides books actually made of?

Read-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

st = pd.read_csv(REPORTS / "reconcile_settlements.csv")
fl = pd.read_csv(REPORTS / "reconcile_fills.csv")

for c in ("yes_count_fp", "no_count_fp", "yes_total_cost_dollars",
          "no_total_cost_dollars", "fee_cost", "cost", "payout", "net"):
    st[c] = pd.to_numeric(st[c], errors="coerce").fillna(0.0)
st["settled_time"] = pd.to_datetime(st["settled_time"], errors="coerce", utc=True)
st["series"] = st["ticker"].astype(str).str.split("-").str[0]

print("=" * 78)
print("1. WHAT AM I ACTUALLY TRADING? (all series, not just BTC)")
print("=" * 78)
g = st.groupby("series").agg(n=("ticker", "size"), staked=("cost", "sum"),
                             fees=("fee_cost", "sum"), net=("net", "sum"))
g["roi"] = g["net"] / g["staked"].replace(0, np.nan)
g["fee_pct_staked"] = g["fees"] / g["staked"].replace(0, np.nan)
print(g.sort_values("staked", ascending=False).round(3).to_string())

print("\n" + "=" * 78)
print("2. BOTH-SIDES vs ONE-SIDE, PER SERIES  <-- the core question")
print("=" * 78)
traded = st[st["cost"] > 0].copy()
traded["both"] = (traded["yes_count_fp"] > 0) & (traded["no_count_fp"] > 0)
piv = traded.groupby(["series", "both"]).agg(
    n=("ticker", "size"), staked=("cost", "sum"),
    fees=("fee_cost", "sum"), net=("net", "sum")).reset_index()
piv["roi"] = piv["net"] / piv["staked"].replace(0, np.nan)
piv["net_per_mkt"] = piv["net"] / piv["n"]
print(piv.round(3).to_string(index=False))

print("\n" + "=" * 78)
print("3. ARE BOTH-SIDES BOOKS SIMULTANEOUS (arb) OR SEQUENTIAL (exit)?")
print("=" * 78)
print("If the YES and NO legs are seconds apart -> deliberate both-sides bet.")
print("If minutes apart -> a position that was exited by buying the other side.\n")

fl["created_time"] = pd.to_datetime(fl["created_time"], errors="coerce", utc=True)
fl["count_fp"] = pd.to_numeric(fl["count_fp"], errors="coerce").fillna(0.0)
fl["fee_cost"] = pd.to_numeric(fl["fee_cost"], errors="coerce").fillna(0.0)
fl["series"] = fl["market_ticker"].astype(str).str.split("-").str[0]
btc = fl[fl["series"].str.startswith("KXBTC")].copy()
print(f"BTC fills: {len(btc):,} of {len(fl):,} total fills")

# per market: first yes-side fill vs first no-side fill
if len(btc):
    btc["leg"] = btc["outcome_side"].astype(str).str.lower()
    grp = btc.groupby(["market_ticker", "leg"])["created_time"].agg(["min", "max", "count"])
    grp = grp.unstack("leg")
    have_both = grp[("min", "yes")].notna() & grp[("min", "no")].notna()
    bb = grp[have_both].copy()
    if len(bb):
        gap = (bb[("min", "yes")] - bb[("min", "no")]).abs().dt.total_seconds()
        print(f"\nmarkets with both legs filled: {len(bb):,}")
        print(f"  gap between first YES fill and first NO fill:")
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            print(f"    p{int(q*100):>2}: {gap.quantile(q):>8.1f}s")
        print(f"  within 5s (simultaneous / arb-like): {(gap <= 5).mean():.1%}")
        print(f"  over 60s (sequential / exit-like)  : {(gap > 60).mean():.1%}")

print("\n" + "=" * 78)
print("4. TAKER vs MAKER — BTC ONLY (the 77% figure needs checking)")
print("=" * 78)
if len(btc):
    btc["is_taker"] = btc["is_taker"].astype(str).str.lower().isin(["true", "1"])
    print(f"  BTC fills      : {len(btc):,}")
    print(f"  taker          : {btc['is_taker'].sum():,} ({btc['is_taker'].mean():.1%})")
    print(f"  maker          : {(~btc['is_taker']).sum():,} ({(~btc['is_taker']).mean():.1%})")
    tf = btc.groupby("is_taker")["fee_cost"].agg(["sum", "mean", "size"])
    tf.index = ["maker", "taker"]
    print("\n  fees by fill type:")
    print(tf.round(4).to_string())
    print(f"\n  total BTC fill fees: ${btc['fee_cost'].sum():,.2f}")
    # fee per contract
    btc["fee_per_ct"] = btc["fee_cost"] / btc["count_fp"].replace(0, np.nan)
    print(f"  mean fee per contract, taker: ${btc.loc[btc['is_taker'],'fee_per_ct'].mean():.4f}")
    mk = btc.loc[~btc["is_taker"], "fee_per_ct"]
    print(f"  mean fee per contract, maker: ${mk.mean():.4f}" if len(mk) else "  no maker fills")
    # by series
    print("\n  taker rate by series:")
    print(btc.groupby("series")["is_taker"].agg(["mean", "size"]).round(3).to_string())

print("\n" + "=" * 78)
print("5. FEE DRAG PER UNIT OF EDGE — why 15m loses and daily wins")
print("=" * 78)
for s in ["KXBTC15M", "KXBTCD"]:
    sub = traded[traded["series"] == s]
    if not len(sub):
        continue
    gross = sub["payout"].sum() - sub["cost"].sum()   # before fees
    fees = sub["fee_cost"].sum()
    print(f"\n  {s}: n={len(sub):,}")
    print(f"    staked           ${sub['cost'].sum():>11,.2f}")
    print(f"    gross P&L (pre-fee) ${gross:>+9,.2f}   ({gross/sub['cost'].sum():+.2%} of staked)")
    print(f"    fees             ${fees:>11,.2f}   ({fees/sub['cost'].sum():.2%} of staked)")
    print(f"    NET              ${sub['net'].sum():>+11,.2f}")
    print(f"    avg stake/market ${sub['cost'].mean():>11,.2f}")
    print(f"    fee per market   ${fees/len(sub):>11,.4f}")
    if gross > 0:
        print(f"    -> fees eat {fees/gross:.0%} of gross edge")
    else:
        print(f"    -> NO gross edge to begin with (loses before fees)")

print("\n" + "=" * 78)
print("6. IS THERE EDGE BEFORE FEES? (the decisive question)")
print("=" * 78)
for s in sorted(traded["series"].unique()):
    sub = traded[traded["series"] == s]
    if len(sub) < 30:
        continue
    gross = sub["payout"].sum() - sub["cost"].sum()
    print(f"  {s:<14} n={len(sub):>5,}  gross {gross:>+10,.2f}  "
          f"fees {sub['fee_cost'].sum():>9,.2f}  net {sub['net'].sum():>+10,.2f}  "
          f"{'EDGE' if gross > 0 else 'NO EDGE'}")
