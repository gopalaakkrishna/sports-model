"""Did exiting via the opposite side help or hurt?

btc_deep.py established that 82.5% of both-sides books are sequential (median
166s between legs), i.e. positions exited by buying the other side rather than
deliberate arbitrage. Those books lost $15.4k while one-side books made $2.9k.

But that comparison alone does NOT prove exiting is bad: you would exit
precisely when a position is going against you, so exited trades are a
self-selected losing sample. The honest test is the counterfactual --
what would the ORIGINAL position have returned if simply held to settlement?

Uses fill ordering to identify which side was entered first.
Read-only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

st = pd.read_csv(REPORTS / "reconcile_settlements.csv")
fl = pd.read_csv(REPORTS / "reconcile_fills.csv")

for c in ("yes_count_fp", "no_count_fp", "yes_total_cost_dollars",
          "no_total_cost_dollars", "fee_cost", "cost", "payout", "net"):
    st[c] = pd.to_numeric(st[c], errors="coerce").fillna(0.0)
st["series"] = st["ticker"].astype(str).str.split("-").str[0]

fl["created_time"] = pd.to_datetime(fl["created_time"], errors="coerce", utc=True)
for c in ("count_fp", "fee_cost", "yes_price_dollars", "no_price_dollars"):
    fl[c] = pd.to_numeric(fl[c], errors="coerce").fillna(0.0)
fl["leg"] = fl["outcome_side"].astype(str).str.lower()

# first leg per market = the side originally entered
first = (fl.sort_values("created_time")
           .groupby("market_ticker")
           .agg(first_leg=("leg", "first"),
                first_time=("created_time", "first"),
                n_fills=("fill_id", "size")))

m = st.merge(first, left_on="ticker", right_index=True, how="inner")
m["both"] = (m["yes_count_fp"] > 0) & (m["no_count_fp"] > 0)
both = m[m["both"] & (m["cost"] > 0)].copy()
print(f"both-sides markets matched to fills: {len(both):,}")

# Counterfactual: hold ONLY the first-entered side to settlement.
#   payout_orig = count(first side) x $1 if that side won, else 0
#   cost_orig   = total cost paid on that side
won_yes = both["market_result"].astype(str).str.lower() == "yes"
first_is_yes = both["first_leg"] == "yes"

both["orig_count"] = np.where(first_is_yes, both["yes_count_fp"], both["no_count_fp"])
both["orig_cost"] = np.where(first_is_yes, both["yes_total_cost_dollars"],
                             both["no_total_cost_dollars"])
orig_won = np.where(first_is_yes, won_yes, ~won_yes)
both["orig_payout"] = np.where(orig_won, both["orig_count"], 0.0)

# fee on the first leg only, apportioned by contract share
share = both["orig_count"] / (both["yes_count_fp"] + both["no_count_fp"]).replace(0, np.nan)
both["orig_fee"] = both["fee_cost"] * share.fillna(0.5)
both["orig_net"] = both["orig_payout"] - both["orig_cost"] - both["orig_fee"]

print("\n" + "=" * 78)
print("COUNTERFACTUAL: exit-via-opposite-side  vs  hold original side")
print("=" * 78)
for s in ["KXBTC15M", "KXBTCD"]:
    sub = both[both["series"] == s]
    if len(sub) < 20:
        continue
    act = sub["net"].sum()
    cf = sub["orig_net"].sum()
    print(f"\n  {s}  (n={len(sub):,})")
    print(f"    ACTUAL (exited)        ${act:>+11,.2f}   per market ${act/len(sub):>+8.3f}")
    print(f"    COUNTERFACTUAL (held)  ${cf:>+11,.2f}   per market ${cf/len(sub):>+8.3f}")
    print(f"    exiting was worth      ${act-cf:>+11,.2f}   "
          f"({'HELPED' if act > cf else 'HURT'})")
    better = (sub["net"] > sub["orig_net"]).mean()
    print(f"    markets where exiting beat holding: {better:.1%}")

allsub = both[both["series"].isin(["KXBTC15M", "KXBTCD"])]
act, cf = allsub["net"].sum(), allsub["orig_net"].sum()
print(f"\n  COMBINED: actual ${act:+,.2f} vs held ${cf:+,.2f} -> "
      f"exiting was worth ${act-cf:+,.2f}")
print("  CAVEAT: the counterfactual assumes the original size would have been")
print("  held unchanged, and apportions fees by contract share. It is an")
print("  estimate, not an exact replay.")

print("\n" + "=" * 78)
print("COST OF A ROUND TRIP (what exiting actually costs mechanically)")
print("=" * 78)
both["yes_avg"] = both["yes_total_cost_dollars"] / both["yes_count_fp"].replace(0, np.nan)
both["no_avg"] = both["no_total_cost_dollars"] / both["no_count_fp"].replace(0, np.nan)
both["pair_price"] = both["yes_avg"] + both["no_avg"]
matched = np.minimum(both["yes_count_fp"], both["no_count_fp"])
both["locked_loss"] = (both["pair_price"] - 1.0) * matched
print(f"  mean pair price          ${both['pair_price'].mean():.4f}")
print(f"  => spread cost per pair  ${both['pair_price'].mean()-1:.4f}")
print(f"  matched contracts        {matched.sum():,.0f}")
print(f"  total locked spread loss ${both['locked_loss'].sum():+,.2f}")
print(f"  fees on both-sides mkts  ${both['fee_cost'].sum():,.2f}")
print(f"  => round-trip drag       ${both['locked_loss'].sum()-both['fee_cost'].sum():+,.2f}")

print("\n" + "=" * 78)
print("MAKER SAVINGS — realistic bound")
print("=" * 78)
btc = fl[fl["market_ticker"].astype(str).str.startswith("KXBTC")].copy()
btc["is_taker"] = btc["is_taker"].astype(str).str.lower().isin(["true", "1"])
tk = btc[btc["is_taker"]]
print(f"  BTC taker fills   {len(tk):,}   fees ${tk['fee_cost'].sum():,.2f}")
print(f"  BTC maker fills   {(~btc['is_taker']).sum():,}   fees "
      f"${btc.loc[~btc['is_taker'],'fee_cost'].sum():,.2f}  (Kalshi maker fee = $0)")
print(f"\n  CEILING if 100% of taker fills became maker: ${tk['fee_cost'].sum():,.2f} saved")
for conv in (0.25, 0.50, 0.75):
    print(f"    at {conv:.0%} conversion: ${tk['fee_cost'].sum()*conv:,.2f}")
print("\n  This is a CEILING, not a forecast. Resting a limit order means:")
print("    - it may never fill (you miss the trade entirely)")
print("    - it fills preferentially when the market moves against you")
print("      (adverse selection), which costs more than the fee saved")
print("  Neither effect is measurable from fill data alone -- it needs a live")
print("  test. Do NOT treat the ceiling as expected savings.")

print("\n" + "=" * 78)
print("BOTTOM LINE ARITHMETIC")
print("=" * 78)
tot = st[st["cost"] > 0]
print(f"  total net                    ${tot['net'].sum():>+11,.2f}")
for s in ["KXBTC15M", "KXBTCD"]:
    sub = tot[tot["series"] == s]
    gross = sub["payout"].sum() - sub["cost"].sum()
    print(f"  {s:<10} gross(pre-fee) ${gross:>+11,.2f}   fees ${sub['fee_cost'].sum():>9,.2f}"
          f"   net ${sub['net'].sum():>+11,.2f}")
other = tot[~tot["series"].isin(["KXBTC15M", "KXBTCD"])]
print(f"  everything else              ${other['net'].sum():>+11,.2f}  (n={len(other)})")
