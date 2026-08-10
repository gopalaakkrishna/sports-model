"""Can 15m be made good? Separate DIRECTIONAL edge from EXECUTION cost.

The headline "-1.80% gross" for KXBTC15M is contaminated: for any market that
was exited by buying the opposite side, `cost` includes BOTH legs, so the
round-trip spread is being counted as if it were a bad directional call.

The decisive question for whether 15m is fixable:
    if every position were simply HELD to settlement, is the directional
    call profitable before fees?

  - If YES, 15m is an EXECUTION problem (stop exiting, pay fewer fees) and is
    worth fixing.
  - If NO, the calls themselves are unprofitable and no execution change saves
    it.

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
fl["leg"] = fl["outcome_side"].astype(str).str.lower()
first = (fl.sort_values("created_time").groupby("market_ticker")
           .agg(first_leg=("leg", "first")))

m = st.merge(first, left_on="ticker", right_index=True, how="left")
m = m[(m["series"] == "KXBTC15M") & (m["cost"] > 0)].copy()
print(f"KXBTC15M settled markets with money at risk: {len(m):,}")

both = (m["yes_count_fp"] > 0) & (m["no_count_fp"] > 0)
print(f"  exited via opposite side (both legs): {both.sum():,}")
print(f"  single-sided (held)                 : {(~both).sum():,}")

won_yes = m["market_result"].astype(str).str.lower() == "yes"

# ---- AS TRADED -------------------------------------------------------------
gross_actual = m["payout"].sum() - m["cost"].sum()
print("\n" + "=" * 74)
print("AS TRADED (what actually happened)")
print("=" * 74)
print(f"  staked            ${m['cost'].sum():>12,.2f}")
print(f"  gross (pre-fee)   ${gross_actual:>+12,.2f}   ({gross_actual/m['cost'].sum():+.2%})")
print(f"  fees              ${m['fee_cost'].sum():>12,.2f}")
print(f"  NET               ${m['net'].sum():>+12,.2f}")

# ---- HELD-ONLY COUNTERFACTUAL ---------------------------------------------
# For every market, keep ONLY the first-entered side, held to settlement.
first_is_yes = m["first_leg"].fillna("") == "yes"
# markets with no fill record fall back to whichever side actually has size
fallback_yes = m["yes_count_fp"] >= m["no_count_fp"]
use_yes = np.where(m["first_leg"].isna(), fallback_yes, first_is_yes)

held_count = np.where(use_yes, m["yes_count_fp"], m["no_count_fp"])
held_cost = np.where(use_yes, m["yes_total_cost_dollars"], m["no_total_cost_dollars"])
held_won = np.where(use_yes, won_yes, ~won_yes)
held_payout = np.where(held_won, held_count, 0.0)

# fee for a single leg, apportioned by that leg's share of contracts
tot_ct = (m["yes_count_fp"] + m["no_count_fp"]).replace(0, np.nan)
held_fee = m["fee_cost"] * (held_count / tot_ct).fillna(1.0)

gross_held = held_payout.sum() - held_cost.sum()
net_held = gross_held - held_fee.sum()

print("\n" + "=" * 74)
print("COUNTERFACTUAL: every position simply HELD to settlement")
print("=" * 74)
print(f"  staked            ${held_cost.sum():>12,.2f}")
print(f"  gross (pre-fee)   ${gross_held:>+12,.2f}   ({gross_held/held_cost.sum():+.2%})  <-- DIRECTIONAL EDGE")
print(f"  fees (1 leg)      ${held_fee.sum():>12,.2f}")
print(f"  NET               ${net_held:>+12,.2f}")

print("\n" + "=" * 74)
print("VERDICT")
print("=" * 74)
print(f"  execution cost of exiting  ${gross_actual - gross_held:>+12,.2f}")
print(f"  fee saved by 1 leg not 2   ${m['fee_cost'].sum() - held_fee.sum():>+12,.2f}")
if gross_held > 0:
    print("\n  >>> DIRECTIONAL EDGE IS POSITIVE.")
    print("      15m is an EXECUTION problem, not a prediction problem.")
    print(f"      Held + current fees:      ${net_held:>+12,.2f}")
    # what if all taker fees became maker (0)?
    print(f"      Held + zero fees (maker): ${gross_held:>+12,.2f}")
else:
    print("\n  >>> DIRECTIONAL EDGE IS NEGATIVE EVEN HELD.")
    print("      No execution change makes 15m profitable.")
    print(f"      Even with ZERO fees it returns ${gross_held:+,.2f}")

# ---- how big does the gap have to be? --------------------------------------
print("\n" + "=" * 74)
print("BREAKEVEN: what win rate would 15m need?")
print("=" * 74)
avg_cost = (held_cost.sum() / max(held_count.sum(), 1)) * 100
wr = float(held_won.mean())
print(f"  avg entry cost per contract   {avg_cost:.1f}c")
print(f"  actual win rate (held)        {wr:.1%}")
print(f"  breakeven WR at that cost     {avg_cost:.1f}%  (before fees)")
fee_per_ct = held_fee.sum() / max(held_count.sum(), 1) * 100
print(f"  fee per contract              {fee_per_ct:.2f}c")
print(f"  breakeven WR incl. fees       {(avg_cost + fee_per_ct):.1f}%")
print(f"  GAP                           {wr*100 - (avg_cost + fee_per_ct):+.1f} points")
