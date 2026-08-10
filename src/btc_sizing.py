"""Do bigger positions do worse?

Flagged in btc_15m_potential.py and never chased: per CONTRACT the held 15m book
shows +2.3 points of edge over breakeven, but DOLLAR-weighted it is -1.50%. Those
two only disagree if the larger positions performed worse than the small ones.

If real, size is a lever on its own -- and with a $130 bankroll it matters more
than any signal tweak.

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
first = fl.sort_values("created_time").groupby("market_ticker").agg(first_leg=("leg", "first"))
m = st.merge(first, left_on="ticker", right_index=True, how="left")
m = m[m["cost"] > 0].copy()

# held basis (first leg only) so exit costs don't pollute the size signal
won_yes = m["market_result"].astype(str).str.lower() == "yes"
fallback_yes = m["yes_count_fp"] >= m["no_count_fp"]
use_yes = np.where(m["first_leg"].isna(), fallback_yes, m["first_leg"] == "yes")
m["ct"] = np.where(use_yes, m["yes_count_fp"], m["no_count_fp"])
m["cst"] = np.where(use_yes, m["yes_total_cost_dollars"], m["no_total_cost_dollars"])
m["won"] = np.where(use_yes, won_yes, ~won_yes)
m["pay"] = np.where(m["won"], m["ct"], 0.0)
tot = (m["yes_count_fp"] + m["no_count_fp"]).replace(0, np.nan)
m["fee"] = m["fee_cost"] * (m["ct"] / tot).fillna(1.0)
m["pnl"] = m["pay"] - m["cst"] - m["fee"]
m = m[m["ct"] > 0].copy()
m["entry_c"] = 100 * m["cst"] / m["ct"]
m["ev_c"] = 100 * m["pnl"] / m["ct"]          # per-contract result
m = m[(m["entry_c"] > 0) & (m["entry_c"] < 100)]

print(f"markets on held basis: {len(m):,}   contracts: {m['ct'].sum():,.0f}")


def by_size(sub, label):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    if len(sub) < 50:
        print("  too few")
        return
    q = sub["cst"].quantile([.2, .4, .6, .8]).values
    bands = [(0, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], 1e9)]
    names = ["smallest 20%", "20-40%", "40-60%", "60-80%", "largest 20%"]
    print(f"{'stake band':<16}{'$ range':>18}{'n':>6}{'WR':>8}{'entry':>8}"
          f"{'EV/ct':>9}{'total $':>11}")
    print("-" * 78)
    for (lo, hi), nm in zip(bands, names):
        b = sub[(sub["cst"] >= lo) & (sub["cst"] < hi)]
        if not len(b):
            continue
        ct = b["ct"].sum()
        print(f"{nm:<16}{f'${lo:,.0f}-{min(hi,99999):,.0f}':>18}{len(b):>6}"
              f"{100*b['won'].mean():>7.1f}%{100*b['cst'].sum()/ct:>7.1f}c"
              f"{100*b['pnl'].sum()/ct:>+8.2f}c{b['pnl'].sum():>+11,.2f}")
    # correlation between stake and per-contract outcome
    r = np.corrcoef(np.log10(sub["cst"].clip(lower=0.01)), sub["ev_c"])[0, 1]
    print(f"\n  corr(log stake, EV per contract) = {r:+.3f}"
          f"   {'-> bigger did WORSE' if r < -0.02 else '-> no clear size effect' if abs(r) <= 0.02 else '-> bigger did BETTER'}")
    # contract-weighted vs dollar-weighted
    cw = 100 * sub["pnl"].sum() / sub["ct"].sum()
    dw = 100 * sub["pnl"].sum() / sub["cst"].sum()
    print(f"  per-contract EV {cw:+.2f}c   |   return on stake {dw:+.2f}%")


by_size(m, "ALL SERIES")
by_size(m[m["series"] == "KXBTC15M"], "KXBTC15M (15-minute)")
by_size(m[m["series"] == "KXBTCD"], "KXBTCD (hourly)")

print(f"\n{'=' * 78}\nTHE BIGGEST BETS\n{'=' * 78}")
top = m.nlargest(15, "cst")[["ticker", "cst", "ct", "entry_c", "won", "pnl"]]
print(top.to_string(index=False,
      formatters={"cst": "${:,.0f}".format, "ct": "{:,.0f}".format,
                  "entry_c": "{:.0f}c".format, "pnl": "${:+,.2f}".format}))
print(f"\n  top 15 by stake: {top['won'].mean():.0%} WR, net ${top['pnl'].sum():+,.2f}")
print(f"  all others     : {m.drop(top.index)['won'].mean():.0%} WR, "
      f"net ${m.drop(top.index)['pnl'].sum():+,.2f}")

print(f"\n{'=' * 78}\nWHAT A FLAT STAKE WOULD HAVE DONE\n{'=' * 78}")
print("  Same trades, same directions, but every position the SAME size.")
flat_ct = m["ct"].median()
flat_pnl = (m["ev_c"] / 100 * flat_ct).sum()
print(f"  actual (as sized)      ${m['pnl'].sum():>+11,.2f}")
print(f"  flat {flat_ct:,.0f} contracts each  ${flat_pnl:>+11,.2f}")
print(f"  difference             ${flat_pnl - m['pnl'].sum():>+11,.2f}")
print("\n  CAVEAT: a flat stake is not costless -- it caps the good trades too.")
print("  This isolates the SIZE decision only, holding direction and timing fixed.")
