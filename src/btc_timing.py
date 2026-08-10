"""Best lock time: win rate and EV by minutes-before-close, on real fills.

Ticker encodes the window close:
    KXBTC15M-26AUG051845-45  -> 2026-08-05 18:45 UTC
    KXBTCD-26AUG0519         -> 2026-08-05 hour 19
First fill on a market = when the position was actually entered.

EV is computed on the HELD basis (first leg only, to settlement) so that the
round-trip cost of exiting does not contaminate the timing signal -- see
btc_15m_potential.py for why the as-traded numbers are misleading.

Read-only.
"""
from __future__ import annotations

import re
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
st["settled_time"] = pd.to_datetime(st["settled_time"], errors="coerce", utc=True)
st["series"] = st["ticker"].astype(str).str.split("-").str[0]

MON = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def close_time(tk: str):
    """Parse the window close instant out of the ticker.

    Kalshi stamps these in US EASTERN, not UTC: a first pass reading them as UTC
    put every close exactly 240.2 min (4h = the EDT offset) before its own
    settlement. Localising to America/New_York also gets EST/EDT right across
    the DST boundary rather than hard-coding -4.
    """
    if not isinstance(tk, str):
        return pd.NaT
    p = tk.split("-")
    if len(p) < 2:
        return pd.NaT
    body = p[1]
    naive = None
    m = re.match(r"^(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})$", body)   # 15m: +HHMM
    if m:
        yy, mon, dd, hh, mi = m.groups()
        naive = pd.Timestamp(2000 + int(yy), MON[mon], int(dd), int(hh), int(mi))
    else:
        m = re.match(r"^(\d{2})([A-Z]{3})(\d{2})(\d{2})$", body)      # hourly: +HH
        if m:
            yy, mon, dd, hh = m.groups()
            naive = pd.Timestamp(2000 + int(yy), MON[mon], int(dd), int(hh), 0)
    if naive is None:
        return pd.NaT
    try:
        return naive.tz_localize("America/New_York").tz_convert("UTC")
    except Exception:
        return pd.NaT


st["close"] = st["ticker"].map(close_time)
print("close-time parse check (close should sit just BEFORE settled_time):")
chk = st.dropna(subset=["close"]).copy()
chk["lag_min"] = (chk["settled_time"] - chk["close"]).dt.total_seconds() / 60
for s in ("KXBTC15M", "KXBTCD"):
    sub = chk[chk["series"] == s]
    if len(sub):
        print(f"  {s}: n={len(sub):,}  settle-minus-close median {sub['lag_min'].median():.1f} min"
              f"  (p10 {sub['lag_min'].quantile(.10):.1f}, p90 {sub['lag_min'].quantile(.90):.1f})")

fl["created_time"] = pd.to_datetime(fl["created_time"], errors="coerce", utc=True)
fl["leg"] = fl["outcome_side"].astype(str).str.lower()
first = (fl.sort_values("created_time").groupby("market_ticker")
           .agg(entry=("created_time", "first"), first_leg=("leg", "first")))

m = st.merge(first, left_on="ticker", right_index=True, how="inner")
m = m[(m["cost"] > 0) & m["close"].notna()].copy()
m["mins_left"] = (m["close"] - m["entry"]).dt.total_seconds() / 60

# held basis: first leg only, to settlement
won_yes = m["market_result"].astype(str).str.lower() == "yes"
use_yes = m["first_leg"] == "yes"
m["h_ct"] = np.where(use_yes, m["yes_count_fp"], m["no_count_fp"])
m["h_cost"] = np.where(use_yes, m["yes_total_cost_dollars"], m["no_total_cost_dollars"])
h_won = np.where(use_yes, won_yes, ~won_yes)
m["h_pay"] = np.where(h_won, m["h_ct"], 0.0)
tot = (m["yes_count_fp"] + m["no_count_fp"]).replace(0, np.nan)
m["h_fee"] = m["fee_cost"] * (m["h_ct"] / tot).fillna(1.0)
m["h_net"] = m["h_pay"] - m["h_cost"] - m["h_fee"]
m["h_won"] = h_won


def table(sub, label, bins):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    if not len(sub):
        print("  no data")
        return
    print(f"{'mins before close':<22}{'n':>6}{'WR':>8}{'avg cost':>10}"
          f"{'EV/contract':>14}{'total':>12}")
    print("-" * 78)
    for lo, hi in bins:
        b = sub[(sub["mins_left"] >= lo) & (sub["mins_left"] < hi)]
        if len(b) < 15:
            print(f"{f'{lo}-{hi}':<22}{len(b):>6}   (too few)")
            continue
        ct = b["h_ct"].sum()
        wr = 100 * b["h_won"].mean()
        avg = 100 * b["h_cost"].sum() / max(ct, 1)
        evc = 100 * b["h_net"].sum() / max(ct, 1)
        print(f"{f'{lo}-{hi}':<22}{len(b):>6}{wr:>7.1f}%{avg:>9.1f}c"
              f"{evc:>+13.2f}c{b['h_net'].sum():>+12,.2f}")


f15 = m[m["series"] == "KXBTC15M"]
table(f15, "KXBTC15M — 15-MINUTE WINDOWS (held basis, net of fees)",
      [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, 30)])

fhr = m[m["series"] == "KXBTCD"]
table(fhr, "KXBTCD — HOURLY WINDOWS (held basis, net of fees)",
      [(0, 3), (3, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 45), (45, 60), (60, 120)])

print(f"\n{'=' * 78}\nBEST WINDOW SUMMARY\n{'=' * 78}")
for name, sub, bins in (
    ("15m", f15, [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 10), (10, 15)]),
    ("hourly", fhr, [(0, 3), (3, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 45), (45, 60)]),
):
    rows = []
    for lo, hi in bins:
        b = sub[(sub["mins_left"] >= lo) & (sub["mins_left"] < hi)]
        if len(b) < 15:
            continue
        ct = b["h_ct"].sum()
        rows.append((f"{lo}-{hi}min", len(b), 100 * b["h_won"].mean(),
                     100 * b["h_net"].sum() / max(ct, 1), b["h_net"].sum()))
    if not rows:
        continue
    best_wr = max(rows, key=lambda r: r[2])
    best_ev = max(rows, key=lambda r: r[3])
    print(f"  {name}: best WIN RATE  {best_wr[0]:<10} {best_wr[2]:.1f}%  (n={best_wr[1]})")
    print(f"  {name}: best EV/ct     {best_ev[0]:<10} {best_ev[3]:+.2f}c (n={best_ev[1]})")
    pos = [r for r in rows if r[3] > 0]
    print(f"  {name}: EV-positive buckets: "
          + (", ".join(f"{r[0]} ({r[3]:+.2f}c, n={r[1]})" for r in pos) if pos else "NONE"))
