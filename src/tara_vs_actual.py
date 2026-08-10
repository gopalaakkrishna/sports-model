"""Do my real Kalshi trades follow Tara's calls -- and does following her pay?

Two datasets that have been conflated:
  * Kalshi settlements/fills = what I ACTUALLY traded (my discretion)
  * Tara's call log           = what Tara RECOMMENDED (betAmt=0 on every row,
                                so nothing was ever executed through the app)

Joins them on the market window so we can finally separate:
  A. windows where I traded AND Tara called the same direction
  B. windows where I traded AND Tara called the OPPOSITE direction
  C. windows where I traded and Tara sat out / had no call
  D. windows where Tara called and I did not trade at all

If A beats B and C, Tara adds value and should be followed more.
If it does not, the app is decoration and my own reads are the edge.

Read-only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TARA_LOG = Path(r"C:\Users\Gohan\Downloads\Tara\Log\tara-call-log-2026-08-06T03-07-23.json")

MON = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def close_time(tk):
    """Ticker -> window close in UTC. Kalshi stamps these in US Eastern."""
    if not isinstance(tk, str):
        return pd.NaT
    p = tk.split("-")
    if len(p) < 2:
        return pd.NaT
    body, naive = p[1], None
    m = re.match(r"^(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})$", body)
    if m:
        yy, mon, dd, hh, mi = m.groups()
        naive = pd.Timestamp(2000 + int(yy), MON[mon], int(dd), int(hh), int(mi))
    else:
        m = re.match(r"^(\d{2})([A-Z]{3})(\d{2})(\d{2})$", body)
        if m:
            yy, mon, dd, hh = m.groups()
            naive = pd.Timestamp(2000 + int(yy), MON[mon], int(dd), int(hh), 0)
    if naive is None:
        return pd.NaT
    try:
        return naive.tz_localize("America/New_York").tz_convert("UTC")
    except Exception:
        return pd.NaT


# ---------------------------------------------------------------- my trades
st = pd.read_csv(REPORTS / "reconcile_settlements.csv")
fl = pd.read_csv(REPORTS / "reconcile_fills.csv")
for c in ("yes_count_fp", "no_count_fp", "yes_total_cost_dollars",
          "no_total_cost_dollars", "fee_cost", "cost", "payout", "net"):
    st[c] = pd.to_numeric(st[c], errors="coerce").fillna(0.0)
st["series"] = st["ticker"].astype(str).str.split("-").str[0]
st["close"] = st["ticker"].map(close_time)

fl["created_time"] = pd.to_datetime(fl["created_time"], errors="coerce", utc=True)
fl["leg"] = fl["outcome_side"].astype(str).str.lower()
first = (fl.sort_values("created_time").groupby("market_ticker")
           .agg(entry=("created_time", "first"), first_leg=("leg", "first")))
me = st.merge(first, left_on="ticker", right_index=True, how="inner")
me = me[(me["cost"] > 0) & me["close"].notna()].copy()

# my direction = the side I entered first;  YES == betting UP
me["my_dir"] = np.where(me["first_leg"] == "yes", "UP", "DOWN")
won_yes = me["market_result"].astype(str).str.lower() == "yes"
use_yes = me["first_leg"] == "yes"
me["h_ct"] = np.where(use_yes, me["yes_count_fp"], me["no_count_fp"])
me["h_cost"] = np.where(use_yes, me["yes_total_cost_dollars"], me["no_total_cost_dollars"])
me["h_won"] = np.where(use_yes, won_yes, ~won_yes)
me["h_pay"] = np.where(me["h_won"], me["h_ct"], 0.0)
tot = (me["yes_count_fp"] + me["no_count_fp"]).replace(0, np.nan)
me["h_fee"] = me["fee_cost"] * (me["h_ct"] / tot).fillna(1.0)
me["h_net"] = me["h_pay"] - me["h_cost"] - me["h_fee"]
print(f"my traded markets with a parseable window: {len(me):,}")

# ------------------------------------------------------------------- Tara
raw = json.loads(TARA_LOG.read_text(encoding="utf-8"))
t = pd.DataFrame(raw["entries"])
t["wclose"] = pd.to_datetime(
    t["windowId"].astype(str).str.replace(r"^\d+m-", "", regex=True),
    errors="coerce", utc=True)
t = t[t["wclose"].notna()].copy()
# Tara's windowId is the window OPEN; 15m windows close 15 min later.
t["wtype"] = t["windowId"].astype(str).str.extract(r"^(\d+)m-")[0].astype(float)
t["close"] = t["wclose"] + pd.to_timedelta(t["wtype"].fillna(15), unit="m")
t["tara_dir"] = t["dir"].astype(str).str.upper()
print(f"tara logged calls with a parseable window: {len(t):,}")
print(f"  tara direction mix: {t['tara_dir'].value_counts().head(4).to_dict()}")

# --------------------------------------------------------------- the join
j = me.merge(t[["close", "tara_dir", "confidence", "tier", "result"]].rename(
    columns={"result": "tara_result"}), on="close", how="left")
j = j.drop_duplicates(subset=["ticker"])
matched = j["tara_dir"].notna()
print(f"\nmy trades matched to a Tara call by window: {matched.sum():,} of {len(j):,}"
      f"  ({matched.mean():.1%})")


def blk(sub, label):
    n = len(sub)
    if n < 15:
        print(f"  {label:<44} n={n:>4}  (too few)")
        return
    ct = sub["h_ct"].sum()
    print(f"  {label:<44} n={n:>4}  WR={100*sub['h_won'].mean():>5.1f}%"
          f"  EV/ct={100*sub['h_net'].sum()/max(ct,1):>+7.2f}c"
          f"  total={sub['h_net'].sum():>+10,.2f}")


print("\n" + "=" * 84)
print("DID FOLLOWING TARA PAY?  (held basis, net of fees)")
print("=" * 84)
mj = j[matched]
agree = mj[mj["my_dir"] == mj["tara_dir"]]
oppose = mj[(mj["tara_dir"].isin(["UP", "DOWN"])) & (mj["my_dir"] != mj["tara_dir"])]
sat = mj[~mj["tara_dir"].isin(["UP", "DOWN"])]
blk(j[~matched], "no Tara call for that window")
blk(sat, "Tara SAT OUT, I traded anyway")
blk(oppose, "Tara said OPPOSITE, I traded my way")
blk(agree, "Tara AGREED with my direction")

print("\n" + "=" * 84)
print("BY SERIES")
print("=" * 84)
for s in ("KXBTC15M", "KXBTCD"):
    print(f"\n{s}:")
    ss = j[j["series"] == s]
    sm = ss[ss["tara_dir"].notna()]
    blk(ss[ss["tara_dir"].isna()], "  no Tara call")
    blk(sm[~sm["tara_dir"].isin(["UP", "DOWN"])], "  Tara sat out")
    blk(sm[(sm["tara_dir"].isin(["UP", "DOWN"])) & (sm["my_dir"] != sm["tara_dir"])], "  opposed Tara")
    blk(sm[sm["my_dir"] == sm["tara_dir"]], "  agreed with Tara")

print("\n" + "=" * 84)
print("COVERAGE: how much of Tara's output do I actually act on?")
print("=" * 84)
tc = t[t["tara_dir"].isin(["UP", "DOWN"])]
traded_closes = set(me["close"].dropna())
acted = tc["close"].isin(traded_closes)
print(f"  Tara directional calls          : {len(tc):,}")
print(f"  ...that I traded that window    : {acted.sum():,} ({acted.mean():.1%})")
print(f"  ...I ignored                    : {(~acted).sum():,}")
print("\n  NOTE: matching is by WINDOW, not by fill. Trading the same window Tara")
print("  called is not proof I acted on her call, and the strike I chose may")
print("  differ from hers. Treat these as upper bounds on agreement.")
