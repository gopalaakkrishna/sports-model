"""Is there a cheaper contract to buy? Strike structure of 15m vs hourly.

The ask is locks at better odds -- i.e. cheaper entries -- not a narrow price
band. Two ways a cheaper entry can exist:

  A. STRIKE CHOICE: several strikes trade on the same window, so a further-out
     strike costs less than the at-the-money one. This is what the hourly ladder
     already exploits.
  B. TIMING: same contract, bought earlier/later at a different price.

Which is available depends on market structure, so measure it rather than
assume. Read-only.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

st = pd.read_csv(REPORTS / "reconcile_settlements.csv")
st["series"] = st["ticker"].astype(str).str.split("-").str[0]

print("=== ticker anatomy ===")
for s in ("KXBTC15M", "KXBTCD"):
    sub = st[st["series"] == s]["ticker"].astype(str)
    print(f"\n{s}: {len(sub):,} settlements")
    for t in sub.head(5):
        print(f"   {t}")
    # how many DISTINCT tickers share the same window key?
    if s == "KXBTC15M":
        # KXBTC15M-26AUG051845-45 -> window key is the datetime part
        key = sub.str.extract(r"^KXBTC15M-(\d{2}[A-Z]{3}\d{6})")[0]
    else:
        # KXBTCD-26AUG0519-T64599.99 -> window key is the date+hour
        key = sub.str.extract(r"^KXBTCD-(\d{2}[A-Z]{3}\d{4})")[0]
    grp = sub.groupby(key).nunique()
    print(f"   distinct windows: {key.nunique():,}")
    print(f"   distinct tickers per window: mean {grp.mean():.2f}, max {grp.max()}")
    multi = (grp > 1).mean()
    print(f"   windows with MORE THAN ONE strike traded: {multi:.1%}")
    if s == "KXBTCD":
        strikes = sub.str.extract(r"-T([\d.]+)$")[0].astype(float)
        print(f"   strike values present: {strikes.notna().sum():,} "
              f"(range {strikes.min():,.0f} .. {strikes.max():,.0f})")
    else:
        suf = sub.str.extract(r"-(\d+)$")[0]
        print(f"   suffix values seen: {sorted(suf.dropna().unique())[:8]}")
        print("   ^ if these are just 00/15/30/45 they are the WINDOW MINUTE,")
        print("     not a strike -> 15m is a single-strike market.")

print("\n" + "=" * 74)
print("VERDICT")
print("=" * 74)
k15 = st[st["series"] == "KXBTC15M"]["ticker"].astype(str)
suf15 = set(k15.str.extract(r"-(\d+)$")[0].dropna().unique())
if suf15 <= {"00", "15", "30", "45"}:
    print("  KXBTC15M: suffix is only " + ",".join(sorted(suf15)) + " -> the window minute.")
    print("  => 15m is a SINGLE-STRIKE market. There is no cheaper strike to pick.")
    print("     Better odds on 15m can only come from TIMING (option B), or from")
    print("     paying the bid instead of the ask (maker), not strike selection.")
else:
    print("  KXBTC15M suffixes vary beyond window minutes -- strike choice may exist.")
kd = st[st["series"] == "KXBTCD"]["ticker"].astype(str)
kdk = kd.str.extract(r"^KXBTCD-(\d{2}[A-Z]{3}\d{4})")[0]
print(f"\n  KXBTCD: {kd.groupby(kdk).nunique().mean():.2f} strikes traded per hour on average")
print("  => hourly IS a strike ladder. Cheaper strikes are selectable there,")
print("     which is what the hourly ladder panel already does.")
