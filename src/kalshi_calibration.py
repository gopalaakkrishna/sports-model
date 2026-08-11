"""Is any Kalshi market MIS-priced (soft), as opposed to merely uncertain?

Info gain is not the right test on its own: a perfectly-priced league of
coin-flips scores near zero because there is little to know, not because the
market is bad. Softness shows up as MIS-CALIBRATION — prices that are
systematically wrong in a direction you could bet against.

So: bucket every leg by its de-vigged price and compare to how often it
actually happened. A sharp market tracks the diagonal.
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"C:\Users\Gohan\OneDrive\Documents\Gopal\sports-model")
d = pd.read_parquet(ROOT / "data/raw/kalshi_closes.parquet")
d = d[d["result"].isin(["yes", "no"])].copy()

recs = []
for series, g in d.groupby("series"):
    for _, e in g.groupby("event_ticker"):
        if e["result"].eq("yes").sum() != 1 or len(e) < 2:
            continue
        s = e["close"].sum()
        if not (0.8 < s < 1.6):
            continue
        for _, r in e.iterrows():
            recs.append({"series": series, "p": r["close"] / s,
                         "won": 1 if r["result"] == "yes" else 0})
df = pd.DataFrame(recs)

print("POOLED — every market together (n is large enough to mean something)")
bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, 1.01]
print(f"  {'price band':<14}{'n':>6}{'said':>8}{'actual':>9}{'error':>9}")
for lo, hi in zip(bins, bins[1:]):
    b = df[(df["p"] >= lo) & (df["p"] < hi)]
    if len(b) < 20:
        continue
    said, act = b["p"].mean(), b["won"].mean()
    se = np.sqrt(max(act * (1 - act), 1e-9) / len(b))
    flag = "  <-- outside 2SE" if abs(act - said) > 2 * se else ""
    print(f"  {lo:.0%}-{hi:.0%}{'':<6}{len(b):>6}{said:>8.1%}{act:>9.1%}"
          f"{act-said:>+9.1%}{flag}")

print("\nFAVOURITE-LONGSHOT BIAS — the classic soft-market signature")
print(f"  {'market':<16}{'n':>6}{'favs said':>11}{'favs won':>10}{'error':>9}")
for series, g in df.groupby("series"):
    fav = g[g["p"] >= 0.55]
    if len(fav) < 15:
        continue
    said, act = fav["p"].mean(), fav["won"].mean()
    se = np.sqrt(max(act * (1 - act), 1e-9) / len(fav))
    sig = "yes" if abs(act - said) > 2 * se else "no"
    nm = series.replace("KX", "").replace("GAME", "").replace("MATCH", "")
    print(f"  {nm:<16}{len(fav):>6}{said:>11.1%}{act:>10.1%}{act-said:>+9.1%}"
          f"   significant: {sig}")

print("\n'significant' = the gap exceeds 2 standard errors. On these sample")
print("sizes almost nothing will be, and that is the finding, not a failure.")
