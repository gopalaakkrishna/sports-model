"""Is Kalshi SOFT in its thin markets, or just EXPENSIVE?

Two different things get conflated. A wide spread means a market is costly to
trade. A badly-priced market means it is beatable. They are not the same, and
only the second is an opportunity.

For each series:
  overround   sum of leg prices per event. 1.00 = no built-in cost. This is
              the toll you pay before being right about anything.
  log loss    of the de-vigged Kalshi price against the actual outcome.
  base rate   log loss of just predicting each outcome's frequency. If Kalshi
              barely beats this, its prices carry little information and there
              is room for a model. If it crushes it, there is not.
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"C:\Users\Gohan\OneDrive\Documents\Gopal\sports-model")
d = pd.read_parquet(ROOT / "data/raw/kalshi_closes.parquet")
d = d[d["result"].isin(["yes", "no"])].copy()
EPS = 1e-15

rows = []
for series, g in d.groupby("series"):
    ev = g.groupby("event_ticker")
    # Only events with a complete book and exactly one winner.
    good = [e for _, e in ev if e["result"].eq("yes").sum() == 1 and len(e) >= 2]
    if len(good) < 5:
        continue
    n_out = int(np.median([len(e) for e in good]))

    p, y, books = [], [], []
    for e in good:
        s = e["close"].sum()
        if not (0.8 < s < 1.6):
            continue
        books.append(s)
        for _, r in e.iterrows():
            p.append(r["close"] / s)          # de-vig
            y.append(1 if r["result"] == "yes" else 0)
    if len(p) < 10:
        continue
    p = np.clip(np.array(p), EPS, 1 - EPS)
    y = np.array(y)
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    base = float(y.mean())
    bl = float(-(y * np.log(base) + (1 - y) * np.log(1 - base)).mean())

    rows.append({
        "series": series.replace("KX", "").replace("GAME", "").replace("MATCH", ""),
        "events": len(books),
        "outcomes": n_out,
        "overround": float(np.mean(books)),
        "kalshi_ll": ll,
        "baserate_ll": bl,
        "info_gain": bl - ll,
    })

df = pd.DataFrame(rows).sort_values("info_gain")
print(f"{'market':<16}{'events':>7}{'ways':>6}{'overround':>11}{'kalshi LL':>11}"
      f"{'base LL':>10}{'info gain':>11}")
for _, r in df.iterrows():
    print(f"{r['series']:<16}{r['events']:>7}{r['outcomes']:>6}"
          f"{r['overround']:>10.3f}{r['kalshi_ll']:>11.4f}{r['baserate_ll']:>10.4f}"
          f"{r['info_gain']:>+11.4f}")

print("\ninfo gain = how much better Kalshi's price is than just guessing the")
print("base rate. LOW gain on a real sample = prices carry little information")
print("= room for a model. HIGH gain = the market already knows.")
print("\noverround is the toll: 1.05 means 5% is taken before anyone is right.")
