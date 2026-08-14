"""Score the simultaneous Kalshi/bookmaker snapshots once games settle.

The question fetch_live_pairs.py exists to answer: when Kalshi and DraftKings
disagree AT THE SAME INSTANT, who is right — and is the gap bigger than the
spread plus Kalshi's 0.07*p*(1-p) fee? The earlier retrospective test could
not answer this (its Kalshi price was 8 hours older than its book price);
these snapshots have no such asymmetry, so whatever they say is real.

Settlement comes from Kalshi itself: each snapshotted leg is a market whose
result field flips to yes/no after the game. Results are cached so a ticker
is only ever fetched once.

Verdict logic, in order of what would kill the idea (mirrors kalshi_vs_book):
  1. Do simultaneous prices actually diverge?
  2. When they diverge, whose probability was better (log loss)?
  3. Does following the sharper side across the spread+fee make money?

Run it any time: it reports sample size honestly and refuses conclusions
below n=150 legs, the size at which the retrospective test's CI first
excluded zero.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data" / "live_pairs.csv"
RESULTS = ROOT / "data" / "live_pairs_results.csv"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
EPS = 1e-12
MIN_LEGS = 150


def fetch_results(tickers: list[str]) -> pd.DataFrame:
    """ticker -> result, cached. Only settled markets enter the cache."""
    cache = (pd.read_csv(RESULTS, dtype=str) if RESULTS.exists()
             else pd.DataFrame(columns=["k_ticker", "result"]))
    known = set(cache["k_ticker"])
    new = []
    for tk in tickers:
        if tk in known:
            continue
        try:
            r = requests.get(f"{KALSHI}/markets/{tk}", timeout=30)
            if r.status_code != 200:
                continue
            m = r.json().get("market", {})
            res = str(m.get("result") or "").strip().lower()
            if res in ("yes", "no"):
                new.append({"k_ticker": tk, "result": res})
            time.sleep(0.1)
        except requests.RequestException:
            continue
    if new:
        cache = pd.concat([cache, pd.DataFrame(new)], ignore_index=True)
        cache.to_csv(RESULTS, index=False)
    if new:
        print(f"fetched {len(new)} newly settled legs")
    return cache


def ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def kalshi_fee(p: float) -> float:
    return 0.07 * p * (1 - p)


def main() -> int:
    if not PAIRS.exists():
        print("no snapshots yet — fetch_live_pairs.py has not run")
        return 0
    d = pd.read_csv(PAIRS)
    d["ts_utc"] = pd.to_datetime(d["ts_utc"], errors="coerce")
    d["start_utc"] = pd.to_datetime(d["start_utc"], errors="coerce")
    print(f"{len(d)} snapshots, {d['espn_id'].nunique()} games, "
          f"{d['ts_utc'].min()} .. {d['ts_utc'].max()}")

    ended = d[d["start_utc"] < pd.Timestamp.now("UTC").tz_localize(None)
              - pd.Timedelta(hours=5)]
    if ended.empty:
        print("nothing settled yet")
        return 0
    res = fetch_results(sorted(ended["k_ticker"].unique()))
    d = d.merge(res, on="k_ticker", how="inner")
    d["won"] = (d["result"] == "yes").astype(int)

    # One row per leg: the LAST pre-game snapshot, i.e. the price a trade
    # placed as late as possible would have gotten. Horizon buckets keep the
    # earlier snapshots useful without double-counting a leg in the headline.
    last = (d.sort_values("ts_utc").groupby("k_ticker", as_index=False).last())
    n = len(last)
    print(f"\nsettled legs with a final pre-game pair: {n}")
    if n < MIN_LEGS:
        print(f"below n={MIN_LEGS} — collecting, not concluding. "
              f"({MIN_LEGS - n} legs to go)")
        if n < 30:
            return 0

    last["gap"] = last["k_mid"] - last["book_prob"]
    print("\n1) DIVERGENCE (simultaneous, final snapshot)")
    print(f"   mean |gap| {last['gap'].abs().mean():.1%}   "
          f"median {last['gap'].abs().median():.1%}   "
          f">5pts {(last['gap'].abs() > .05).mean():.0%}   "
          f">10pts {(last['gap'].abs() > .10).mean():.0%}")

    print("\n2) WHO IS RIGHT (log loss, lower wins)")
    kl, bl = ll(last["k_mid"], last["won"]), ll(last["book_prob"], last["won"])
    print(f"   overall     kalshi {kl:.4f}   book {bl:.4f}   gap {kl - bl:+.4f}")
    big = last[last["gap"].abs() > .04]
    if len(big) >= 20:
        print(f"   |gap|>4pts  kalshi {ll(big['k_mid'], big['won']):.4f}   "
              f"book {ll(big['book_prob'], big['won']):.4f}   (n={len(big)})")
    if n >= 40:
        diff = (-(last["won"] * np.log(np.clip(last["k_mid"], EPS, 1)))
                - ((1 - last["won"]) * np.log(np.clip(1 - last["k_mid"], EPS, 1)))
                + (last["won"] * np.log(np.clip(last["book_prob"], EPS, 1)))
                + ((1 - last["won"]) * np.log(np.clip(1 - last["book_prob"], EPS, 1))))
        rng = np.random.default_rng(0)
        boot = [diff.sample(len(diff), replace=True,
                            random_state=int(rng.integers(1e9))).mean()
                for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"   95% CI on gap [{lo:+.4f}, {hi:+.4f}]  "
              f"-> {'book sharper' if lo > 0 else 'kalshi sharper' if hi < 0 else 'not distinguishable yet'}")

    print("\n3) DOES CROSSING THE SPREAD PAY (buy YES at ask when book says cheap)")
    for margin in (0.02, 0.04, 0.06):
        t = last[last["book_prob"] - last["k_ask"] > margin]
        if t.empty:
            print(f"   edge>{margin:.0%}: no trades")
            continue
        pnl = t["won"] - t["k_ask"] - t["k_ask"].map(kalshi_fee)
        print(f"   edge>{margin:.0%}: {len(t)} trades, {t['won'].mean():.0%} won, "
              f"total {pnl.sum():+.2f} units, roi {pnl.sum() / t['k_ask'].sum():+.1%}")

    if n < MIN_LEGS:
        print(f"\nAll of the above is provisional until n>={MIN_LEGS}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
