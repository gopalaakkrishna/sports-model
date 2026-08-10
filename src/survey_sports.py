"""Which sports on Kalshi are actually worth modelling?

Coverage should follow liquidity, not enthusiasm. A perfect cricket model is
worthless if nobody quotes cricket. This surveys every sports series, finds the
game-level ones, and measures real orderbook depth — so the build order is set
by where money can actually be traded.

Reminder from earlier in this project: Kalshi's `volume` and `liquidity_dollars`
fields read as empty on the current API. Depth must come from the orderbook, and
the live price fields are the `*_dollars` variants. Reading the legacy fields
makes the entire exchange look dead.
"""

from __future__ import annotations

import concurrent.futures as cf
import re
from collections import defaultdict

import requests

K = "https://api.elections.kalshi.com/trade-api/v2"

# Map ticker fragments to a sport. Order matters — first match wins.
SPORT_PATTERNS = [
    ("baseball", r"MLB|BASEBALL|NCAABB|WBC"),
    ("soccer", r"MLS|LIGAMX|EPL|LALIGA|SERIEA|BUNDES|LIGUE|UCL|LEAGUESCUP|SOCCER|WC[A-Z]|UEFA"),
    ("basketball", r"NBA|WNBA|NCAABB?K|BASKETBALL|EUROLEAGUE"),
    ("football", r"NFL|NCAAF|CFB|SUPERBOWL"),
    ("hockey", r"NHL|HOCKEY"),
    ("tennis", r"TENNIS|ATP|WTA|USOPEN|WIMBLEDON"),
    ("golf", r"GOLF|PGA|MASTERS"),
    ("motorsport", r"F1|NASCAR|MOTOGP|INDYCAR|RACE"),
    ("cricket", r"CRICKET|IPL|T20|ASHES"),
    ("mma_boxing", r"UFC|MMA|BOXING"),
    ("esports", r"LOL|CSGO|DOTA|VALORANT|ESPORT"),
]


def classify(ticker: str) -> str:
    t = ticker.upper()
    for sport, pat in SPORT_PATTERNS:
        if re.search(pat, t):
            return sport
    return "other"


def depth(ticker: str, within: float = 0.05) -> float:
    try:
        r = requests.get(f"{K}/markets/{ticker}/orderbook",
                         params={"depth": 6}, timeout=25)
        if r.status_code != 200:
            return 0.0
        book = r.json().get("orderbook") or r.json().get("orderbook_fp") or {}
    except requests.RequestException:
        return 0.0
    tot = 0.0
    for side in ("yes", "yes_dollars", "no", "no_dollars"):
        lv = book.get(side) or []
        px = []
        for l in lv:
            try:
                px.append(float(l[0]))
            except (TypeError, ValueError, IndexError):
                pass
        if not px:
            continue
        best = max(px)
        for l in lv:
            try:
                p, s = float(l[0]), float(l[1])
            except (TypeError, ValueError, IndexError):
                continue
            if abs(p - best) <= within:
                tot += p * s
    return tot


def series_probe(s: dict) -> dict | None:
    tick = s.get("ticker", "")
    try:
        r = requests.get(f"{K}/markets",
                         params={"series_ticker": tick, "status": "open",
                                 "limit": 40}, timeout=30)
        if r.status_code != 200:
            return None
        ms = r.json().get("markets", [])
    except requests.RequestException:
        return None
    if not ms:
        return None
    quoted = []
    for m in ms:
        try:
            b = float(m.get("yes_bid_dollars"))
            a = float(m.get("yes_ask_dollars"))
        except (TypeError, ValueError):
            continue
        if a - b <= 0.06 and a > 0:
            quoted.append(m)
    if not quoted:
        return {"ticker": tick, "title": s.get("title"), "open": len(ms),
                "quoted": 0, "depth": 0.0}
    # Sample depth on a few of the tightest markets.
    sample = sorted(quoted, key=lambda m: float(m["yes_ask_dollars"]) -
                    float(m["yes_bid_dollars"]))[:3]
    d = sum(depth(m["ticker"]) for m in sample) / len(sample)
    return {"ticker": tick, "title": s.get("title"), "open": len(ms),
            "quoted": len(quoted), "depth": d}


def main():
    r = requests.get(f"{K}/series", params={"category": "Sports"}, timeout=60)
    series = r.json().get("series", [])
    print(f"{len(series)} sports series on Kalshi")

    # Game-level series are the ones a match model can price.
    game = [s for s in series if "GAME" in str(s.get("ticker", "")).upper()]
    print(f"{len(game)} look like game-level markets\n")

    by_sport = defaultdict(list)
    for s in game:
        by_sport[classify(s.get("ticker", ""))].append(s)
    print("game-level series by sport:")
    for sp, lst in sorted(by_sport.items(), key=lambda z: -len(z[1])):
        print(f"  {sp:<14}{len(lst):>4}")

    print("\nprobing for live liquidity (this takes a minute)...")
    results = []
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(series_probe, s): s for s in game}
        for f in cf.as_completed(futs):
            out = f.result()
            if out and out["quoted"] > 0:
                out["sport"] = classify(out["ticker"])
                results.append(out)

    if not results:
        print("no game-level series currently quoted")
        return

    print(f"\n{len(results)} series with live two-sided quotes\n")
    agg = defaultdict(lambda: {"series": 0, "quoted": 0, "depth": 0.0})
    for r_ in results:
        a = agg[r_["sport"]]
        a["series"] += 1
        a["quoted"] += r_["quoted"]
        a["depth"] += r_["depth"]
    print(f"  {'sport':<14}{'series':>8}{'quoted mkts':>13}{'mean depth':>13}")
    for sp, a in sorted(agg.items(), key=lambda z: -z[1]["quoted"]):
        md = a["depth"] / max(a["series"], 1)
        print(f"  {sp:<14}{a['series']:>8}{a['quoted']:>13}{md:>12,.0f}$")

    print(f"\n  most liquid individual series:")
    for r_ in sorted(results, key=lambda z: -z["depth"])[:14]:
        print(f"    {r_['ticker']:<26}{str(r_['title'])[:30]:<32}"
              f"quoted={r_['quoted']:<4} depth=${r_['depth']:,.0f}")


if __name__ == "__main__":
    main()
