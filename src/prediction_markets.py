"""Read Kalshi and Polymarket prices, with liquidity as a first-class filter.

Both platforms expose public read APIs — no credentials needed to pull prices.
(Placing trades needs an authenticated account and is done by you, not by this
code.)

The critical lesson from probing these venues: **a price is not an opportunity
unless you can trade on it.** Many soccer markets sit at 0.50/0.50 with $4 of
liquidity and a bid/ask spread of 0.99. Against a model saying 70% that shows as
a colossal "edge", but the only price you can actually buy at is the ask, and
the ask is 0.99. The edge is an artefact of an empty order book.

So everything here reports `spread`, `liquidity` and `volume` alongside price,
and the scanner refuses to flag a market that fails the tradeability screen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import requests

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"

# A market must clear all of these before it is worth comparing to a model.
MIN_LIQUIDITY = 500.0   # USD resting in the book
MAX_SPREAD = 0.06       # 6 cents between bid and ask
MIN_VOLUME = 250.0      # USD traded


@dataclass
class MarketQuote:
    venue: str
    event: str
    question: str
    price: float | None      # mid / last
    bid: float | None
    ask: float | None
    spread: float | None
    liquidity: float
    volume: float
    tradeable: bool
    reason: str


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def screen(bid, ask, liquidity, volume) -> tuple[bool, str]:
    """Can this realistically be traded? Returns (ok, reason-if-not)."""
    if bid is None or ask is None:
        return False, "no two-sided quote"
    spread = ask - bid
    if spread > MAX_SPREAD:
        return False, f"spread {spread:.2f} > {MAX_SPREAD}"
    if liquidity < MIN_LIQUIDITY:
        return False, f"liquidity ${liquidity:,.0f} < ${MIN_LIQUIDITY:,.0f}"
    if volume < MIN_VOLUME:
        return False, f"volume ${volume:,.0f} < ${MIN_VOLUME:,.0f}"
    return True, ""


def polymarket_soccer(limit: int = 200) -> list[MarketQuote]:
    out = []
    r = requests.get(f"{GAMMA}/events",
                     params={"closed": "false", "limit": limit, "tag_slug": "soccer"},
                     timeout=60)
    r.raise_for_status()
    for e in r.json():
        title = str(e.get("title", ""))
        for m in (e.get("markets") or []):
            q = str(m.get("question", ""))
            if "win on" not in q and "end in a draw" not in q:
                continue
            try:
                prices = json.loads(m.get("outcomePrices") or "[]")
                price = float(prices[0]) if prices else None
            except Exception:
                price = None
            bid = m.get("bestBid")
            ask = m.get("bestAsk")
            bid = _f(bid, None) if bid is not None else None
            ask = _f(ask, None) if ask is not None else None
            liq = _f(m.get("liquidityNum") or m.get("liquidity"))
            vol = _f(m.get("volumeNum") or m.get("volume"))
            ok, why = screen(bid, ask, liq, vol)
            out.append(MarketQuote(
                venue="polymarket", event=title, question=q, price=price,
                bid=bid, ask=ask,
                spread=(ask - bid) if (bid is not None and ask is not None) else None,
                liquidity=liq, volume=vol, tradeable=ok, reason=why))
    return out


def kalshi_orderbook_depth(ticker: str, within: float = 0.05) -> float:
    """Dollar depth resting within `within` of the best price, both sides.

    Volume alone is misleading on a freshly opened market: it can be zero while
    the book already holds thousands of contracts. Depth is what you can
    actually trade against.
    """
    try:
        r = requests.get(f"{KALSHI}/markets/{ticker}/orderbook",
                         params={"depth": 10}, timeout=45)
        if r.status_code != 200:
            return 0.0
        j = r.json()
        book = j.get("orderbook") or j.get("orderbook_fp") or {}
    except requests.RequestException:
        return 0.0

    total = 0.0
    for side in ("yes", "yes_dollars", "no", "no_dollars"):
        levels = book.get(side) or []
        if not levels:
            continue
        prices = [_f(l[0]) for l in levels if len(l) >= 2]
        if not prices:
            continue
        best = max(prices)
        for lvl in levels:
            if len(lvl) < 2:
                continue
            px, size = _f(lvl[0]), _f(lvl[1])
            if abs(px - best) <= within:
                total += px * size
    return total


def kalshi_markets(series_filter: list[str], max_series: int = 30,
                   with_book: bool = True) -> list[MarketQuote]:
    """Open Kalshi markets for series whose ticker matches any filter term.

    NOTE ON FIELD NAMES: Kalshi migrated to dollar-denominated fields
    (`yes_bid_dollars`, `volume_fp`, `open_interest_fp`). The older `yes_bid` /
    `volume` keys still appear in responses but come back empty, so reading them
    makes every market look dead. That mistake would rule out venues that are in
    fact perfectly tradeable.
    """
    out = []
    r = requests.get(f"{KALSHI}/series", params={"category": "Sports"}, timeout=60)
    r.raise_for_status()
    series = r.json().get("series", [])
    targets = [s for s in series
               if any(k in str(s.get("ticker", "")).upper() for k in series_filter)]

    for s in targets[:max_series]:
        rr = requests.get(f"{KALSHI}/markets",
                          params={"series_ticker": s.get("ticker"),
                                  "status": "open", "limit": 100}, timeout=60)
        if rr.status_code != 200:
            continue
        for m in rr.json().get("markets", []):
            bid = m.get("yes_bid_dollars")
            ask = m.get("yes_ask_dollars")
            bid = _f(bid, None) if bid not in (None, "") else None
            ask = _f(ask, None) if ask not in (None, "") else None
            vol = _f(m.get("volume_fp"))
            oi = _f(m.get("open_interest_fp"))
            liq = _f(m.get("liquidity_dollars"))

            depth = kalshi_orderbook_depth(m["ticker"]) if with_book else 0.0
            # Treat resting depth as the liquidity measure, and let depth stand
            # in for volume on newly opened markets that have not traded yet.
            eff_liq = max(liq, depth, oi)
            eff_vol = max(vol, depth)
            ok, why = screen(bid, ask, eff_liq, eff_vol)
            price = (bid + ask) / 2 if (bid is not None and ask is not None) else None
            out.append(MarketQuote(
                venue="kalshi", event=str(m.get("event_ticker", "")),
                question=f"{m.get('title', '')} [{m.get('yes_sub_title', '')}]",
                price=price, bid=bid, ask=ask,
                spread=(ask - bid) if (bid is not None and ask is not None) else None,
                liquidity=eff_liq, volume=eff_vol, tradeable=ok, reason=why))
    return out


def kalshi_soccer(max_series: int = 25) -> list[MarketQuote]:
    return kalshi_markets(["MLSGAME", "LIGAMXGAME", "EPLGAME", "LALIGAGAME",
                           "SERIEAGAME", "BUNDESLIGAGAME", "LIGUE1GAME",
                           "LEAGUESCUPGAME", "UCLGAME"], max_series)


def kalshi_mlb(max_series: int = 6) -> list[MarketQuote]:
    return kalshi_markets(["KXMLBGAME"], max_series)


def summarise(quotes: list[MarketQuote]) -> None:
    ok = [q for q in quotes if q.tradeable]
    print(f"  total markets      {len(quotes)}")
    print(f"  pass tradeability  {len(ok)}")
    if not quotes:
        return
    reasons: dict[str, int] = {}
    for q in quotes:
        if not q.tradeable:
            key = q.reason.split(" ")[0] + " " + q.reason.split(" ")[1] if " " in q.reason else q.reason
            reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        print("  rejected because:")
        for k, v in sorted(reasons.items(), key=lambda z: -z[1]):
            print(f"    {v:>4}  {k}")
    for q in ok[:15]:
        print(f"    [{q.venue}] {q.question[:50]:<52} "
              f"bid={q.bid} ask={q.ask} liq=${q.liquidity:,.0f} vol=${q.volume:,.0f}")


if __name__ == "__main__":
    print("POLYMARKET (soccer)")
    summarise(polymarket_soccer())
    print("\nKALSHI (MLB game winners)")
    summarise(kalshi_mlb())
    print("\nKALSHI (soccer game winners)")
    summarise(kalshi_soccer())
    print("\nNote: reading prices needs no credentials. Placing trades does —")
    print("and that stays with you; this code never places an order.")
