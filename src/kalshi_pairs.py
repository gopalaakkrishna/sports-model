"""The one place that decides which two teams a Kalshi game event is.

Kalshi moved per-market titles from "Milwaukee vs Tampa Bay Winner?" to the
per-outcome form ("Tampa Bay wins"). Every file that split a MARKET title on
" vs " then matched nothing. That broke, one file at a time:

  * kalshi_edge and settle (soccer prices and results)
  * mlb_predict (MLB picks had no bid/ask for about four weeks)
  * fetch_kalshi_mlb (2026-10-29: 507 settled games, zero rows, full refresh
    failed), with inplay_live and today.py silently pairing nothing

Each fix was local, so the next file broke the same way. The EVENTS endpoint
still carries the pairing in its title ("Game 3: Milwaukee vs Tampa Bay"), so
pairing lives here and nowhere else. tests/test_kalshi_pairs.py fails the
build if any src file splits a title on " vs " outside this module.

Order is Kalshi's: first team, then second. For MLB, NFL and WNBA that is
away vs home, the convention the callers already assumed.
"""

from __future__ import annotations

import re
import time

import requests

import guards as G

K = "https://api.elections.kalshi.com/trade-api/v2"

_GAME_PREFIX = re.compile(r"^\s*Game\s+\d+\s*:\s*", re.IGNORECASE)
_PAIR = re.compile(r"^\s*(.+?)\s+vs\.?\s+(.+?)\s*$")
# Doubleheaders end the title with "(Game 1)": "Boston vs New York Y (Game 1)".
_DOUBLEHEADER = re.compile(r"\s*\(Game\s+\d+\)\s*$", re.IGNORECASE)


def split_pair(title) -> tuple[str, str] | None:
    """'Game 3: Milwaukee vs Tampa Bay' -> ('Milwaukee', 'Tampa Bay').

    Also accepts the old market form ('A vs B Winner?') and a trailing
    ': descriptor'. Returns None for anything that is not a pairing, such as
    the per-outcome market title 'Tampa Bay wins'.
    """
    t = _GAME_PREFIX.sub("", str(title or ""))
    t = t.split(":", 1)[0].replace(" Winner?", "")
    m = _PAIR.match(_DOUBLEHEADER.sub("", t))
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def event_pairs(series: str, status: str = "open", *, session=None,
                timeout: float = 60, max_pages: int = 50,
                strict: bool = True) -> dict[str, tuple[str, str]]:
    """{event_ticker: (first, second)} from the EVENTS endpoint, all pages.

    strict (the default) raises guards.ParseYieldedNothing when Kalshi
    returned events and none of them parsed, with sample titles in the
    message, so the next format change fails loudly on day one instead of
    quietly emptying a board. An empty series stays silent.
    """
    get = (session or requests).get
    events, cursor = [], None
    for _ in range(max_pages):
        params = {"series_ticker": series, "status": status, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        r = get(f"{K}/events", params=params, timeout=timeout)
        if r.status_code != 200:
            break
        j = r.json()
        batch = j.get("events", []) or []
        events.extend(batch)
        cursor = j.get("cursor")
        if not cursor or not batch:
            break
        time.sleep(0.1)

    out = {}
    for e in events:
        pair = split_pair(e.get("title"))
        if pair:
            out[str(e.get("event_ticker"))] = pair
    if strict:
        G.parsed_or_die(events, out, what=f"{series} event pairs",
                        series=series,
                        samples=[e.get("title") for e in events])
    return out


def pair_for(event_ticker, pairs: dict, market_title=None):
    """The pairing for one event: the events map first, then an old-format
    market title (in case Kalshi ever reverts). None when neither has it."""
    return pairs.get(str(event_ticker)) or split_pair(market_title)
