"""Recover real kick-off times for fixtures whose source file only had a date.

Several producers emit a date and no clock — the WNBA and NFL exports, and every
ledger row, whose event string is "Home v Away (YYYY-MM-DD)". Rendered in the app
those all collapse to "all day", which is useless for a board whose whole point
is what is on next.

The times exist, just not in those files:

  * Kalshi carries `occurrence_datetime` on each market (UTC).
  * MLB StatsAPI carries `gameDate` (UTC) on the schedule.

So this builds a lookup keyed by (date, both team names) and offers a fuzzy
match, because "Los Angeles Angels @ Baltimore Orioles" and Kalshi's
"Angels vs Orioles" are the same fixture spelled two ways.

Everything here is best-effort by design. A network failure or an unmatched
fixture returns None and the caller keeps the date-only value — a missing clock
is a cosmetic gap, but a WRONG clock would put a fixture in the wrong day bucket
and mark a live game as upcoming.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

ET = ZoneInfo("America/New_York")

K = "https://api.elections.kalshi.com/trade-api/v2"
STATS = "https://statsapi.mlb.com/api/v1"

# Words that carry no identifying information once you are already comparing
# two teams within the same fixture and date.
_STOP = {"fc", "sc", "cf", "club", "city", "united", "the", "of", "and",
         "team", "los", "las", "san", "new", "de", "real"}


# The NFL upcoming file carries 2-3 letter team codes; Kalshi's NFL titles
# carry the city name. No token from "SEA" ever appears in "Seattle", so
# matching needs this fixed translation rather than fuzzy overlap.
_NFL_CODE = {
    "ARI": "arizona", "ATL": "atlanta", "BAL": "baltimore", "BUF": "buffalo",
    "CAR": "carolina", "CHI": "chicago", "CIN": "cincinnati", "CLE": "cleveland",
    "DAL": "dallas", "DEN": "denver", "DET": "detroit", "GB": "green bay",
    "HOU": "houston", "IND": "indianapolis", "JAX": "jacksonville",
    "KC": "kansas city", "LA": "los angeles", "LAC": "los angeles",
    "LAR": "los angeles", "LV": "las vegas", "MIA": "miami", "MIN": "minnesota",
    "NE": "new england", "NO": "new orleans", "NYG": "new york",
    "NYJ": "new york", "PHI": "philadelphia", "PIT": "pittsburgh",
    "SEA": "seattle", "SF": "san francisco", "TB": "tampa bay",
    "TEN": "tennessee", "WAS": "washington",
}


def _tokens(name: str) -> set[str]:
    expanded = _NFL_CODE.get(str(name).strip().upper())
    if expanded is not None:
        name = expanded
    t = re.sub(r"[^a-z0-9 ]+", " ", str(name).lower())
    return {w for w in t.split() if len(w) > 2 and w not in _STOP}


def _score(a_tok: set[str], b_tok: set[str]) -> int:
    return len(a_tok & b_tok)


class StartTimes:
    """date (YYYY-MM-DD, UTC) -> list of (tokensA, tokensB, datetime)."""

    def __init__(self) -> None:
        self._by_date: dict[str, list[tuple[set[str], set[str], datetime]]] = {}
        self.stats = {"kalshi_markets": 0, "mlb_games": 0, "errors": 0}

    def _add(self, when: datetime, a: str, b: str) -> None:
        """Index under the fixture's EASTERN date, and only that date.

        An earlier version indexed the UTC date plus one day either side, to
        absorb the fact that a 9:40pm ET first pitch is 01:40 UTC the following
        day. That "fix" was far worse than the problem: teams play series, so
        spreading each game across three buckets put Monday, Tuesday and
        Wednesday's Detroit-Seattle games into the same bucket, and the token
        match — which cannot tell them apart, the names being identical —
        returned whichever landed first. Detroit @ Seattle on the 6th resolved
        to the 4th.

        Eastern is the right key because it is what the event strings use:
        ledger events are written "(YYYY-MM-DD)" from the ET date, so an exact
        ET match is exact by construction. No neighbouring days, no spread.
        """
        if when is None:
            return
        key = when.astimezone(ET).strftime("%Y-%m-%d")
        self._by_date.setdefault(key, []).append((_tokens(a), _tokens(b), when))

    def load_kalshi(self, series: list[str]) -> None:
        for s in series:
            for attempt in range(4):
                try:
                    r = requests.get(f"{K}/markets",
                                     params={"series_ticker": s, "status": "open",
                                             "limit": 200}, timeout=30)
                    if r.status_code == 429:
                        time.sleep(2 ** attempt)
                        continue
                    r.raise_for_status()
                    for m in r.json().get("markets", []):
                        t = m.get("occurrence_datetime") or m.get("expected_expiration_time")
                        if not t:
                            continue
                        title = str(m.get("title", ""))
                        # Two title shapes seen across series:
                        #   "Home vs Away Winner?"                    (soccer/MLS)
                        #   "Will X win the A vs B Pro Football game?" (NFL)
                        # The NFL pattern must be tried first and anchored on
                        # "the " — without that anchor a non-greedy match starting
                        # at position 0 swallows "Will Seattle win the Dallas" as
                        # team A, since re.search prefers the leftmost start.
                        vs_match = (re.search(r"\bthe\s+(.+?)\s+vs\.?\s+(.+?)\s+"
                                              r"(?:Pro Football|Football)\s+game",
                                              title)
                                   or re.search(r"^(.+?)\s+vs\.?\s+(.+?)"
                                               r"(?:\s+Winner)?\s*\??$", title))
                        if not vs_match:
                            continue
                        a, b = vs_match.group(1).strip(), vs_match.group(2).strip()
                        try:
                            when = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
                        except ValueError:
                            continue
                        self._add(when, a, b)
                        self.stats["kalshi_markets"] += 1
                    break
                except requests.RequestException:
                    self.stats["errors"] += 1
                    time.sleep(1.5 * (attempt + 1))

    def load_mlb(self, start: str, end: str) -> None:
        try:
            r = requests.get(f"{STATS}/schedule",
                             params={"sportId": 1, "startDate": start,
                                     "endDate": end, "hydrate": "team"}, timeout=45)
            r.raise_for_status()
        except requests.RequestException:
            self.stats["errors"] += 1
            return
        for d in r.json().get("dates", []):
            for g in d.get("games", []):
                t = g.get("gameDate")
                if not t:
                    continue
                try:
                    when = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
                except ValueError:
                    continue
                self._add(when, g["teams"]["home"]["team"]["name"],
                          g["teams"]["away"]["team"]["name"])
                self.stats["mlb_games"] += 1

    def lookup(self, team_a: str, team_b: str, date: str | None) -> datetime | None:
        """Best match on the given date, or None.

        Requires BOTH sides to match on at least one distinctive token. A single
        shared token is how "New York Yankees" quietly matches "New York Mets";
        demanding both ends match makes that impossible within a fixture.
        """
        if not date:
            return None
        ta, tb = _tokens(team_a), _tokens(team_b)
        if not ta or not tb:
            return None
        cands = self._by_date.get(str(date)[:10])
        if not cands:
            # Nothing on the stated date. A late game can sit on the next ET
            # day while its source file records the venue-local date — a WNBA
            # tip at 22:00 PT is 01:00 ET tomorrow. Widening to the neighbours
            # is safe ONLY because the ambiguity guard below still applies: a
            # series would put several same-named fixtures in the widened pool
            # and be rejected, which is exactly the collision that made
            # Detroit @ Seattle resolve two days early.
            cands = []
            base = datetime.strptime(str(date)[:10], "%Y-%m-%d")
            for delta in (1, -1):
                alt = (base + timedelta(days=delta)).strftime("%Y-%m-%d")
                cands += self._by_date.get(alt, [])
            if not cands:
                return None
        best, best_score, ties = None, 0, 0
        for ca, cb, when in cands:
            fwd = min(_score(ta, ca), _score(tb, cb))
            rev = min(_score(ta, cb), _score(tb, ca))
            sc = max(fwd, rev)
            if sc > best_score:
                best, best_score, ties = when, sc, 1
            elif sc == best_score and sc > 0 and when != best:
                ties += 1
        # Two fixtures scoring identically on the same day means the names
        # cannot separate them — a doubleheader, or a series game that slipped
        # into the wrong bucket. Guessing here is how Detroit @ Seattle on the
        # 6th became the 4th. Return nothing and keep the date-only value.
        if ties > 1:
            self.stats["ambiguous"] = self.stats.get("ambiguous", 0) + 1
            return None
        return best if best_score >= 1 else None
