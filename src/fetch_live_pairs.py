"""Capture Kalshi and bookmaker prices for the same game at the same moment.

WHY THIS EXISTS

kalshi_vs_book.py found bookmakers significantly sharper than Kalshi and a
naive +14.4% ROI buying Kalshi when it sat below the book — but that test
compared an ~8-hour-old Kalshi price against the CLOSING book line, so the
"edge" is partly just the market moving over those 8 hours. Lookahead.
The only way to answer the question cleanly is to sample BOTH prices at the
same instant, going forward, and settle them later. That is this module.

It is a measuring instrument, not a picker. Nothing here writes to the
ledger or the board; it appends snapshots to data/live_pairs.csv and
pair_analysis.py scores them once games settle. If the gap survives spread
and fees at simultaneous timestamps, THEN there is something to trade.

THE BOOK SIDE

ESPN's public site API carries DraftKings moneylines per event, keyless:

    site.api.espn.com  /apis/site/v2/sports/{sport}/{league}/scoreboard
    sports.core.api.espn.com  /v2/sports/{s}/leagues/{l}/events/{id}/
                              competitions/{id}/odds

DraftKings is not Pinnacle, but FINDINGS showed the sharper-than-Kalshi
conclusion held against the bookmaker AVERAGE too, and DK is the only line
available keyless at snapshot time. The scoreboard endpoint drops odds the
moment a game ends and often lacks them for far-future games — odds are
fetched per event, and only for events we are about to record anyway, which
keeps this to a few dozen requests per run.

CADENCE

Runs inside auto_update light mode (every ~15 min in CI). Hourly snapshots
per game, tightening to every run inside the final 2 hours where Kalshi's
book is deepest and a trade would actually happen. Majors only: this is
also the league set the user asked the whole system to narrow to.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from team_names import normalise

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
SITE = "https://site.api.espn.com/apis/site/v2/sports"
CORE = "https://sports.core.api.espn.com/v2/sports"

OUT = ROOT / "data" / "live_pairs.csv"

# Kalshi game-winner series -> (espn sport, espn league). Majors only.
SERIES_ESPN = {
    "KXMLBGAME": ("baseball", "mlb"),
    "KXWNBAGAME": ("basketball", "wnba"),
    "KXNFLGAME": ("football", "nfl"),
    "KXMLSGAME": ("soccer", "usa.1"),
    "KXLALIGAGAME": ("soccer", "esp.1"),
    "KXLIGAMXGAME": ("soccer", "mex.1"),
    "KXEPLGAME": ("soccer", "eng.1"),
    "KXSERIEAGAME": ("soccer", "ita.1"),
    "KXLIGUE1GAME": ("soccer", "fra.1"),
    "KXBUNDESLIGAGAME": ("soccer", "ger.1"),
}

MAX_HOURS_OUT = 30.0     # ignore games further out than this
THIN_MINUTES = 55.0      # one snapshot per game per hour...
DENSE_HOURS = 2.0        # ...except inside 2h of start: every run


def _now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)


def american_to_prob(ml) -> float | None:
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return None
    if ml == 0:
        return None
    return (-ml / (-ml + 100.0)) if ml < 0 else (100.0 / (ml + 100.0))


def espn_events(sport: str, league: str) -> list[dict]:
    """Pre-game events starting within MAX_HOURS_OUT."""
    now = _now()
    dates = f"{now:%Y%m%d}-{(now + pd.Timedelta(days=2)):%Y%m%d}"
    r = requests.get(f"{SITE}/{sport}/{league}/scoreboard",
                     params={"dates": dates}, timeout=45)
    if r.status_code != 200:
        return []
    out = []
    for e in r.json().get("events", []):
        try:
            if e["status"]["type"]["state"] != "pre":
                continue
            start = pd.Timestamp(e["date"]).tz_convert("UTC").tz_localize(None)
            hrs = (start - now).total_seconds() / 3600.0
            if not (-0.1 <= hrs <= MAX_HOURS_OUT):
                continue
            comp = e["competitions"][0]
            home = away = None
            for c in comp.get("competitors", []):
                nm = (c.get("team") or {}).get("displayName")
                if c.get("homeAway") == "home":
                    home = nm
                elif c.get("homeAway") == "away":
                    away = nm
            if home and away:
                out.append({"id": str(e["id"]), "start": start,
                            "hours": hrs, "home": home, "away": away})
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return out


def espn_odds(sport: str, league: str, event_id: str) -> dict | None:
    """DraftKings (or first available) moneylines for one event."""
    r = requests.get(f"{CORE}/{sport}/leagues/{league}/events/{event_id}"
                     f"/competitions/{event_id}/odds", timeout=45)
    if r.status_code != 200:
        return None
    items = r.json().get("items", [])
    items.sort(key=lambda it: (it.get("provider", {}).get("name") != "DraftKings"))
    for it in items:
        h = american_to_prob((it.get("homeTeamOdds") or {}).get("moneyLine"))
        a = american_to_prob((it.get("awayTeamOdds") or {}).get("moneyLine"))
        d = american_to_prob((it.get("drawOdds") or {}).get("moneyLine"))
        if h is None or a is None:
            continue
        return {"provider": it.get("provider", {}).get("name", "?"),
                "H": h, "A": a, "D": d}
    return None


def kalshi_events(series: str) -> dict[str, list[dict]]:
    """event_ticker -> open winner legs with two-sided quotes."""
    r = requests.get(f"{KALSHI}/markets",
                     params={"series_ticker": series, "status": "open",
                             "limit": 200}, timeout=60)
    if r.status_code != 200:
        return {}
    ev: dict[str, list[dict]] = {}
    for m in r.json().get("markets", []):
        try:
            bid = float(m.get("yes_bid_dollars"))
            ask = float(m.get("yes_ask_dollars"))
        except (TypeError, ValueError):
            continue
        if not (0.0 < bid <= ask < 1.0):
            continue
        leg = str(m.get("yes_sub_title") or "").strip()
        if not leg:
            continue
        ev.setdefault(str(m.get("event_ticker", "")), []).append(
            {"ticker": m["ticker"], "leg": leg, "bid": bid, "ask": ask})
    return ev


def _side_of(leg: str, g: dict) -> str | None:
    """Which side of an ESPN game a Kalshi leg name refers to, if exactly one."""
    nl = normalise(leg)
    if not nl:
        return None
    hit = []
    for side, name in (("H", g["home"]), ("A", g["away"])):
        nn = normalise(name)
        if nn.startswith(nl) or nl.startswith(nn) or nl in nn:
            hit.append(side)
    return hit[0] if len(hit) == 1 else None


def _teams_match(names: list[str], g: dict) -> bool:
    return {_side_of(n, g) for n in names} == {"H", "A"}


_MON = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def _ticker_stamp(ev: str):
    import re
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})(\d{4})", ev)
    if not m or m.group(2) not in _MON:
        return None
    return m


def _filter_by_ticker_date(ev: str, cands: list[dict]) -> list[dict]:
    """Keep only games on the ticker's (ET) date.

    Kalshi lists the same matchup across several days of a series; without
    this, tomorrow's SEAHOU event also matches tonight's SEA@HOU game and
    the pair collides away entirely.
    """
    m = _ticker_stamp(ev)
    if not m:
        return cands
    try:
        d = pd.Timestamp(year=2000 + int(m.group(1)), month=_MON[m.group(2)],
                         day=int(m.group(3))).date()
    except ValueError:
        return cands
    return [g for g in cands
            if (g["start"] - pd.Timedelta(hours=4)).date() == d]


def _pick_by_ticker_time(ev: str, cands: list[dict]) -> list[dict]:
    """Disambiguate a doubleheader using the HHMM (ET) inside the ticker."""
    m = _ticker_stamp(ev)
    if not m:
        return cands
    hh, mm = int(m.group(4)[:2]), int(m.group(4)[2:])
    scored = []
    for g in cands:
        # ET is UTC-4 in Aug, UTC-5 in winter; a 90-minute tolerance and
        # nearest-wins absorbs the DST difference for doubleheader gaps,
        # which are hours apart.
        et = g["start"] - pd.Timedelta(hours=4)
        diff = abs((et.hour * 60 + et.minute) - (hh * 60 + mm))
        scored.append((min(diff, 1440 - diff), g))
    scored.sort(key=lambda x: x[0])
    if scored[0][0] <= 90 and (len(scored) == 1 or scored[1][0] > scored[0][0]):
        return [scored[0][1]]
    return cands


def last_snapshots() -> dict[str, pd.Timestamp]:
    """espn_id -> most recent snapshot ts, for thinning."""
    if not OUT.exists():
        return {}
    try:
        d = pd.read_csv(OUT, usecols=["ts_utc", "espn_id"],
                        dtype={"espn_id": str})
    except (ValueError, OSError):
        return {}
    d["ts_utc"] = pd.to_datetime(d["ts_utc"], errors="coerce")
    return d.groupby("espn_id")["ts_utc"].max().to_dict()


def main() -> int:
    now = _now()
    seen = last_snapshots()
    rows, dropped_ambiguous, dropped_unmatched = [], 0, 0

    for series, (sport, league) in SERIES_ESPN.items():
        games = espn_events(sport, league)
        if not games:
            continue
        kev = kalshi_events(series)
        if not kev:
            continue

        # Kalshi event -> ESPN game. Kalshi legs are truncated city names
        # ("Los Angeles D"), ESPN uses full names ("Los Angeles Dodgers"),
        # so the join is prefix/containment on normalised names, per event.
        # MLB doubleheaders give two games with the same team pair; the
        # Kalshi ticker encodes the start time (KXMLBGAME-26AUG161920SEAHOU
        # = 19:20 ET), which picks the right one. Anything still ambiguous
        # is skipped outright — one wrong join would poison the whole
        # comparison this file exists to make.
        matched: dict[str, tuple[dict, str, list[dict]]] = {}
        for ev, legs in kev.items():
            names = [x["leg"] for x in legs
                     if x["leg"].lower() not in ("tie", "draw")]
            if len(names) != 2:
                dropped_unmatched += 1
                continue
            cands = [g for g in games if _teams_match(names, g)]
            cands = _filter_by_ticker_date(ev, cands)
            if len(cands) > 1:
                cands = _pick_by_ticker_time(ev, cands)
            if len(cands) != 1:
                dropped_ambiguous += len(cands) > 1
                dropped_unmatched += not cands
                continue
            g = cands[0]
            if g["id"] in matched:            # two Kalshi events, one game
                dropped_ambiguous += 1
                matched.pop(g["id"])
                continue
            matched[g["id"]] = (g, ev, legs)

        for gid, (g, ev, legs) in matched.items():
            # Thinning: hourly, but every run inside the dense window.
            last = seen.get(gid)
            if (g["hours"] > DENSE_HOURS and last is not None
                    and (now - last).total_seconds() < THIN_MINUTES * 60):
                continue

            odds = espn_odds(sport, league, gid)
            if odds is None:
                continue
            time.sleep(0.2)  # be polite to ESPN

            # De-vig across whichever legs the book actually quotes.
            has_draw = any(x["leg"].lower() in ("tie", "draw") for x in legs)
            parts = [odds["H"], odds["A"]] + (
                [odds["D"]] if (has_draw and odds["D"]) else [])
            s = sum(parts)
            if not (0.9 < s < 1.4):
                continue

            for x in legs:
                low = x["leg"].lower()
                if low in ("tie", "draw"):
                    side, prob = "D", odds["D"]
                    if not (has_draw and prob):
                        continue
                else:
                    side = _side_of(x["leg"], g)
                    if side is None:
                        continue
                    prob = odds[side]
                rows.append({
                    "ts_utc": now.isoformat(timespec="seconds"),
                    "series": series, "league": league,
                    "espn_id": g["id"], "k_event": ev,
                    "start_utc": g["start"].isoformat(timespec="seconds"),
                    "hours_to_start": round(g["hours"], 2),
                    "match": f"{g['away']} @ {g['home']}",
                    "side": side, "team": x["leg"],
                    "k_ticker": x["ticker"],
                    "k_bid": x["bid"], "k_ask": x["ask"],
                    "k_mid": round((x["bid"] + x["ask"]) / 2, 4),
                    "book": odds["provider"],
                    "book_prob": round(prob / s, 4),
                })

    if rows:
        df = pd.DataFrame(rows)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    print(f"live_pairs: {len(rows)} legs snapshotted "
          f"({len({r['espn_id'] for r in rows})} games); "
          f"skipped {dropped_ambiguous} ambiguous, "
          f"{dropped_unmatched} unmatched kalshi events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
