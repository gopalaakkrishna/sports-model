"""Settle open ledger predictions automatically from the source data.

Manual settlement does not scale past a handful of picks, and hand-entering
results is exactly where a tracking record quietly acquires a favourable bias.
This looks each open prediction up in the underlying data and settles only on an
unambiguous match.

Deliberately conservative: anything it cannot resolve with confidence is left
open and reported, rather than guessed. A ledger with a few unsettled rows is
far better than one settled wrongly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
import data as D
from team_names import TeamResolver

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "processed" / "ledger.jsonl"
STATS = "https://statsapi.mlb.com/api/v1"


def load() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def save(rows: list[dict]) -> None:
    LEDGER.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def parse_event(event: str) -> tuple[str, str, str | None]:
    """'Home v Away (2026-08-05)' -> (home, away, date)."""
    date = None
    m = re.search(r"\((\d{4}-\d{2}-\d{2})\)\s*$", event)
    if m:
        date = m.group(1)
        event = event[:m.start()].strip()
    for sep in (" v ", " vs ", " @ "):
        if sep in event:
            a, b = event.split(sep, 1)
            if sep == " @ ":       # "Away @ Home"
                return b.strip(), a.strip(), date
            return a.strip(), b.strip(), date
    return event.strip(), "", date


def settle_soccer(rec: dict, hist: pd.DataFrame) -> tuple[str, str] | None:
    home, away, date = parse_event(rec["event"])
    if not away:
        return None
    played = hist[hist["FTHG"].notna()]
    teams = sorted(set(played["HomeTeam"].dropna()) | set(played["AwayTeam"].dropna()))
    r = TeamResolver(teams)
    h, a = r.resolve(home), r.resolve(away)
    if h is None or a is None:
        return None

    cand = played[(played["HomeTeam"] == h) & (played["AwayTeam"] == a)]
    if date:
        d0 = pd.Timestamp(date)
        cand = cand[(cand["Date"] >= d0 - pd.Timedelta(days=2))
                    & (cand["Date"] <= d0 + pd.Timedelta(days=2))]
    else:
        # No date given: only accept a match in the last three weeks.
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=21)
        cand = cand[cand["Date"] >= cutoff]
    if len(cand) != 1:
        return None
    g = cand.iloc[0]
    outcome = {"H": "HOME", "D": "DRAW", "A": "AWAY"}.get(str(g["FTR"]))
    if outcome is None:
        return None
    return outcome, f"{g['HomeTeam']} {g['FTHG']:.0f}-{g['FTAG']:.0f} {g['AwayTeam']}"


def settle_mlb(rec: dict) -> tuple[str, str] | None:
    home, away, date = parse_event(rec["event"])
    if not away or not date:
        return None
    try:
        r = requests.get(f"{STATS}/schedule",
                         params={"sportId": 1, "startDate": date, "endDate": date,
                                 "hydrate": "team,linescore"}, timeout=45)
        r.raise_for_status()
    except requests.RequestException:
        return None
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            t = g["teams"]
            hn, an = t["home"]["team"]["name"], t["away"]["team"]["name"]
            if home not in hn and hn not in home:
                continue
            if away not in an and an not in away:
                continue
            if (g.get("status") or {}).get("detailedState") != "Final":
                return None
            hs, as_ = t["home"].get("score"), t["away"].get("score")
            if hs is None or as_ is None:
                return None
            return ("HOME" if hs > as_ else "AWAY"), f"{an} {as_} @ {hn} {hs}"
    return None


K = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi settled titles come in two shapes across every sport we track. The
# winning leg is the one with result == "yes", and it names the winner:
#
#   "Will Carolina win the Carolina vs Arizona Pro Football game?"
#   "Los Angeles vs Minnesota women's Pro Basketball game: Los Angeles wins?"
#   "London Spirit vs MI London men's cricket match: MI London wins"
#
# The "Will X win the A vs B" form must be tried FIRST and anchored on "the ",
# or a leftmost non-greedy match reads "Will Carolina" as side A.
_KT_WILL = re.compile(r"^Will\s+(.+?)\s+win\s+the\s+(.+?)\s+vs\.?\s+(.+?)\s+"
                      r"(?:Pro Football|Football|Pro Basketball|cricket)", re.I)
# Anchored on the men's/women's token that every one of these titles carries.
# Without that anchor the non-greedy second group stops at the first word it
# can: "London Spirit vs MI London men's cricket match: MI London wins" parsed
# side B as "MI", which then failed to match the fixture and never settled.
_KT_COLON = re.compile(r"^(.+?)\s+vs\.?\s+(.+?)\s+(?:men's|women's)\b.*?:\s*"
                       r"(.+?)\s+wins", re.I)

# ledger sport -> Kalshi series that settles it
_KALSHI_SERIES = {"basketball": "KXWNBAGAME", "wnba": "KXWNBAGAME",
                  "cricket": "KXHUNDREDMATCH",
                  "nfl": "KXNFLGAME", "football": "KXNFLGAME"}


def _kalshi_results(series: str) -> dict:
    """(YYYY-MM-DD, frozenset of both sides) -> winning side, from Kalshi.

    Basketball had no settle path at all, so every WNBA row auto-lock created
    would have stayed open forever — Indiana Fever sat in Open having already
    lost. Cricket and NFL were about to hit the same wall: nothing locked yet,
    but the models produce ALIGNED calls in both and a lock would have stranded.

    Kalshi is the right source for all three. It settles within minutes of the
    final whistle, and it is the same venue the price came from, so the result
    matches the market the pick was scored against. The sport-specific archives
    are slower — basketball-reference still had a finished game unplayed hours
    afterwards.
    """
    out: dict = {}
    cursor = None
    for _ in range(6):                       # ~1200 markets is months of games
        params = {"series_ticker": series, "status": "settled", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        try:
            r = requests.get(f"{K}/markets", params=params, timeout=45)
            r.raise_for_status()
            body = r.json()
        except (requests.RequestException, ValueError):
            break
        ms = body.get("markets", [])
        for m in ms:
            if str(m.get("result", "")).lower() != "yes":
                continue                     # only the winning leg names it
            title = str(m.get("title", ""))
            tick = str(m.get("ticker", ""))
            mw = _KT_WILL.match(title)
            if mw:
                winner, a, b = (x.strip() for x in mw.groups())
            else:
                mc = _KT_COLON.match(title)
                if not mc:
                    continue
                a, b, winner = (x.strip() for x in mc.groups())
            # close_time is just after the final whistle; the ticker carries the
            # scheduled date (…-26AUG06LVIND-…), which is the one our event
            # strings use. Prefer it over deriving a date from close_time, which
            # rolls past midnight UTC for a night game.
            dm = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", tick)
            if not dm:
                continue
            yy, mon, dd = dm.groups()
            try:
                d = datetime.strptime(f"{dd}{mon}{yy}", "%d%b%y").date()
            except ValueError:
                continue
            out[(d.isoformat(), frozenset({a.lower(), b.lower()}))] = winner.lower()
        cursor = body.get("cursor")
        if not cursor or not ms:
            break
    return out


_K_CACHE: dict[str, dict] = {}


def settle_via_kalshi(rec: dict, sport: str) -> tuple[str, str] | None:
    series = _KALSHI_SERIES.get(sport)
    if not series:
        return None
    home, away, date = parse_event(rec["event"])
    if not away or not date:
        return None
    if series not in _K_CACHE:
        _K_CACHE[series] = _kalshi_results(series)
    # Kalshi names cities ("Las Vegas"), our events name franchises ("Las Vegas
    # Aces"), so match on whichever side's words overlap.
    def toks(s):
        # Two characters, not three. "MI London" vs "London Spirit" both reduce
        # to {london} under a 3-char floor, the tie-break picks the wrong side,
        # and the fixture never settles. "MI" is the whole distinction.
        return {w for w in re.sub(r"[^a-z ]", " ", s.lower()).split() if len(w) >= 2}
    ht, at = toks(home), toks(away)
    # Kalshi's ticker date is the venue-local date; our event string carries the
    # ET date. A 01:00 ET tip-off is the previous day at the venue, which is why
    # Toronto @ Portland (event 08-07, ticker 26AUG06) would not match on an
    # exact date. Widening by a day is safe here because BOTH team names must
    # still match — two teams do not meet twice in 24 hours.
    want = {date}
    try:
        d0 = datetime.strptime(date, "%Y-%m-%d")
        want |= {(d0 + timedelta(days=k)).strftime("%Y-%m-%d") for k in (-1, 1)}
    except ValueError:
        pass
    for (d, pair), winner in _K_CACHE[series].items():
        if d not in want:
            continue
        names = list(pair)
        m_home = max(names, key=lambda n: len(toks(n) & ht))
        m_away = max(names, key=lambda n: len(toks(n) & at))
        if m_home == m_away or not (toks(m_home) & ht) or not (toks(m_away) & at):
            continue
        outcome = "HOME" if winner == m_home else "AWAY" if winner == m_away else None
        if outcome is None:
            continue
        return outcome, f"Kalshi settled: {winner.title()} won"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = load()
    # Voided rows are excluded from scoring, so settling them adds noise to the
    # log without changing any number.
    open_rows = [r for r in rows
                 if r.get("outcome") is None and not r.get("voided")]
    if not open_rows:
        print("no open predictions")
        return
    print(f"{len(open_rows)} open prediction(s)")

    hist = None
    settled = unresolved = 0
    for rec in rows:
        if rec.get("outcome") is not None or rec.get("voided"):
            continue
        sport = str(rec.get("sport", "")).lower()
        res = None
        if sport == "soccer":
            if hist is None:
                hist = D.load_history()
            res = settle_soccer(rec, hist)
        elif sport in ("baseball", "mlb"):
            res = settle_mlb(rec)
        elif sport in _KALSHI_SERIES:
            res = settle_via_kalshi(rec, sport)

        if res is None:
            unresolved += 1
            print(f"  #{rec['id']:<3} UNRESOLVED  {rec['event']}")
            continue
        outcome, detail = res
        won = bool(outcome == rec["pick"])
        print(f"  #{rec['id']:<3} {outcome:<5} ({'WON ' if won else 'lost'})  "
              f"{rec['event']}  [{detail}]")
        if args.dry_run:
            continue
        rec["outcome"] = outcome
        rec["settled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rec["won"] = won
        if rec.get("odds") and rec.get("stake_units"):
            rec["pnl_units"] = (rec["stake_units"] * (rec["odds"] - 1)) if won \
                else -rec["stake_units"]
        settled += 1

    if not args.dry_run and settled:
        save(rows)
    print(f"\nsettled {settled}, left open {unresolved}"
          f"{'  (dry run — nothing written)' if args.dry_run else ''}")
    if unresolved:
        print("  Unresolved rows are left alone on purpose: a guessed settlement")
        print("  corrupts the record more than a missing one.")


if __name__ == "__main__":
    main()
