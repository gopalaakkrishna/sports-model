"""Price MLB season win totals by simulating the rest of the schedule.

WHY THIS MARKET

Game winners are the sharpest market in sports — most volume, most attention,
most professional money — and 68k matches plus a live record say we lose to
them. Kalshi lists 3,193 sports series and we trade exactly one type. Season
win totals are a different market: fewer eyes, and they require compounding
forty-odd games, which retail traders are demonstrably bad at.

We can price them today with no new model. mlb_model already prices any
matchup; the remaining schedule comes from StatsAPI; a Monte Carlo over those
games gives a full distribution of final win totals, and "P(team wins >= N)"
reads straight off it.

WHAT THIS DOES NOT ESTABLISH

That we can beat this market. Our per-game model loses to the per-game line, so
compounding it over 40 games does not obviously produce an edge — it may just
compound the same error. The bet is that the SEASON market is softer than the
GAME market, and that is a hypothesis this script tests rather than assumes.

Disagreement with Kalshi is reported, not acted on. On the game market, large
disagreement was measured to be model error far more often than edge (1.0373
vs 0.9641 log loss), and there is no reason yet to think this market is
different.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import mlb_model as MM

STATS = "https://statsapi.mlb.com/api/v1"
K = "https://api.elections.kalshi.com/trade-api/v2"


def standings(season: int) -> dict:
    """team id -> (full name, wins, losses).

    Keyed on ID, not name. The standings endpoint returns the SHORT name
    ("Rays") while the schedule returns the full one ("Tampa Bay Rays"), so
    joining on name silently matched nothing — every simulated game was
    dropped and each team's projection came back exactly equal to its current
    win total. The id is the only field both endpoints agree on.
    """
    out = {}
    r = requests.get(f"{STATS}/standings",
                     params={"leagueId": "103,104", "season": season,
                             "standingsTypes": "regularSeason"}, timeout=60)
    r.raise_for_status()
    for rec in r.json().get("records", []):
        for t in rec.get("teamRecords", []):
            out[int(t["team"]["id"])] = (t["team"]["name"],
                                         int(t["wins"]), int(t["losses"]))
    return out


def remaining(start: str, end: str) -> list[dict]:
    r = requests.get(f"{STATS}/schedule",
                     params={"sportId": 1, "startDate": start, "endDate": end,
                             "hydrate": "team,venue"}, timeout=90)
    r.raise_for_status()
    games = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            if g.get("gameType") != "R":
                continue
            st = (g.get("status") or {}).get("detailedState", "")
            if st in ("Final", "Game Over", "Completed Early"):
                continue
            t = g["teams"]
            games.append({"home": t["home"]["team"]["name"],
                          "away": t["away"]["team"]["name"],
                          "home_id": int(t["home"]["team"]["id"]),
                          "away_id": int(t["away"]["team"]["id"]),
                          "venue": (g.get("venue") or {}).get("name")})
    return games


def kalshi_wins_markets() -> dict:
    """team abbrev -> [(threshold, ask, bid)] from KXMLBWINS-* series."""
    out = defaultdict(list)
    r = requests.get(f"{K}/series", params={"category": "Sports"}, timeout=60)
    if r.status_code != 200:
        return out
    tickers = [s["ticker"] for s in r.json().get("series", [])
               if str(s.get("ticker", "")).startswith("KXMLBWINS-")]
    for tk in tickers:
        try:
            m = requests.get(f"{K}/markets",
                             params={"series_ticker": tk, "status": "open",
                                     "limit": 200}, timeout=45)
            if m.status_code != 200:
                continue
            for mk in m.json().get("markets", []):
                title = str(mk.get("title", ""))
                import re
                n = re.search(r"at least (\d+) games", title)
                if not n:
                    continue
                try:
                    ask = float(mk.get("yes_ask_dollars"))
                    bid = float(mk.get("yes_bid_dollars"))
                except (TypeError, ValueError):
                    continue
                out[tk.replace("KXMLBWINS-", "")].append(
                    (int(n.group(1)), ask, bid, title))
        except requests.RequestException:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--season", type=int, default=None)
    args = ap.parse_args()

    today = pd.Timestamp.now().normalize()
    season = args.season or today.year

    games = pd.read_parquet(ROOT / "data" / "raw" / "mlb_games.parquet")
    cal = None
    calp = ROOT / "data" / "processed" / "mlb_calibration.json"
    if calp.exists():
        import json
        c = json.loads(calp.read_text())
        cal = (c.get("a"), c.get("b"))
    fit = MM.fit(games, today, calib=cal)
    print(f"fitted {fit.n_games:,} games, {len(fit.teams)} teams")

    st = standings(season)
    rem = remaining(today.strftime("%Y-%m-%d"), f"{season}-10-05")
    print(f"standings for {len(st)} teams, {len(rem)} games left to play")
    if not rem:
        print("no remaining games")
        return 0

    # Win probability for each remaining game, once.
    probs = []
    for g in rem:
        p = MM.predict(fit, g["home"], g["away"], None, None, g["venue"])
        if p is None:
            continue
        probs.append((g["home_id"], g["away_id"], float(p["p_home"])))
    print(f"priced {len(probs)} of {len(rem)} remaining games")

    team_ids = sorted(st)
    idx = {tid: i for i, tid in enumerate(team_ids)}
    name = {tid: st[tid][0] for tid in team_ids}
    base = np.array([st[tid][1] for tid in team_ids], dtype=np.int32)

    rng = np.random.default_rng(0)
    sims = np.repeat(base[None, :], args.sims, axis=0)
    used = 0
    for hid, aid, ph in probs:
        if hid not in idx or aid not in idx:
            continue
        hw = rng.random(args.sims) < ph
        sims[:, idx[hid]] += hw
        sims[:, idx[aid]] += ~hw
        used += 1
    print(f"simulated {used} of {len(probs)} priced games")
    if used == 0:
        print("!! nothing simulated — team ids did not join. Aborting rather "
              "than reporting projections equal to current records.")
        return 1

    print(f"\n{'team':<24}{'now':>8}{'proj':>8}{'p5':>6}{'p95':>6}")
    proj = sims.mean(axis=0)
    for tid in sorted(team_ids, key=lambda x: -proj[idx[x]])[:8]:
        i = idx[tid]
        print(f"  {name[tid]:<22}{st[tid][1]:>4}-{st[tid][2]:<3}{proj[i]:>8.1f}"
              f"{np.percentile(sims[:, i], 5):>6.0f}"
              f"{np.percentile(sims[:, i], 95):>6.0f}")

    mk = kalshi_wins_markets()
    if not mk:
        print("\nno open KXMLBWINS markets to compare against")
        return 0

    # Keyed on the SHORT name the standings endpoint returns ("Rays"), not the
    # full one the schedule returns ("Tampa Bay Rays") — mixing the two is what
    # already broke the simulation join once.
    ABBR = {
        "Angels": "LAA", "Astros": "HOU", "Athletics": "ATH", "Blue Jays": "TOR",
        "Braves": "ATL", "Brewers": "MIL", "Cardinals": "STL", "Cubs": "CHC",
        "D-backs": "AZ", "Dodgers": "LAD", "Giants": "SF", "Guardians": "CLE",
        "Mariners": "SEA", "Marlins": "MIA", "Mets": "NYM", "Nationals": "WSH",
        "Orioles": "BAL", "Padres": "SD", "Phillies": "PHI", "Pirates": "PIT",
        "Rangers": "TEX", "Rays": "TB", "Red Sox": "BOS", "Reds": "CIN",
        "Rockies": "COL", "Royals": "KC", "Tigers": "DET", "Twins": "MIN",
        "White Sox": "CWS", "Yankees": "NYY",
    }
    print(f"\n{'market':<34}{'model':>8}{'ask':>7}{'edge':>8}{'spread':>8}")
    hits = 0
    for tid, i in idx.items():
        ab = ABBR.get(name[tid])
        if not ab or ab not in mk:
            continue
        for thresh, ask, bid, title in sorted(mk[ab]):
            p = float((sims[:, i] >= thresh).mean())
            edge = p - ask
            sp = ask - bid
            flag = "  <--" if abs(edge) > 0.10 and sp <= 0.06 else ""
            print(f"  {ab + ' ' + str(thresh) + '+ wins':<32}{p:>8.0%}{ask:>7.2f}"
                  f"{edge:>+8.0%}{sp * 100:>7.0f}c{flag}")
            hits += 1
    if not hits:
        print("  (no team-name matches against open markets)")
    print("\nEdge shown is model minus ask. It is NOT a recommendation — on the")
    print("game market, disagreement this large measured as model error, not")
    print("opportunity. Log these and score them before believing any of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
