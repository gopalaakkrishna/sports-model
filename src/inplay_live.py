"""Compare the in-play model against Kalshi's live prices, and log both for scoring.

This is the test the in-play model has not had. Beating a base rate only shows it
knows more than nothing; the question is whether it knows anything Kalshi's live
price does not — and Kalshi sees baserunners, the current pitcher and the
bullpen, none of which this model has.

Each poll writes one row per in-progress game: the game state, the model's win
probability, and the market's. Rows accumulate into a file that can be scored
once the games settle, which is the only way to answer the question honestly.

    python src/inplay_live.py            # one snapshot
    python src/inplay_live.py --watch 12 # poll every 2 min for 12 cycles

Nothing here places a trade.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from inplay_mlb import InPlayState, win_probability

ROOT = Path(__file__).resolve().parents[1]
STATS = "https://statsapi.mlb.com/api/v1"
K = "https://api.elections.kalshi.com/trade-api/v2"
LOG = ROOT / "data" / "processed" / "inplay_live_log.csv"

KALSHI_TO_MLB = {
    "Los Angeles A": "Los Angeles Angels", "Los Angeles D": "Los Angeles Dodgers",
    "New York Y": "New York Yankees", "New York M": "New York Mets",
    "Chicago C": "Chicago Cubs", "Chicago W": "Chicago White Sox",
    "Chicago WS": "Chicago White Sox", "A's": "Athletics",
    "Athletics": "Athletics", "Boston": "Boston Red Sox",
    "Atlanta": "Atlanta Braves", "Detroit": "Detroit Tigers",
    "San Francisco": "San Francisco Giants", "Arizona": "Arizona Diamondbacks",
    "Houston": "Houston Astros", "San Diego": "San Diego Padres",
    "Tampa Bay": "Tampa Bay Rays", "Seattle": "Seattle Mariners",
    "Baltimore": "Baltimore Orioles", "Texas": "Texas Rangers",
    "Colorado": "Colorado Rockies", "St. Louis": "St. Louis Cardinals",
    "Miami": "Miami Marlins", "Milwaukee": "Milwaukee Brewers",
    "Minnesota": "Minnesota Twins", "Kansas City": "Kansas City Royals",
    "Cleveland": "Cleveland Guardians", "Cincinnati": "Cincinnati Reds",
    "Pittsburgh": "Pittsburgh Pirates", "Philadelphia": "Philadelphia Phillies",
    "Washington": "Washington Nationals", "Toronto": "Toronto Blue Jays",
}


def load_cal() -> dict:
    p = ROOT / "data" / "processed" / "inplay_calibration.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"platt_a": 1.0, "platt_b": 0.0, "extras_home_edge": 0.495,
            "league_runs_9": 4.5}


def team_rates() -> dict[str, tuple[float, float]]:
    g = pd.read_parquet(ROOT / "data" / "raw" / "mlb_linescores.parquet")
    g["date"] = pd.to_datetime(g["date"])
    recent = g[g["date"] >= g["date"].max() - pd.Timedelta(days=400)]
    out = {}
    for t in set(recent["home_team"]) | set(recent["away_team"]):
        h, a = recent[recent["home_team"] == t], recent[recent["away_team"] == t]
        out[t] = (float(pd.concat([h["home_score"], a["away_score"]]).mean()),
                  float(pd.concat([h["away_score"], a["home_score"]]).mean()))
    return out


def live_games() -> list[dict]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = []
    for date in (today,):
        try:
            r = requests.get(f"{STATS}/schedule",
                             params={"sportId": 1, "date": date,
                                     "hydrate": "linescore,team"}, timeout=45)
            for d in r.json().get("dates", []):
                for g in d.get("games", []):
                    if (g.get("status") or {}).get("abstractGameState") != "Live":
                        continue
                    ls = g.get("linescore") or {}
                    t = g["teams"]
                    out.append({
                        "gamePk": g["gamePk"],
                        "home": t["home"]["team"]["name"],
                        "away": t["away"]["team"]["name"],
                        "home_score": t["home"].get("score") or 0,
                        "away_score": t["away"].get("score") or 0,
                        "inning": ls.get("currentInning"),
                        "half": "top" if ls.get("isTopInning") else "bottom",
                        "outs": ls.get("outs") or 0,
                    })
        except requests.RequestException:
            pass
    return out


def kalshi_live() -> dict[tuple[str, str], dict]:
    try:
        r = requests.get(f"{K}/markets", params={"series_ticker": "KXMLBGAME",
                                                 "status": "open", "limit": 200},
                         timeout=45)
        ms = r.json().get("markets", [])
    except requests.RequestException:
        return {}
    from collections import defaultdict
    by = defaultdict(dict)
    for m in ms:
        title = str(m.get("title", "")).replace(" Winner?", "").strip()
        if " vs " not in title:
            continue
        a_raw, b_raw = [x.strip() for x in title.split(" vs ", 1)]
        away, home = KALSHI_TO_MLB.get(a_raw), KALSHI_TO_MLB.get(b_raw)
        team = KALSHI_TO_MLB.get(str(m.get("yes_sub_title", "")).strip())
        if not (away and home and team):
            continue
        try:
            bid = float(m.get("yes_bid_dollars"))
            ask = float(m.get("yes_ask_dollars"))
        except (TypeError, ValueError):
            continue
        by[(home, away)][team] = {"bid": bid, "ask": ask}
    return dict(by)


def snapshot(rates, cal) -> pd.DataFrame:
    games = live_games()
    mk = kalshi_live()
    a, b = cal["platt_a"], cal["platt_b"]
    league = cal["league_runs_9"]
    rows = []
    now = datetime.now(timezone.utc)
    for g in games:
        if not g["inning"]:
            continue
        hs, ha = rates.get(g["home"], (league, league))
        as_, aa = rates.get(g["away"], (league, league))
        p = win_probability(
            InPlayState(int(g["inning"]), g["half"], int(g["outs"]),
                        int(g["home_score"]), int(g["away_score"])),
            (hs + aa) / 2, (as_ + ha) / 2,
            extras_home_edge=cal["extras_home_edge"])
        z = np.log(max(p["p_home"], 1e-15) / max(1 - p["p_home"], 1e-15))
        model = float(1 / (1 + np.exp(-(a * z + b))))

        legs = mk.get((g["home"], g["away"]), {})
        hm, am = legs.get(g["home"]), legs.get(g["away"])
        mkt = np.nan
        if hm and am and (hm["ask"] + am["ask"]) > 0:
            mkt = hm["ask"] / (hm["ask"] + am["ask"])
        rows.append({
            "ts": now.isoformat(timespec="seconds"), "gamePk": g["gamePk"],
            "match": f"{g['away']} @ {g['home']}",
            "inning": g["inning"], "half": g["half"], "outs": g["outs"],
            "score": f"{g['away_score']}-{g['home_score']}",
            "model_home": model, "model_raw": p["p_home"],
            "mkt_home": mkt,
            "home_bid": hm["bid"] if hm else np.nan,
            "home_ask": hm["ask"] if hm else np.nan,
            "diff": (model - mkt) if mkt == mkt else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=1, help="number of polls")
    ap.add_argument("--interval", type=int, default=120, help="seconds between polls")
    args = ap.parse_args()

    cal = load_cal()
    rates = team_rates()
    print(f"in-play calibration a={cal['platt_a']:.3f} b={cal['platt_b']:+.3f}, "
          f"extras edge {cal['extras_home_edge']:.3f}\n")

    for i in range(args.watch):
        d = snapshot(rates, cal)
        stamp = datetime.now(timezone.utc).astimezone()
        if d.empty:
            print(f"[{stamp:%H:%M:%S}] no MLB games in progress")
        else:
            print(f"[{stamp:%H:%M:%S}] {len(d)} live game(s)")
            print(f"  {'match':<40}{'state':<16}{'model':>7}{'mkt':>7}{'diff':>8}")
            for _, r in d.iterrows():
                state = f"{r['half'][:3]} {r['inning']}, {r['outs']}out {r['score']}"
                mk = f"{r['mkt_home']:.0%}" if r["mkt_home"] == r["mkt_home"] else "  -"
                df = f"{r['diff']:+.1%}" if r["diff"] == r["diff"] else "   -"
                print(f"  {r['match'][:39]:<40}{state:<16}"
                      f"{r['model_home']:>7.0%}{mk:>7}{df:>8}")
            LOG.parent.mkdir(parents=True, exist_ok=True)
            d.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
        if i < args.watch - 1:
            time.sleep(args.interval)

    if LOG.exists():
        allrows = pd.read_csv(LOG)
        print(f"\n  log now holds {len(allrows):,} state snapshots "
              f"across {allrows['gamePk'].nunique()} games -> {LOG}")
        got = allrows.dropna(subset=["mkt_home"])
        if len(got) > 30:
            print(f"  mean |model - market| = {got['diff'].abs().mean():.3f}")
            print("  Score these once the games settle: that comparison, not the")
            print("  base-rate result, decides whether the in-play model is useful.")


if __name__ == "__main__":
    main()
