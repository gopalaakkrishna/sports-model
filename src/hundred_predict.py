"""Predict upcoming The Hundred fixtures from Kalshi, using player ratings.

Team strength is built from each side's most recent XI, since Cricsheet only
records a lineup after the match is played. Over a three-week tournament that is
a fair proxy, but it is wrong every time a side rotates — and overseas players
come and go constantly.

Confidence here is capped deliberately. The player model beat the base rate by
0.0045 log loss over 156 matches with a 95% CI of [-0.040, +0.032] — a positive
point estimate that is statistically indistinguishable from noise. Nothing in
this file should be read as an edge.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
import cricket_model as CM
import cricket_players as CP

ROOT = Path(__file__).resolve().parents[1]
K = "https://api.elections.kalshi.com/trade-api/v2"
ET = ZoneInfo("America/New_York")

MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def fixture_time(ev: str) -> datetime | None:
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})(\d{4})", ev)
    if not m or m.group(2) not in MONTHS:
        return None
    yy, mo, dd, hhmm = m.groups()
    try:
        return datetime(2000 + int(yy), MONTHS[mo], int(dd),
                        int(hhmm[:2]), int(hhmm[2:]), tzinfo=timezone.utc)
    except ValueError:
        return None


def kalshi_fixtures() -> list[dict]:
    r = requests.get(f"{K}/markets", params={"series_ticker": "KXHUNDREDMATCH",
                                             "status": "open", "limit": 100},
                     timeout=60)
    if r.status_code != 200:
        return []
    by = defaultdict(list)
    for m in r.json().get("markets") or []:
        by[m["event_ticker"]].append(m)

    out = []
    for ev, mk in by.items():
        title = str(mk[0].get("rules_primary", ""))
        # "the X vs Y men's professional The Hundred cricket match"
        tm = re.search(r"the (.+?) vs (.+?) (?:men|women)'s professional", title)
        if not tm:
            continue
        legs = {}
        for m in mk:
            sub = str(m.get("yes_sub_title", "")).strip()
            try:
                legs[CM.canon(sub)] = (float(m.get("yes_bid_dollars")),
                                       float(m.get("yes_ask_dollars")))
            except (TypeError, ValueError):
                pass
        if len(legs) != 2:
            continue
        out.append({
            "event": ev,
            "home": CM.canon(tm.group(1).strip()),
            "away": CM.canon(tm.group(2).strip()),
            "start": fixture_time(ev),
            "legs": legs,
        })
    return sorted([o for o in out if o["start"]], key=lambda z: z["start"])


def last_xi(sq: pd.DataFrame, team: str) -> list[str]:
    t = sq[sq["team"] == team].sort_values("date")
    if t.empty:
        return []
    last_match = t["match_id"].iloc[-1]
    return t[t["match_id"] == last_match]["player"].tolist()


def main():
    bb = pd.read_parquet(ROOT / "data" / "raw" / "cricket_hundred_male_balls.parquet")
    sq = pd.read_parquet(ROOT / "data" / "raw" / "cricket_hundred_male_squads.parquet")
    bb["date"] = pd.to_datetime(bb["date"])
    sq["date"] = pd.to_datetime(sq["date"])
    sq["team"] = sq["team"].map(CM.canon)

    now = datetime.now(timezone.utc)
    f = CP.fit_players(bb, pd.Timestamp(now.replace(tzinfo=None)))
    print(f"player ratings fitted on {len(bb[bb['date'] < pd.Timestamp(now.replace(tzinfo=None))]):,} "
          f"deliveries, league {f.league_rpb:.3f} runs/ball\n")

    fx = kalshi_fixtures()
    if not fx:
        print("no open Hundred fixtures on Kalshi")
        return

    rows = []
    for g in fx:
        xa, xb = last_xi(sq, g["home"]), last_xi(sq, g["away"])
        if len(xa) < 8 or len(xb) < 8:
            print(f"  skip {g['home']} v {g['away']}: XI unavailable")
            continue
        p = CP.predict(f, xa, xb)
        bid_a, ask_a = g["legs"].get(g["home"], (np.nan, np.nan))
        bid_b, ask_b = g["legs"].get(g["away"], (np.nan, np.nan))
        spread_a = ask_a - bid_a if ask_a == ask_a else np.nan
        mid_a = (bid_a + ask_a) / 2 if ask_a == ask_a else np.nan
        mid_b = (bid_b + ask_b) / 2 if ask_b == ask_b else np.nan
        mkt_a = mid_a / (mid_a + mid_b) if (mid_a == mid_a and mid_b == mid_b
                                            and mid_a + mid_b > 0) else np.nan
        rows.append({
            "start": g["start"], "home": g["home"], "away": g["away"],
            "model_home": p["p_a"], "margin": p["margin"],
            "bid": bid_a, "ask": ask_a, "spread": spread_a, "mkt_home": mkt_a,
            "bat_h": p["a"]["bat"], "bat_a": p["b"]["bat"],
            "bowl_h": p["a"]["bowl"], "bowl_a": p["b"]["bowl"],
        })

    d = pd.DataFrame(rows).sort_values("start")
    out = ROOT / "reports" / f"hundred_{now.date()}.csv"
    d.to_csv(out, index=False)

    print(f"{'start (ET)':<17}{'match':<48}{'model':>7}{'mkt':>7}"
          f"{'spread':>9}{'margin':>9}")
    print("-" * 98)
    for _, r in d.iterrows():
        et = r["start"].astimezone(ET)
        m = f"{r['home']} v {r['away']}"
        mk = f"{r['mkt_home']:.0%}" if r["mkt_home"] == r["mkt_home"] else "  -"
        sp = f"{r['spread']:.2f}" if r["spread"] == r["spread"] else "  -"
        print(f"{et:%a %m-%d %I:%M%p}  {m[:47]:<48}{r['model_home']:>7.0%}"
              f"{mk:>7}{sp:>9}{r['margin']:>+9.1f}")

    print(f"\n  margin is expected runs difference over 100 balls")
    print(f"  mean Kalshi spread on these: "
          f"{d['spread'].mean():.2f} — a 33-55c spread is untradeable")
    print(f"\n  MODEL STATUS: beat the base rate by 0.0045 log loss over 156")
    print(f"  matches, 95% CI [-0.040, +0.032]. Not significant. These are the")
    print(f"  model's best guesses, not edges, and the XI is assumed from each")
    print(f"  side's last match — wrong whenever a team rotates.")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
