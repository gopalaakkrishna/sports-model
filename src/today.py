"""Everything starting from now, ordered by kick-off, with a confidence tier.

Pulls start times from the sources that carry them — Kalshi event tickers and
occurrence times for soccer and WNBA, MLB StatsAPI for baseball — then attaches
the model's view and the market's, and sorts chronologically.

Confidence tiers are ordered by MEASURED reliability, not by apparent edge:

  HIGH    model within 3 points of the market — the zone where backtests show
          the model matching the closing line almost exactly
  MEDIUM  3-7 points apart
  LOW     more than 7 points apart — measured to be model error far more often
          than edge (large-disagreement backtest: model 1.0373, market 0.9641)

So a big disagreement lowers confidence rather than raising it. That is
backwards from how a tip sheet reads, and it is what the data supports.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
K = "https://api.elections.kalshi.com/trade-api/v2"
STATS = "https://statsapi.mlb.com/api/v1"

# Display in US Eastern. Using the zone rather than a fixed -5 offset so the
# EST/EDT switch is handled automatically — in August this resolves to EDT.
ET = ZoneInfo("America/New_York")


def tier(disagreement: float) -> str:
    if disagreement != disagreement:      # NaN — no market to compare against
        return "n/a"
    if disagreement < 0.03:
        return "HIGH"
    if disagreement < 0.07:
        return "MEDIUM"
    return "LOW"


def kalshi_events(series: str) -> dict[str, dict]:
    """event_ticker -> {start, title, legs:{sub: (bid, ask)}}"""
    try:
        r = requests.get(f"{K}/markets", params={"series_ticker": series,
                                                 "status": "open", "limit": 200},
                         timeout=45)
        if r.status_code != 200:
            return {}
        ms = r.json().get("markets", [])
    except requests.RequestException:
        return {}
    by = defaultdict(lambda: {"legs": {}, "start": None, "title": ""})
    for m in ms:
        ev = m.get("event_ticker")
        e = by[ev]
        e["title"] = str(m.get("title", ""))
        t = m.get("occurrence_datetime") or m.get("expected_expiration_time")
        if t and not e["start"]:
            try:
                e["start"] = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
            except ValueError:
                pass
        try:
            e["legs"][str(m.get("yes_sub_title", "")).strip()] = (
                float(m.get("yes_bid_dollars")), float(m.get("yes_ask_dollars")))
        except (TypeError, ValueError):
            pass
    return dict(by)


def load_predictions() -> pd.DataFrame:
    """Reuse whatever today's per-sport runs already produced."""
    today = pd.Timestamp.now().normalize().date()
    frames = []
    for name, path, cols in [
        ("soccer", f"kalshi_edge_{today}.csv",
         ("match", "leg", "model", "ask", "when")),
        ("wnba", f"wnba_kalshi_{today}.csv",
         ("match", "team", "model", "ask", "date")),
        ("mlb", f"mlb_predictions_{today}.csv",
         ("match", "team", "model", "ask", "date")),
    ]:
        p = ROOT / "reports" / path
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if d.empty:
            continue
        d = d.rename(columns={cols[1]: "pick", cols[4]: "when"})
        d["sport"] = name
        keep = [c for c in ("sport", "match", "pick", "model", "ask", "when",
                            "tradeable", "depth") if c in d.columns]
        frames.append(d[keep])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def mlb_times() -> dict[str, datetime]:
    out = {}
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    try:
        r = requests.get(f"{STATS}/schedule",
                         params={"sportId": 1, "date": today, "hydrate": "team"},
                         timeout=45)
        for d in r.json().get("dates", []):
            for g in d.get("games", []):
                t = g.get("gameDate")
                if not t:
                    continue
                key = (f"{g['teams']['away']['team']['name']} @ "
                       f"{g['teams']['home']['team']['name']}")
                out[key] = datetime.fromisoformat(t.replace("Z", "+00:00"))
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=30,
                    help="look ahead this many hours")
    ap.add_argument("--all", action="store_true",
                    help="show every fixture, not just HIGH-confidence ones")
    ap.add_argument("--lock", action="store_true",
                    help="also write the HIGH picks to the ledger")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    now_et = now.astimezone(ET)
    print(f"Now: {now_et:%Y-%m-%d %I:%M %p} {now_et:%Z}   "
          f"(showing the next {args.hours}h)\n")

    preds = load_predictions()
    if preds.empty:
        print("No prediction files for today. Run daily_slate.py first.")
        return

    # Start times: Kalshi for soccer/WNBA, StatsAPI for MLB.
    starts: dict[str, datetime] = {}
    for series in ("KXMLSGAME", "KXLALIGAGAME", "KXBUNDESLIGA2GAME",
                   "KXLIGAMXGAME", "KXALLSVENSKANGAME", "KXSCOTTISHPREMGAME",
                   "KXJLEAGUEGAME", "KXELITESERIENGAME", "KXWNBAGAME"):
        for ev, e in kalshi_events(series).items():
            if not e["start"]:
                continue
            t = e["title"].replace(" Winner?", "").strip()
            if " vs " in t:
                a, b = [x.strip() for x in t.split(" vs ", 1)]
                for key in (f"{a} v {b}", f"{b} v {a}",
                            f"{a} @ {b}", f"{b} @ {a}"):
                    starts.setdefault(key, e["start"])
    starts.update(mlb_times())

    def match_start(row):
        m = str(row["match"])
        if m in starts:
            return starts[m]
        # Fall back to a loose match on the two team names.
        parts = re.split(r"\s+(?:v|@|vs)\s+", m)
        if len(parts) == 2:
            for k, v in starts.items():
                if parts[0][:12] in k and parts[1][:12] in k:
                    return v
        return None

    preds["start"] = preds.apply(match_start, axis=1)
    have = preds.dropna(subset=["start"]).copy()
    if have.empty:
        print("Could not resolve start times for any fixture.")
        return

    # Market probability, normalised per fixture so it is a probability.
    have["mkt"] = have.groupby(["sport", "match"])["ask"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else s)
    have["disagreement"] = (have["model"] - have["mkt"]).abs()
    have["tier"] = have["disagreement"].map(tier)

    # Only games that have NOT started. Kalshi keeps quoting through a game, so
    # an in-play price reflects the current score while the model still holds a
    # pre-game view. Comparing the two invents enormous edges: a Mets fixture an
    # hour after first pitch showed the market at 78% against the model's 46%,
    # which is the scoreboard talking, not a disagreement worth acting on.
    horizon = now + pd.Timedelta(hours=args.hours)
    started = have[have["start"] <= now]
    live = have[(have["start"] > now) & (have["start"] <= horizon)]
    if len(started):
        n_started = started.groupby(["sport", "match"]).ngroups
        print(f"  ({n_started} fixture(s) already under way — excluded, "
              f"their prices are in-play)\n")
    if live.empty:
        nxt = have[have["start"] > now]["start"].min()
        print(f"Nothing in the next {args.hours}h. Next start: {nxt}")
        return

    # One row per fixture: the model's most likely outcome.
    best = (live.sort_values("model", ascending=False)
                .groupby(["sport", "match"], as_index=False).first()
                .sort_values("start"))

    # HIGH only by default. That band is where the model measurably matches the
    # closing line; MEDIUM and LOW are where it disagrees, and disagreement has
    # been measured as the model's error far more often than as edge.
    if not args.all:
        shown = best[best["tier"] == "HIGH"]
        hidden = len(best) - len(shown)
        best = shown
        if hidden:
            print(f"  ({hidden} fixture(s) hidden: MEDIUM/LOW confidence or no "
                  f"market. Use --all to see them.)\n")
        if best.empty:
            print("No HIGH-confidence fixtures in this window.")
            print("Reporting none rather than promoting weaker candidates.")
            return

    # The BET goes first and in capitals. An earlier layout put the fixture
    # name first and the pick in a middle column, and a row reading
    # "Detroit @ Seattle ... Seattle ... HIGH" was acted on as a Detroit bet.
    # The side being backed must be the first thing read, never inferred from
    # the fixture name.
    print(f"{'start (ET)':<18}{'BET ON':<26}{'in fixture':<34}"
          f"{'model':>7}{'mkt':>7}  conf")
    print("=" * 106)
    for _, r in best.iterrows():
        et = r["start"].astimezone(ET)
        mkt = f"{r['mkt']:.0%}" if r["mkt"] == r["mkt"] else "  -"
        pick = str(r["pick"])
        # For 1X2 soccer legs, spell out which side HOME/AWAY means.
        fixture = str(r["match"])
        if pick in ("HOME", "AWAY", "DRAW"):
            parts = re.split(r"\s+(?:v|@|vs)\s+", fixture)
            if len(parts) == 2:
                pick = {"HOME": parts[0], "AWAY": parts[1],
                        "DRAW": "DRAW"}[pick]
        print(f"{et:%a %m-%d %I:%M%p}  {pick.upper()[:25]:<26}"
              f"{fixture[:33]:<34}{r['model']:>7.0%}{mkt:>7}  {r['tier']}")

    # Surfacing a pick and recording it must be one action. Toronto @ Cubs was
    # called HIGH on 2026-08-06 and never written down, so it was absent from
    # the record while still having been a call. A pick that only exists in a
    # printout is a pick that quietly disappears when it loses.
    if args.lock:
        import subprocess
        here = Path(__file__).parent
        locked = 0
        for _, r in best[best["tier"] == "HIGH"].iterrows():
            ev = f"{r['match']} ({r['start'].astimezone(ET):%Y-%m-%d})"
            cmd = [sys.executable, str(here / "ledger.py"), "lock",
                   "--sport", str(r["sport"]), "--event", ev,
                   "--market", "1X2" if r["sport"] == "soccer" else "ML",
                   "--pick", str(r["pick"]),
                   "--model-prob", f"{r['model']:.4f}",
                   "--venue", "kalshi", "--stake-units", "1.0",
                   "--notes", "auto-locked from today.py --lock (HIGH tier)"]
            if r["mkt"] == r["mkt"]:
                cmd += ["--market-prob", f"{r['mkt']:.4f}"]
            out = subprocess.run(cmd, capture_output=True, text=True)
            print("  " + out.stdout.strip().replace("\n", "\n  "))
            locked += int("locked #" in out.stdout)
        print(f"\n  wrote {locked} pick(s) to the ledger")

    n = best["tier"].value_counts()
    print(f"\n  HIGH {n.get('HIGH', 0)}   MEDIUM {n.get('MEDIUM', 0)}   "
          f"LOW {n.get('LOW', 0)}")
    print("  HIGH   = model within 3 pts of market (most reliable)")
    print("  LOW    = model far from market (measured to be model error)")


if __name__ == "__main__":
    main()
