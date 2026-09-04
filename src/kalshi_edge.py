"""Compare the soccer model against Kalshi's three-way game markets.

Kalshi lists each fixture as three binary markets (Home / Tie / Away), which
maps directly onto the 1X2 model. This finds the open fixtures, prices them with
the model as of today, and reports where the two disagree.

Three things this does that a naive edge scanner does not:

* **Buys at the ask.** You cannot transact at the mid. EV is computed against
  the price you would actually pay.
* **Charges Kalshi's fee.** Kalshi takes 0.07 * P * (1-P) per contract, which is
  1.75c at a 50c price. Typical model edges are of the same order, so ignoring
  the fee turns losing bets into apparent winners.
* **Screens for tradeability** before reporting anything, so an empty book
  cannot masquerade as a huge edge.

The three-way structure also gives a free diagnostic: the three asks should sum
to a little over 1. How far over is Kalshi's effective spread on that fixture.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
import data as D
import model as M
from team_names import TeamResolver

ROOT = Path(__file__).resolve().parents[1]
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi soccer series -> (country group to FIT on, division to price in).
#
# The fit must use the whole country group, never the single division. Pooling
# all of a country's tiers is what gives promoted and relegated clubs real
# ratings: fitting Bundesliga 2 alone would rate a side just down from the top
# flight purely on its handful of second-tier games. The division is still
# needed separately, because home advantage is fitted per division.
SERIES_COUNTRY = {
    "KXLALIGAGAME": ("Spain", "SP1"),
    "KXEPLGAME": ("England", "E0"),
    "KXSERIEAGAME": ("Italy", "I1"),
    "KXBUNDESLIGAGAME": ("Germany", "D1"),
    "KXBUNDESLIGA2GAME": ("Germany", "D2"),
    "KXLIGUE1GAME": ("France", "F1"),
    "KXMLSGAME": ("USA", "USA:MLS"),
    "KXLIGAMXGAME": ("Mexico", "MEX:Liga MX"),
    # Deliberately ABSENT: 2. Bundesliga, Allsvenskan, Eliteserien, Scottish
    # Premiership, J-League. They were added during the European summer for
    # coverage while the big leagues were dark, and cut 2026-08-14 when those
    # leagues came back. Two reasons, in order: the user asked the board to
    # narrow to major leagues; and the model's ratings are only as good as the
    # fixture graph behind them — the Nordic and second-tier graphs are thin
    # and poorly connected, so their predictions carry the most estimation
    # error exactly where the markets are least liquid. Their already-locked
    # picks still settle (settle.py keeps the full series list on purpose).
}

FEE_RATE = 0.07  # Kalshi trading fee coefficient


def kalshi_fee(price: float) -> float:
    """Fee per $1 contract at a given price."""
    return FEE_RATE * price * (1.0 - price)


def _norm(s: str) -> str:
    """Lowercase and strip punctuation, for tolerant name matching."""
    s = re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()
    return re.sub(r"[ ]+", " ", s)


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _event_titles(series: str) -> dict:
    """event_ticker -> (home, away), read from the EVENTS endpoint.

    Kalshi reformatted the per-market title in late Aug/early Sep 2026. It
    used to carry the whole fixture ("Arsenal vs Coventry Winner?") and now
    carries only the outcome ("Villarreal wins", "Tie is the result"). The
    fixture pairing lives on the EVENT, which still reads "A vs B".

    That change silently emptied the winner board: the old regex matched
    nothing, fetch_fixtures returned zero, this script printed "no comparable
    fixtures found" and exited 0, so auto_update logged it "ok" and
    export_tara fell back to the newest report on disk. The board kept
    publishing August picks under a fresh September timestamp for days.
    totals_predict.py was unaffected because it already read the events
    endpoint — which is why the totals lane kept working while this died.
    """
    r = requests.get(f"{KALSHI}/events",
                     params={"series_ticker": series, "status": "open",
                             "limit": 200}, timeout=60)
    if r.status_code != 200:
        return {}
    out = {}
    for e in r.json().get("events", []):
        m = re.match(r"^\s*(.+?)\s+vs\.?\s+(.+?)\s*$",
                     str(e.get("title", "")).split(":")[0])
        if m:
            out[str(e.get("event_ticker"))] = (m.group(1).strip(),
                                               m.group(2).strip())
    return out


def _leg_side(sub: str, home: str, away: str) -> str | None:
    """Which side a leg's yes_sub_title refers to.

    Exact match first, then a normalised prefix/containment test, because
    Kalshi's leg label and its own event title do not always agree on the
    long form ("Newcastle" vs "Newcastle United"). Ambiguity returns None
    rather than guessing — a leg attached to the wrong team would price the
    opposite side of the fixture.
    """
    s = sub.strip()
    if s.lower() in ("tie", "draw"):
        return "DRAW"
    if s == home:
        return "HOME"
    if s == away:
        return "AWAY"
    n = _norm(s)
    if not n:
        return None
    hit = []
    for side, name in (("HOME", home), ("AWAY", away)):
        nn = _norm(name)
        if nn and (nn.startswith(n) or n.startswith(nn) or n in nn or nn in n):
            hit.append(side)
    return hit[0] if len(hit) == 1 else None


def fetch_fixtures(series: str) -> list[dict]:
    """Group Kalshi's per-outcome markets back into fixtures."""
    events = _event_titles(series)
    if not events:
        return []
    r = requests.get(f"{KALSHI}/markets",
                     params={"series_ticker": series, "status": "open", "limit": 500},
                     timeout=60)
    if r.status_code != 200:
        return []
    by_event = defaultdict(list)
    for m in r.json().get("markets", []):
        by_event[m.get("event_ticker")].append(m)

    fixtures = []
    for ev, mk in by_event.items():
        pair = events.get(str(ev))
        if not pair:
            continue
        home, away = pair
        when = mk[0].get("occurrence_datetime") or mk[0].get("expected_expiration_time")
        legs = {}
        for m in mk:
            key = _leg_side(str(m.get("yes_sub_title", "")), home, away)
            if key is None:
                continue
            legs[key] = {
                "ticker": m.get("ticker"),
                "bid": _f(m.get("yes_bid_dollars")),
                "ask": _f(m.get("yes_ask_dollars")),
                "liq": _f(m.get("liquidity_dollars"), 0.0),
                "oi": _f(m.get("open_interest_fp"), 0.0),
            }
        if {"HOME", "DRAW", "AWAY"} <= set(legs):
            fixtures.append({"event": ev, "home": home, "away": away,
                             "when": when, "legs": legs})
    return fixtures


def orderbook_depth(ticker: str, within: float = 0.05) -> float:
    try:
        r = requests.get(f"{KALSHI}/markets/{ticker}/orderbook",
                         params={"depth": 10}, timeout=45)
        if r.status_code != 200:
            return 0.0
        book = r.json().get("orderbook") or r.json().get("orderbook_fp") or {}
    except requests.RequestException:
        return 0.0
    total = 0.0
    for side in ("yes", "yes_dollars", "no", "no_dollars"):
        levels = book.get(side) or []
        px = [_f(l[0], 0.0) for l in levels if len(l) >= 2]
        if not px:
            continue
        best = max(px)
        for l in levels:
            if len(l) >= 2 and abs(_f(l[0], 0.0) - best) <= within:
                total += _f(l[0], 0.0) * _f(l[1], 0.0)
    return total


def _blend_all(pending: list[dict], hist: pd.DataFrame) -> dict:
    """Blend the GBM into Dixon-Coles for every eligible pending fixture.

    Returns {(price_div, home, away): {"HOME":p, "DRAW":p, "AWAY":p}}; a
    fixture absent from the result keeps its Dixon-Coles forecast untouched.

    Only the five European majors are eligible. The booster is trained on
    football-data divisions E0/SP1/I1/D1/F1 and has never seen MLS or Liga MX,
    so blending them would be applying a model to leagues it was not fitted
    on. Those keep pure Dixon-Coles, which is what they had before.

    Every failure path here returns {} — an unavailable booster must degrade
    to the previous behaviour, never block the board. This step produces the
    picks; it is not allowed to be the thing that stops producing them.
    """
    if not pending:
        return {}
    try:
        import gbm_live as GL
        import gbm_model as GM
    except ImportError as e:
        print(f"  gbm blend unavailable ({e}); using Dixon-Coles alone")
        return {}

    w = GL.load_weight()
    if w <= 0:
        print("  gbm blend weight is 0; using Dixon-Coles alone")
        return {}

    elig = [r for r in pending if r["price_div"] in GM.MAJORS]
    skipped = len(pending) - len(elig)
    if not elig:
        return {}
    try:
        base = GM.build(list(GM.MAJORS))
        fixtures = pd.DataFrame([{
            "Div": r["price_div"], "Date": pd.Timestamp.now().normalize(),
            "HomeTeam": r["h"], "AwayTeam": r["a"],
            "m_home": r["pred"]["p_home"], "m_draw": r["pred"]["p_draw"],
            "m_away": r["pred"]["p_away"],
            "lam_h": r["pred"]["lambda_home"], "lam_a": r["pred"]["lambda_away"],
            "m_over25": r["pred"]["p_over25"],
        } for r in elig])
        out = GL.predict(base, fixtures, weight=w)
    except Exception as e:                      # noqa: BLE001 - never fatal
        print(f"  gbm blend failed ({type(e).__name__}: {e}); "
              f"using Dixon-Coles alone")
        return {}

    res = {}
    for _, r in out.iterrows():
        res[(r["Div"], r["HomeTeam"], r["AwayTeam"])] = {
            "HOME": float(r["p_home"]), "DRAW": float(r["p_draw"]),
            "AWAY": float(r["p_away"])}
    print(f"  gbm blend applied to {len(res)} fixtures at w={w:.2f}"
          + (f"; {skipped} left on Dixon-Coles (not a fitted division)"
             if skipped else ""))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xi", type=float, default=0.0018)
    ap.add_argument("--reg", type=float, default=2.0)
    ap.add_argument("--max-spread", type=float, default=0.06)
    ap.add_argument("--min-depth", type=float, default=500.0)
    ap.add_argument("--min-ev", type=float, default=0.02)
    ap.add_argument("--min-eff-n", type=float, default=15.0,
                    help="minimum time-weighted matches behind BOTH teams' ratings")
    args = ap.parse_args()

    hist = D.load_history()
    hist = hist[hist["FTHG"].notna()].copy()
    today = pd.Timestamp.now().normalize()

    groups = D.country_groups(hist)
    fits: dict[str, M.FitResult] = {}
    rows = []
    pending: list[dict] = []

    for series, (country, price_div) in SERIES_COUNTRY.items():
        fx = fetch_fixtures(series)
        if not fx:
            continue
        divs = groups.get(country)
        if not divs:
            print(f"  {series}: no history for country {country}")
            continue
        sub = hist[hist["Div"].isin(divs)]
        if sub.empty:
            continue
        if country not in fits:
            try:
                fits[country] = M.fit(sub, today, xi=args.xi, reg=args.reg)
            except ValueError as e:
                print(f"  {series}: cannot fit ({e})")
                continue
        fr = fits[country]
        resolver = TeamResolver(fr.teams)
        print(f"\n{series}: {len(fx)} open fixtures, fitted on {country} "
              f"{divs} ({fr.n_matches:,} matches, {len(fr.teams)} teams), "
              f"pricing in {price_div}")

        for f in fx:
            h = resolver.resolve(f["home"])
            a = resolver.resolve(f["away"])
            if h is None or a is None:
                print(f"  unresolved: {f['home']!r} v {f['away']!r}")
                continue
            pred = M.predict(fr, h, a, price_div)
            if pred is None:
                continue
            # Collected rather than priced immediately: the GBM blend is
            # applied to every fixture at once after this loop, because
            # training the booster per fixture would be absurd and its
            # features need the whole history in one frame anyway.
            pending.append({"series": series, "price_div": price_div,
                            "f": f, "h": h, "a": a, "pred": pred})

    blended = _blend_all(pending, hist)

    for rec in pending:
        series, price_div = rec["series"], rec["price_div"]
        f, h, a, pred = rec["f"], rec["h"], rec["a"], rec["pred"]
        model = blended.get((price_div, h, a))
        if model is None:
            model = {"HOME": pred["p_home"], "DRAW": pred["p_draw"],
                     "AWAY": pred["p_away"]}

        asks = [f["legs"][k]["ask"] for k in ("HOME", "DRAW", "AWAY")]
        book_sum = sum(x for x in asks if x is not None)

        for k in ("HOME", "DRAW", "AWAY"):
            leg = f["legs"][k]
            bid, ask = leg["bid"], leg["ask"]
            if bid is None or ask is None:
                continue
            spread = ask - bid
            depth = max(leg["liq"], leg["oi"], orderbook_depth(leg["ticker"]))
            p = model[k]
            # Buy at the ask; profit is (1 - ask) less fee, loss is the ask.
            fee = kalshi_fee(ask)
            ev = p * (1.0 - ask) - (1.0 - p) * ask - fee
            enough_data = pred["eff_n_min"] >= args.min_eff_n
            tradeable = (spread <= args.max_spread
                         and depth >= args.min_depth
                         and enough_data)
            rows.append({
                "series": series, "event": f["event"], "when": str(f["when"])[:16],
                "match": f"{h} v {a}", "leg": k,
                "model": p, "bid": bid, "ask": ask, "spread": spread,
                "depth": depth, "fee": fee, "ev": ev,
                "eff_n_min": pred["eff_n_min"],
                "book_sum": book_sum, "tradeable": tradeable,
                "thin_data": not enough_data,
            })

    if not rows:
        # Non-zero, so auto_update logs this FAILED and the log says why.
        # This returned 0 before, which is how a dead parser stayed invisible
        # for days: the step read "ok", export_tara's latest() fell back to
        # the newest report still on disk, and the board kept publishing
        # two-week-old picks under a fresh timestamp. Producing nothing is a
        # failure — eight major leagues do not all go dark on the same day.
        print("\nno comparable fixtures found — FAILING rather than exiting "
              "clean. Eight leagues cannot all be idle at once, so this is "
              "the feed shape or the parser having changed.")
        return 1

    df = pd.DataFrame(rows)
    out = ROOT / "reports" / f"kalshi_edge_{today.date()}.csv"
    df.to_csv(out, index=False)

    print(f"\n{'=' * 100}\nALL COMPARISONS ({len(df)} legs, "
          f"{df['event'].nunique()} fixtures)\n{'=' * 100}")
    print(f"{'match':<34}{'leg':<6}{'model':>7}{'bid':>6}{'ask':>6}"
          f"{'spr':>6}{'depth':>9}{'EV':>8}  trade")
    for _, r in df.sort_values(["match", "leg"]).iterrows():
        print(f"{r['match'][:33]:<34}{r['leg']:<6}{r['model']:>7.1%}"
              f"{r['bid']:>6.2f}{r['ask']:>6.2f}{r['spread']:>6.2f}"
              f"{r['depth']:>9,.0f}{r['ev']:>+8.1%}  {'yes' if r['tradeable'] else 'no'}")

    print(f"\n  Kalshi three-way ask sums (1.00 = no spread at all):")
    for ev_, g in df.groupby("event"):
        print(f"    {g['match'].iloc[0][:36]:<38} {g['book_sum'].iloc[0]:.3f}")

    thin = df[df["thin_data"] & (df["ev"] >= args.min_ev)]
    if not thin.empty:
        print(f"\n  SUPPRESSED — thin data (eff_n < {args.min_eff_n:.0f} "
              f"weighted matches behind a team's rating):")
        for _, r in thin.sort_values("ev", ascending=False).iterrows():
            print(f"    {r['match'][:33]:<34}{r['leg']:<6} model {r['model']:>6.1%} "
                  f"ask {r['ask']:.2f}  apparent EV {r['ev']:+.1%}  "
                  f"eff_n {r['eff_n_min']:.1f}")
        print("    A promoted club with a handful of games can land a mid-table")
        print("    rating on luck of the draw. These are model error, not edge.")

    good = df[df["tradeable"] & (df["ev"] >= args.min_ev)]
    print(f"\n{'=' * 100}\nPOSITIVE EV AND TRADEABLE (EV >= {args.min_ev:.0%})\n{'=' * 100}")
    if good.empty:
        print("  none")
    else:
        for _, r in good.sort_values("ev", ascending=False).iterrows():
            print(f"  {r['match'][:33]:<34}{r['leg']:<6} model {r['model']:>6.1%} "
                  f"ask {r['ask']:.2f}  EV {r['ev']:+.1%}  depth ${r['depth']:,.0f}"
                  f"  eff_n {r['eff_n_min']:.0f}")
    print(f"\nsaved -> {out}")
    print("\n  Backtest context: this model loses to sharp closing lines by 0.017")
    print("  log loss and betting its disagreements lost 4.6-7.4%. Any EV above")
    print("  is a hypothesis to be logged in the ledger and scored, not a signal.")


if __name__ == "__main__":
    # main()'s return value has to reach the shell, or the guard above is
    # decorative — auto_update decides pass/fail purely on the exit code.
    raise SystemExit(main())
