"""Price Kalshi's soccer TOTAL GOALS markets with the model we already have.

WHY THIS EXISTS

Dixon-Coles does not model "who wins" — it models two goal rates and a
low-score correlation. The winner probability is a DERIVED quantity, obtained
by summing one triangle of the score matrix. Totals come off the same matrix.
`model.predict()` has always returned p_over25; nothing consumed it, so we
were pricing one market per fixture while Kalshi listed roughly seven.

That is the entire case for this file: more picks, on the same games, from a
model already fitted, with no new data source. Every total market measured
quotes a 1c spread, the same as the winner market.

WHAT THE RESEARCH SAYS, INCLUDING THE PARTS THAT CONSTRAIN IT

From totals_backtest.py over 24,898 walk-forward matches:

  * The model beats a constant base rate on over 2.5 (+0.0067 log loss). It
    carries real information about goals.
  * It does NOT beat the closing line. Blend weight 0.00, market sharper by
    +0.0121 with the CI entirely above zero — the same verdict the winner
    market got. Disagreement is model error: at |gap|>10% the model scores
    0.7418 against the market's 0.6856. So no edge, and nothing here is
    priced off disagreement.
  * Raw probabilities are overconfident in both tails. Platt scaling on major
    divisions fixes the OVER side (stated 64.7% / actual 64.1% held out).
  * It does NOT fix the UNDER side (stated 62.9% / actual 51.0%). Unders are
    excluded entirely — an 11-point overstatement is not a pick, it is a loss
    with a confident label on it.
  * BTTS showed NO skill at all (-0.0002 against a constant). Never priced.

So this ships exactly one thing: calibrated OVER picks, in major divisions,
where the model AGREES with Kalshi — the same discipline the winner board
uses, because agreement is the only regime that has ever held up.

These are PAPER picks in their own lane. They also serve as a control: if the
winner board's floor edge is real rather than luck, the same filter should
reproduce it here. If it does not, that is informative about the winner board.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
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
CALIB = ROOT / "data" / "processed" / "totals_calibration.json"

# Majors only. The standing instruction is that the board covers major leagues
# in major sports — games that can actually be watched — so research and
# pricing are scoped identically. Each entry maps a Kalshi totals series to the
# country group to FIT on and the division to PRICE in, exactly as kalshi_edge
# does for the winner market.
SERIES_COUNTRY = {
    "KXEPLTOTAL": ("England", "E0"),
    "KXLALIGATOTAL": ("Spain", "SP1"),
    "KXSERIEATOTAL": ("Italy", "I1"),
    "KXBUNDESLIGATOTAL": ("Germany", "D1"),
    "KXLIGUE1TOTAL": ("France", "F1"),
    "KXMLSTOTAL": ("USA", "USA:MLS"),
    "KXLIGAMXTOTAL": ("Mexico", "MEX:Liga MX"),
}

# Only the 2.5 line is backed by evidence. The calibration was fitted on
# over-2.5 outcomes, and a Platt fit does not transfer to a different
# threshold: 0.5 and 1.5 sit far out in a tail where the model measured worst,
# and 4.5+ are the mirror image. Pricing them would extrapolate a correction
# validated nowhere near those lines.
LINE = 2.5

MIN_CONVICTION = 0.60   # matches the winner board's floor
ALIGNED_PTS = 0.03      # |model - market| within 3 points == "agrees"
MAX_SPREAD = 0.04


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_calibration():
    if not CALIB.exists():
        return None
    j = json.loads(CALIB.read_text(encoding="utf-8"))
    a, b = j.get("a"), j.get("b")
    return (float(a), float(b)) if a is not None and b is not None else None


def apply_calibration(p: float, ab) -> float:
    if ab is None:
        return p
    a, b = ab
    p = min(max(p, 1e-6), 1 - 1e-6)
    z = np.log(p / (1 - p))
    return float(1.0 / (1.0 + np.exp(-(a * z + b))))


def fetch_events(series: str) -> dict:
    """event_ticker -> {home, away}.

    The markets endpoint titles a total as "Will over 2.5 goals be scored?"
    with no teams in it; only the events endpoint carries them.
    """
    r = requests.get(f"{KALSHI}/events",
                     params={"series_ticker": series, "status": "open",
                             "limit": 200}, timeout=60)
    if r.status_code != 200:
        return {}
    out = {}
    for e in r.json().get("events", []):
        m = re.match(r"^(.*?)\s+vs\.?\s+(.*?)\s*:\s*Total Goals\s*$",
                     str(e.get("title", "")), re.I)
        if m:
            out[str(e.get("event_ticker"))] = {"home": m.group(1).strip(),
                                               "away": m.group(2).strip()}
    return out


def fetch_lines(series: str) -> dict:
    """event_ticker -> the over-2.5 market for that fixture."""
    r = requests.get(f"{KALSHI}/markets",
                     params={"series_ticker": series, "status": "open",
                             "limit": 500}, timeout=60)
    if r.status_code != 200:
        return {}
    out = {}
    for m in r.json().get("markets", []):
        t = str(m.get("yes_sub_title") or m.get("title") or "")
        hit = re.search(r"over\s+([0-9.]+)\s+goals", t, re.I)
        if not hit or abs(float(hit.group(1)) - LINE) > 1e-9:
            continue
        bid, ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
        if bid is None or ask is None or not (0 < bid <= ask < 1):
            continue
        out[str(m.get("event_ticker"))] = {
            "ticker": m.get("ticker"), "bid": bid, "ask": ask,
            "when": m.get("occurrence_datetime"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xi", type=float, default=0.0018)
    ap.add_argument("--reg", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ab = load_calibration()
    if ab is None:
        print("!! no totals calibration — refusing to price uncalibrated.")
        return 1
    print(f"calibration a={ab[0]:.4f} b={ab[1]:.4f}")

    hist = D.load_history()
    hist = hist[hist["FTHG"].notna()].copy()
    today = pd.Timestamp.now().normalize()
    groups = D.country_groups(hist)
    fits = {}
    rows = []

    for series, (country, price_div) in SERIES_COUNTRY.items():
        events, lines = fetch_events(series), fetch_lines(series)
        if not events or not lines:
            continue
        divs = groups.get(country)
        if not divs:
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
        res = TeamResolver(fr.teams)
        n_ok = 0

        for ev, teams in events.items():
            mk = lines.get(ev)
            if mk is None:
                continue
            h, a = res.resolve(teams["home"]), res.resolve(teams["away"])
            if h is None or a is None:
                continue
            pred = M.predict(fr, h, a, price_div)
            if pred is None:
                continue
            raw = float(pred["p_over25"])
            model = apply_calibration(raw, ab)
            mkt = mk["ask"]
            spread = mk["ask"] - mk["bid"]
            gap = model - mkt
            rows.append({
                "series": series, "league": price_div, "event": ev,
                "match": f"{h} v {a}", "line": LINE,
                "model_raw": round(raw, 4), "model": round(model, 4),
                "bid": mk["bid"], "ask": mk["ask"],
                "spread": round(spread, 4), "gap": round(gap, 4),
                "when": mk["when"], "ticker": mk["ticker"],
                # A pick only where the model AGREES and clears the floor.
                # Disagreement measured as model error on this market exactly
                # as on the winner market, so it is never traded on.
                "aligned": bool(abs(gap) <= ALIGNED_PTS),
                "tradeable": bool(spread <= MAX_SPREAD),
                "pick": bool(abs(gap) <= ALIGNED_PTS
                             and model >= MIN_CONVICTION
                             and spread <= MAX_SPREAD),
            })
            n_ok += 1
        print(f"  {series:<22} {n_ok:>3} fixtures priced "
              f"({len(events)} events, {len(lines)} lines)")

    if not rows:
        print("no totals fixtures priced")
        return 0

    df = pd.DataFrame(rows).sort_values(["when", "match"])
    out = Path(args.out) if args.out else (
        ROOT / "reports" / f"totals_{today.date()}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    picks = df[df["pick"]]
    print(f"\n{len(df)} fixtures priced, {int(df['aligned'].sum())} aligned, "
          f"{len(picks)} clear the {MIN_CONVICTION:.0%} floor")
    if len(picks):
        print(f"\n  {'match':<38}{'line':>6}{'model':>8}{'ask':>7}{'gap':>8}")
        for _, r in picks.iterrows():
            print(f"  {str(r['match'])[:36]:<38}O{r['line']:<5}"
                  f"{r['model']:>8.0%}{r['ask']:>7.2f}{r['gap']:>+8.1%}")
    print(f"\nwrote {out}")
    print("\nPAPER ONLY. The model does not beat the closing line on this "
          "market (blend weight 0.00); these exist to test whether the "
          "aligned+floor filter reproduces the winner board's record.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
