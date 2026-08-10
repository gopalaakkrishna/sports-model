"""Decode any Kalshi multi-leg parlay and price it live.

Give it a market ticker; it reads the selected legs, pulls each leg's current
market state, and reports the joint probability against what was paid.

Legs already decided are treated as 1 or 0 rather than as a probability, so the
figure reported is the CURRENT chance from here — not the chance at purchase.
"""

from __future__ import annotations

import argparse

import numpy as np
import requests

K = "https://api.elections.kalshi.com/trade-api/v2"


def market(ticker: str) -> dict:
    r = requests.get(f"{K}/markets/{ticker}", timeout=45)
    return r.json().get("market", {}) if r.status_code == 200 else {}


def event_norm(event_ticker: str) -> float:
    r = requests.get(f"{K}/markets", params={"event_ticker": event_ticker,
                                             "limit": 10}, timeout=45)
    if r.status_code != 200:
        return 1.0
    tot = 0.0
    for m in r.json().get("markets", []):
        try:
            tot += float(m.get("yes_ask_dollars") or 0)
        except (TypeError, ValueError):
            pass
    return tot if tot > 0 else 1.0


def leg_probability(m: dict) -> tuple[float, str]:
    """Current probability for a leg, and a human-readable state."""
    res = str(m.get("result") or "")
    status = str(m.get("status") or "")
    if res == "yes":
        return 1.0, "WON"
    if res == "no":
        return 0.0, "LOST"
    try:
        bid = float(m.get("yes_bid_dollars"))
        ask = float(m.get("yes_ask_dollars"))
    except (TypeError, ValueError):
        return float("nan"), status
    mid = (bid + ask) / 2
    if status == "inactive":
        # Game over, settlement pending; the price is effectively the result.
        if mid >= 0.95:
            return 1.0, "won (pending)"
        if mid <= 0.05:
            return 0.0, "lost (pending)"
    return mid, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--cost", type=float, default=None,
                    help="total cost including fee")
    ap.add_argument("--contracts", type=float, default=None)
    args = ap.parse_args()

    m = market(args.ticker)
    if not m:
        print("market not found")
        return
    legs = m.get("mve_selected_legs") or []
    price = float(m.get("last_price_dollars") or 0)
    print(f"parlay: {args.ticker}")
    print(f"  legs: {len(legs)}   last price: {price:.4f}")
    print(f"  closes: {m.get('close_time')}\n")

    joint = 1.0
    dead = False
    print(f"  {'leg':<34}{'prob':>9}  state")
    for l in legs:
        lm = market(l.get("market_ticker", ""))
        if not lm:
            print(f"  {l.get('market_ticker','?'):<34}{'?':>9}  unavailable")
            continue
        p, state = leg_probability(lm)
        norm = event_norm(lm.get("event_ticker", ""))
        shown = p if p in (0.0, 1.0) else (p / norm if norm > 0 else p)
        title = f"{lm.get('title','')} [{lm.get('yes_sub_title','')}]"
        print(f"  {title[:33]:<34}{shown:>9.1%}  {state}")
        joint *= shown
        if shown == 0.0:
            dead = True

    print(f"\n  {'=' * 52}")
    if dead:
        print("  PARLAY IS DEAD — at least one leg has lost.")
    print(f"  current joint probability: {joint:.3%}")
    if args.contracts:
        payout = args.contracts
        print(f"  payout if it hits:         ${payout:,.2f}")
        print(f"  current expected value:    ${joint * payout:,.2f}")
        if args.cost:
            print(f"  paid:                      ${args.cost:,.2f}")
            print(f"  (sunk — the question is only what it is worth from here)")


if __name__ == "__main__":
    main()
