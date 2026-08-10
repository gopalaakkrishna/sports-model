"""Price a Kalshi multi-leg parlay against the model and against the legs' own markets.

A parlay is only good value if its price is below the product of the leg
probabilities. Two subtleties matter and are handled explicitly:

* **You buy at the ask on each leg**, so the market-implied fair value uses ask
  prices, normalised per fixture to strip the three-way overround.
* **Legs are not independent.** Six "home team wins" in one competition on one
  night share a common driver — if the league-strength assumption behind them is
  wrong, they fail together. Positive correlation raises the true joint
  probability above the naive product, so a correlation sensitivity is shown
  rather than assumed away.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import requests

K = "https://api.elections.kalshi.com/trade-api/v2"

# The six legs, with the model's probability for each (connected MLS/LigaMX fit).
LEGS = [
    ("KXLEAGUESCUPGAME-26AUG05DALQUE-DAL", "FC Dallas", 0.58),
    ("KXLEAGUESCUPGAME-26AUG05LAFCCDG-LAFC", "LAFC", 0.53),
    ("KXLEAGUESCUPGAME-26AUG05MIAASL-MIA", "Inter Miami", 0.70),
    ("KXLEAGUESCUPGAME-26AUG05MONORL-ORL", "Orlando City", 0.42),
    ("KXLEAGUESCUPGAME-26AUG05NSHLEO-NSH", "Nashville SC", 0.61),
    ("KXLEAGUESCUPGAME-26AUG05TOLSEA-TOL", "Toluca", 0.45),
]

STAKE = 18.666480
FEE = 1.279320
CONTRACTS = 888.88
PRICE = 0.0210


def leg_market(ticker: str):
    r = requests.get(f"{K}/markets/{ticker}", timeout=45)
    if r.status_code != 200:
        return None
    return r.json().get("market", {})


def event_normaliser(event_ticker: str) -> float:
    """Sum of the three asks in a fixture, for stripping the overround."""
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


def main():
    print(f"{'leg':<16}{'model':>8}{'bid':>7}{'ask':>7}{'mkt(norm)':>11}  status")
    model_p, mkt_p = [], []
    for ticker, name, mp in LEGS:
        m = leg_market(ticker)
        if not m:
            print(f"{name:<16}{mp:>8.0%}   (market unavailable)")
            model_p.append(mp)
            continue
        bid = float(m.get("yes_bid_dollars") or 0)
        ask = float(m.get("yes_ask_dollars") or 0)
        norm = event_normaliser(m.get("event_ticker", ""))
        implied = ask / norm if norm > 0 else ask
        status = m.get("status")
        res = m.get("result") or ""
        model_p.append(mp)
        mkt_p.append(implied)
        print(f"{name:<16}{mp:>8.0%}{bid:>7.2f}{ask:>7.2f}{implied:>11.1%}  "
              f"{status}{' -> ' + res if res else ''}")

    mprod = float(np.prod(model_p))
    kprod = float(np.prod(mkt_p)) if len(mkt_p) == len(LEGS) else float("nan")

    print(f"\n{'=' * 66}\nJOINT PROBABILITY (assuming independence)\n{'=' * 66}")
    print(f"  model product   {mprod:.3%}")
    if np.isfinite(kprod):
        print(f"  market product  {kprod:.3%}")
    print(f"  price paid      {PRICE:.3%}")

    total_cost = STAKE + FEE
    payout = CONTRACTS * 1.0
    print(f"\n{'=' * 66}\nECONOMICS\n{'=' * 66}")
    print(f"  contracts       {CONTRACTS:,.2f}")
    print(f"  stake           ${STAKE:,.2f}")
    print(f"  fee             ${FEE:,.2f}   ({FEE/STAKE:.1%} of stake)")
    print(f"  total cost      ${total_cost:,.2f}")
    print(f"  payout if hit   ${payout:,.2f}")
    print(f"  breakeven prob  {total_cost/payout:.3%}")

    print(f"\n  EV at model probability   "
          f"${mprod * payout - total_cost:+,.2f}")
    if np.isfinite(kprod):
        print(f"  EV at market probability  "
              f"${kprod * payout - total_cost:+,.2f}")

    print(f"\n{'=' * 66}\nCORRELATION SENSITIVITY\n{'=' * 66}")
    print("  Legs share a common driver (mostly MLS sides at home). Modelling")
    print("  that as a shared factor lifting the joint probability above the")
    print("  independent product:\n")
    print(f"    {'uplift':<12}{'joint':>10}{'EV (model)':>14}{'EV (market)':>14}")
    for lift in (1.0, 1.25, 1.5, 2.0):
        jm = min(mprod * lift, 1.0)
        jk = min(kprod * lift, 1.0) if np.isfinite(kprod) else float("nan")
        evk = f"${jk * payout - total_cost:+,.2f}" if np.isfinite(jk) else "n/a"
        print(f"    x{lift:<11.2f}{jm:>10.3%}{f'${jm * payout - total_cost:+,.2f}':>14}"
              f"{evk:>14}")


if __name__ == "__main__":
    main()
