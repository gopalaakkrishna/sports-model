"""Watch a live position and print the hold-vs-exit maths as the price moves.

This does not tell you to sell. It shows the three numbers that get lost in the
moment of watching a game:

  1. what the lead is historically worth, from the in-play state data
  2. what you are risking to gain the remainder
  3. what the exit costs right now

The reason a watcher is worth having: the numbers move fastest exactly when you
are least able to compute them. A two-run lead going into the ninth is worth
93%, but if you are watching the ninth you are not looking that up.

Poll interval defaults to 30s. Kalshi rate limits, and a live game does not need
sub-minute resolution.

    python inplay_watch.py --ticker KXMLBGAME-26AUG06CHCTOR-TOR \\
        --cost 0.5685 --contracts 209
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from cashout import FEE_RATE, lead_safety

ROOT = Path(__file__).resolve().parents[1]
K = "https://api.elections.kalshi.com/trade-api/v2"
ET = ZoneInfo("America/New_York")


def quote(ticker: str) -> tuple[float | None, float | None]:
    """(bid, ask) in dollars, or (None, None) if unavailable.

    A rate-limit or transport failure must never be read as 'no market' — that
    mistake once turned 926 games into 24. Returning None keeps the caller
    honest about the difference between 'no price' and 'no answer'.
    """
    for attempt in range(4):
        try:
            r = requests.get(f"{K}/markets/{ticker}", timeout=20)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            m = r.json().get("market", {})
            return (float(m.get("yes_bid_dollars")),
                    float(m.get("yes_ask_dollars")))
        except (requests.RequestException, TypeError, ValueError):
            time.sleep(1.5 * (attempt + 1))
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, help="Kalshi market ticker")
    ap.add_argument("--cost", type=float, required=True, help="cost basis / contract")
    ap.add_argument("--contracts", type=float, required=True)
    ap.add_argument("--every", type=int, default=30, help="poll seconds")
    ap.add_argument("--alert-ratio", type=float, default=4.0,
                    help="flag when risk:reward on the remaining move exceeds this")
    args = ap.parse_args()

    st = pd.read_parquet(ROOT / "data" / "raw" / "mlb_inplay_states.parquet")
    safety = lead_safety(st)

    total_cost = args.cost * args.contracts
    print(f"watching {args.ticker}")
    print(f"  {args.contracts:,.0f} contracts @ {args.cost:.3f} "
          f"= ${total_cost:,.2f} at risk")
    print(f"  polling every {args.every}s — Ctrl-C to stop\n")
    print(f"{'time (ET)':<11}{'bid':>7}{'sell net':>11}{'P&L':>10}"
          f"{'at risk':>10}{'to gain':>10}{'ratio':>8}  ")
    print("=" * 70)

    last = None
    while True:
        bid, ask = quote(args.ticker)
        now = datetime.now(ET)
        if bid is None:
            print(f"{now:%I:%M:%S%p}  no quote (rate limit or closed) — retrying")
            time.sleep(args.every)
            continue

        gross = bid * args.contracts
        fee = FEE_RATE * bid * (1 - bid) * args.contracts
        net = gross - fee
        pnl = net - total_cost
        to_gain = args.contracts - gross
        ratio = gross / to_gain if to_gain > 1e-9 else float("inf")

        flag = ""
        if ratio >= args.alert_ratio:
            flag = f"  <-- risking {ratio:.1f}x the remaining upside"
        if last is not None and abs(bid - last) >= 0.05:
            flag += f"  [moved {bid - last:+.2f}]"

        print(f"{now:%I:%M:%S%p}{bid:>7.2f}{net:>11,.2f}{pnl:>+10,.2f}"
              f"{gross:>10,.2f}{to_gain:>10,.2f}{ratio:>8.1f}{flag}")
        last = bid

        if bid >= 0.99 or bid <= 0.01:
            print("\n  price is at the boundary — the market considers this decided.")
            print("  Holding to settlement now costs no fee and gains what is left.")
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
