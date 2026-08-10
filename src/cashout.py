"""When to take profit on a live position — measured, not felt.

Two separate things get confused in the moment:

1. HOW SAFE IS THE LEAD. Answered empirically from 250,866 half-inning states:
   how often does a team leading by N entering inning I actually win?

2. WHAT AM I RISKING TO GAIN. At price p a contract pays 1, so holding risks p
   to gain (1 - p). At 0.86 that is risking 86c to make 14c — about 6:1 against
   — even though the EV is neutral because the market price IS the expected
   value.

The second point is the one that matters and it is not about prediction at all.
An efficient market offers your own expected value back, so there is no edge in
holding OR selling. What changes is the shape: late in a winning position you
carry large downside for small remaining upside. Whether that is worth it is a
variance preference, not a forecast.

The genuine asymmetry is fees. Kalshi charges 0.07 * p * (1-p) per contract,
which is largest at 0.50 and near zero at the extremes. Exiting at 0.86 costs
about 0.8c a contract; exiting at 0.55 costs 1.7c. Holding to settlement costs
nothing. So an exit is cheapest exactly when the position is already won.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEE_RATE = 0.07


def lead_safety(states: pd.DataFrame) -> pd.DataFrame:
    """Win rate for the leading side, by lead size and inning."""
    # "Entering inning N" = the top half, before either side has batted in it.
    d = states[states["half"] == "top"].copy()
    d["lead"] = d["diff"]          # home minus away, going into the inning
    rows = []
    for inning in range(1, 11):
        for lead in range(1, 7):
            sub = d[(d["inning"] == inning) & (d["lead"] == lead)]
            if len(sub) < 40:
                continue
            rows.append({"inning": inning, "lead": lead, "n": len(sub),
                         "home_wins": sub["home_won"].mean()})
            sub2 = d[(d["inning"] == inning) & (d["lead"] == -lead)]
            if len(sub2) >= 40:
                rows.append({"inning": inning, "lead": -lead, "n": len(sub2),
                             "home_wins": sub2["home_won"].mean()})
    return pd.DataFrame(rows)


def advise(price: float, cost_basis: float, contracts: float) -> None:
    fee = FEE_RATE * price * (1 - price) * contracts
    gross = price * contracts
    total_cost = cost_basis * contracts
    print(f"\n  POSITION: {contracts:,.2f} contracts, cost basis {cost_basis:.3f}, "
          f"now {price:.2f}")
    print(f"    sell now   gross ${gross:,.2f} - fee ${fee:,.2f} "
          f"= ${gross - fee:,.2f}   ({gross - fee - total_cost:+,.2f})")
    print(f"    hold & win ${contracts:,.2f}   ({contracts - total_cost:+,.2f})")
    print(f"    hold & lose $0.00   ({-total_cost:+,.2f})")
    print(f"\n    risking ${gross:,.2f} to gain a further "
          f"${contracts - gross:,.2f}")
    ratio = price / (1 - price) if price < 1 else float("inf")
    print(f"    risk:reward on the remaining move = {ratio:.1f} : 1")
    print(f"    exit fee is {fee / max(gross - total_cost, 1e-9):.1%} of the "
          f"profit you would bank")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", type=float, help="current bid")
    ap.add_argument("--cost", type=float, help="your cost basis per contract")
    ap.add_argument("--contracts", type=float, default=100.0)
    args = ap.parse_args()

    st = pd.read_parquet(ROOT / "data" / "raw" / "mlb_inplay_states.parquet")
    tbl = lead_safety(st)

    print("HOW SAFE IS A LEAD — home team win rate, by lead entering each inning")
    print("(from 250,866 half-inning states, 13,752 games)\n")
    piv = tbl[tbl["lead"] > 0].pivot(index="inning", columns="lead",
                                     values="home_wins")
    print("  home leading by:")
    print(piv.round(3).to_string())

    print("\n  A 2-run lead entering the 9th:")
    row = tbl[(tbl["inning"] == 9) & (tbl["lead"] == 2)]
    if len(row):
        r = row.iloc[0]
        print(f"    home wins {r['home_wins']:.1%} of {int(r['n'])} such games")
        print(f"    -> blown roughly 1 time in "
              f"{1 / max(1 - r['home_wins'], 1e-9):.0f}")
    row = tbl[(tbl["inning"] == 9) & (tbl["lead"] == -2)]
    if len(row):
        r = row.iloc[0]
        print(f"    AWAY leading by 2 entering the 9th: away wins "
              f"{1 - r['home_wins']:.1%} of {int(r['n'])}")
        print(f"    -> blown roughly 1 time in "
              f"{1 / max(r['home_wins'], 1e-9):.0f}  "
              f"(the home side still has last at-bat)")

    if args.price and args.cost:
        advise(args.price, args.cost, args.contracts)

    print("\n  THE RULE THAT ACTUALLY HELPS")
    print("  Price IS expected value in an efficient market, so neither holding")
    print("  nor selling has an edge. What changes is shape:")
    print("    at 0.90 you risk 9.0 to gain 1.0")
    print("    at 0.75 you risk 3.0 to gain 1.0")
    print("    at 0.50 you risk 1.0 to gain 1.0")
    print("  If a loss from here would change how you size the next bet, the")
    print("  asymmetry matters more than the probability. Fees do not stop you:")
    print("  exiting at 0.86 costs about 0.8c a contract, the cheapest it ever")
    print("  gets outside settlement.")


if __name__ == "__main__":
    main()
