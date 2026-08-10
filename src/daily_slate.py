"""Daily recommendations across every sport the project covers.

    python src/daily_slate.py              # today's slate, top picks
    python src/daily_slate.py --n 5        # ask for 5
    python src/daily_slate.py --log        # also write them to the ledger

Every candidate must clear the same screens before it can be recommended:

  * a two-sided quote, spread within limits, real orderbook depth
  * both teams above a minimum effective sample (no thin-data phantoms)
  * expected value computed at the ASK and net of fees, never at the mid

Candidates are then sorted and assigned a confidence tier. The tiers exist
because the backtests are unambiguous about where this model is reliable:

  A  model and market agree closely — the model's most accurate zone
     (0-2.8% disagreement: log loss 0.9938 vs market 0.9923, effectively level)
  B  moderate disagreement, some evidence either way
  C  large disagreement — measured to be MODEL ERROR far more often than edge
     (11.6%+ disagreement: model 1.0373 vs market 0.9641)

That ordering is deliberate and will look backwards to anyone expecting a tip
sheet: the biggest "edges" are ranked LAST, because that is what the data says
they are worth. A day with no A or B candidates reports none rather than
promoting C picks to fill a quota — a fixed number of daily selections would
guarantee output on days when the honest answer is that nothing stands out.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def tier(disagreement: float) -> str:
    if disagreement < 0.028:
        return "A"
    if disagreement < 0.068:
        return "B"
    return "C"


TIER_NOTE = {
    "A": "model agrees with market — most reliable zone",
    "B": "moderate disagreement — mixed evidence",
    "C": "large disagreement — measured to be model error more often than edge",
}


def run(script: str, args: list[str]) -> str:
    try:
        r = subprocess.run([PY, "-u", str(Path(__file__).parent / script)] + args,
                           capture_output=True, text=True, timeout=1800)
        return r.stdout
    except Exception as e:
        return f"__ERROR__ {type(e).__name__}: {e}"


def collect_soccer() -> pd.DataFrame:
    """Kalshi soccer edges, already screened by kalshi_edge.py."""
    out = run("kalshi_edge.py", [])
    today = pd.Timestamp.now().normalize().date()
    p = ROOT / "reports" / f"kalshi_edge_{today}.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    if d.empty:
        return d
    d = d[d["tradeable"]].copy()
    d["sport"] = "soccer"
    d["market_prob"] = d["ask"]
    d["disagreement"] = (d["model"] - d["ask"]).abs()
    d["selection"] = d["match"] + " — " + d["leg"]
    return d[["sport", "selection", "model", "market_prob", "ask", "ev",
              "depth", "disagreement", "when"]]


def collect_mlb() -> pd.DataFrame:
    out = run("mlb_predict.py", [])
    today = pd.Timestamp.now().normalize().date()
    p = ROOT / "reports" / f"mlb_predictions_{today}.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    if d.empty or "tradeable" not in d:
        return pd.DataFrame()
    d = d[d["tradeable"] & np.isfinite(d["ask"])].copy()
    d["sport"] = "mlb"
    d["market_prob"] = d["ask"]
    d["disagreement"] = (d["model"] - d["ask"]).abs()
    d["selection"] = d["match"] + " — " + d["team"]
    d["when"] = d["date"].astype(str)
    return d[["sport", "selection", "model", "market_prob", "ask", "ev",
              "depth", "disagreement", "when"]]


def collect_wnba() -> pd.DataFrame:
    out = run("wnba_kalshi.py", [])
    today = pd.Timestamp.now().normalize().date()
    p = ROOT / "reports" / f"wnba_kalshi_{today}.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    if d.empty or "tradeable" not in d:
        return pd.DataFrame()
    d = d[d["tradeable"]].copy()
    d["sport"] = "wnba"
    # Normalise the two asks per game so "market" is a probability, not a price.
    d["key"] = d["date"].astype(str) + "|" + d["match"]
    d["market_prob"] = d["ask"] / d.groupby("key")["ask"].transform("sum")
    d["disagreement"] = (d["model"] - d["market_prob"]).abs()
    d["selection"] = d["match"] + " — " + d["team"]
    d["when"] = d["date"].astype(str)
    return d[["sport", "selection", "model", "market_prob", "ask", "ev",
              "depth", "disagreement", "when"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="how many to surface")
    ap.add_argument("--min-ev", type=float, default=0.0)
    ap.add_argument("--log", action="store_true",
                    help="write the surfaced picks to the ledger")
    ap.add_argument("--include-c", action="store_true",
                    help="include large-disagreement (tier C) candidates")
    args = ap.parse_args()

    today = pd.Timestamp.now().normalize().date()
    print(f"DAILY SLATE — {today}\n{'=' * 78}")

    frames = []
    for name, fn in (("soccer", collect_soccer), ("mlb", collect_mlb),
                     ("wnba", collect_wnba)):
        try:
            d = fn()
            print(f"  {name:<8} {len(d)} screened candidate(s)")
            if not d.empty:
                frames.append(d)
        except Exception as e:
            print(f"  {name:<8} failed: {type(e).__name__}: {str(e)[:90]}")

    if not frames:
        print("\nNo candidates cleared the screens today.")
        print("Usually means no fixtures, or no market met the liquidity bar.")
        return

    d = pd.concat(frames, ignore_index=True)
    d["tier"] = d["disagreement"].map(tier)
    d = d[d["ev"] >= args.min_ev]
    if not args.include_c:
        d = d[d["tier"] != "C"]

    if d.empty:
        print(f"\nNothing in tier A or B today with EV >= {args.min_ev:.0%}.")
        print("Reporting none rather than promoting large-disagreement picks.")
        print("Use --include-c to see them anyway, knowing what they are.")
        return

    order = {"A": 0, "B": 1, "C": 2}
    d["_o"] = d["tier"].map(order)
    d = d.sort_values(["_o", "ev"], ascending=[True, False]).head(args.n)

    print(f"\n{'#':<3}{'tier':<6}{'sport':<8}{'selection':<44}"
          f"{'model':>7}{'mkt':>7}{'EV':>8}")
    print("-" * 84)
    for i, (_, r) in enumerate(d.iterrows(), 1):
        print(f"{i:<3}{r['tier']:<6}{r['sport']:<8}{str(r['selection'])[:43]:<44}"
              f"{r['model']:>7.1%}{r['market_prob']:>7.1%}{r['ev']:>+8.1%}")

    print(f"\n  tiers present:")
    for t in sorted(d["tier"].unique()):
        print(f"    {t}: {TIER_NOTE[t]}")

    if args.log:
        print("\nlogging to ledger...")
        for _, r in d.iterrows():
            subprocess.run([
                PY, str(Path(__file__).parent / "ledger.py"), "lock",
                "--sport", str(r["sport"]),
                "--event", str(r["selection"]),
                "--market", "1X2" if r["sport"] == "soccer" else "ML",
                "--pick", "MODEL",
                "--model-prob", f"{r['model']:.4f}",
                "--market-prob", f"{r['market_prob']:.4f}",
                "--venue", "kalshi",
                "--notes", f"daily slate tier {r['tier']}",
            ], capture_output=True)
        print(f"  logged {len(d)}")

    print(f"\n  Measured context: this model is 0.017 log loss behind sharp")
    print("  closing lines in soccer and at parity in MLB. Tier C selections")
    print("  lost 4.6-7.4% flat-stake in backtest. These are the best available")
    print("  candidates, which is not the same as a positive expectation.")


if __name__ == "__main__":
    main()
