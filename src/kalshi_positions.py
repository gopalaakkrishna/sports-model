"""Read actual Kalshi activity — positions, fills, settlements.

The ledger currently scores predictions at an assumed 1-unit stake. Real fills
give the prices actually paid and the sizes actually taken, which is what P&L
depends on. Two things this can reveal that a model-only ledger cannot:

  * slippage — the gap between the price I quoted and the price you got
  * whether exposure lines up with stated conviction

Read only. This module contains no order-placement code.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import kalshi_auth as KA

ROOT = Path(__file__).resolve().parents[1]


def _dollars(cents) -> float:
    try:
        return float(cents) / 100.0
    except (TypeError, ValueError):
        return float("nan")


def fetch_fills(limit: int = 500) -> pd.DataFrame:
    rows, cursor = [], None
    while len(rows) < limit:
        params = {"limit": min(200, limit - len(rows))}
        if cursor:
            params["cursor"] = cursor
        j = KA.get("/trade-api/v2/portfolio/fills", params)
        batch = j.get("fills", [])
        rows.extend(batch)
        cursor = j.get("cursor")
        if not cursor or not batch:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in ("yes_price", "no_price"):
        if c in df:
            df[c + "_usd"] = df[c].map(_dollars)
    if "created_time" in df:
        df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce")
    return df


def fetch_positions() -> pd.DataFrame:
    j = KA.get("/trade-api/v2/portfolio/positions", {"limit": 200})
    mk = j.get("market_positions", [])
    return pd.DataFrame(mk) if mk else pd.DataFrame()


def series_of(ticker: str) -> str:
    t = str(ticker)
    return t.split("-")[0] if "-" in t else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    try:
        bal = KA.get("/trade-api/v2/portfolio/balance")
    except KA.CredentialsMissing as e:
        print(f"not configured: {e}")
        return
    print(f"balance: ${_dollars(bal.get('balance')):,.2f}\n")

    fills = fetch_fills(args.limit)
    if fills.empty:
        print("no fills found")
    else:
        print(f"FILLS: {len(fills)}")
        if "created_time" in fills:
            print(f"  from {fills['created_time'].min()} to {fills['created_time'].max()}")
        by_series = defaultdict(lambda: {"n": 0, "contracts": 0})
        for _, r in fills.iterrows():
            s = series_of(r.get("ticker", ""))
            by_series[s]["n"] += 1
            by_series[s]["contracts"] += int(r.get("count") or 0)
        print(f"\n  {'series':<32}{'fills':>7}{'contracts':>11}")
        for s, v in sorted(by_series.items(), key=lambda z: -z[1]["contracts"]):
            print(f"  {s:<32}{v['n']:>7}{v['contracts']:>11,}")

        cols = [c for c in ("created_time", "ticker", "side", "action",
                            "count", "yes_price_usd") if c in fills.columns]
        print(f"\n  most recent fills:")
        print(fills[cols].head(12).to_string(index=False))

    pos = fetch_positions()
    if pos.empty:
        print("\nno open positions")
    else:
        live = pos[pos.get("position", 0) != 0] if "position" in pos else pos
        print(f"\nPOSITIONS: {len(pos)} rows, {len(live)} with non-zero exposure")
        cols = [c for c in ("ticker", "position", "market_exposure",
                            "realized_pnl", "total_traded") if c in pos.columns]
        if cols and not live.empty:
            show = live[cols].copy()
            for c in ("market_exposure", "realized_pnl", "total_traded"):
                if c in show:
                    show[c] = show[c].map(_dollars)
            print(show.head(20).to_string(index=False))
        if "realized_pnl" in pos:
            print(f"\n  total realized P&L: ${pos['realized_pnl'].map(_dollars).sum():,.2f}")

    outp = ROOT / "reports" / "kalshi_activity.csv"
    if not fills.empty:
        fills.to_csv(outp, index=False)
        print(f"\nsaved fills -> {outp}")


if __name__ == "__main__":
    main()
