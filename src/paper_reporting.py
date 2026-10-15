"""One price-matched denominator for paper edge, fee and return reporting.

These are hypothetical flat one-contract returns, never exchange fill P&L.
Scalar/cancelled outcomes and missing original prices cannot become W/L or cost.
"""
import math


def paper_performance(rows, price_field="ask"):
    done = [r for r in rows if isinstance(r.get("won"), bool) and not r.get("voided")]
    priced = []
    for row in done:
        raw = row.get(price_field)
        if raw is None or isinstance(raw, bool):
            continue
        try:
            price = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and 0 < price < 1:
            priced.append((row, price))
    wins = sum(r["won"] for r in done)
    pw = sum(r["won"] for r, _ in priced)
    n, pn = len(done), len(priced)
    cost = sum(p + .07 * p * (1 - p) for _, p in priced)
    return {
        "n_total": len(rows), "n_settled": n, "wins": wins, "losses": n - wins,
        "win_rate": wins / n if n else None,
        "priced_n": pn, "unpriced_n": n - pn, "priced_wins": pw,
        "priced_losses": pn - pw, "priced_win_rate": pw / pn if pn else None,
        "breakeven": cost / pn if pn else None,
        "edge": (pw - cost) / pn if pn else None,
        "units": pw - cost if pn else None,
        "return_basis": "Original captured price; flat one-contract estimate; quadratic fee estimate; not actual fills.",
    }
