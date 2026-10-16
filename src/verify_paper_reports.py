"""Read-only, independent checks of published paper-report denominators.

Run after export and before publication. This does not settle, lock, reprice,
write ledgers or contact any account.
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def expected_metrics(rows, price_field="ask"):
    binary = [r for r in rows if isinstance(r.get("won"), bool) and not r.get("voided")]
    priced = []
    for row in binary:
        value = row.get(price_field)
        if value is None or isinstance(value, bool):
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and 0 < value < 1:
            priced.append((row, value))
    wins = sum(r["won"] for r in binary)
    priced_wins = sum(r["won"] for r, _ in priced)
    n, pn = len(binary), len(priced)
    costs = [p + .07 * p * (1 - p) for _, p in priced]
    rate = priced_wins / pn if pn else None
    breakeven = sum(costs) / pn if pn else None
    return {
        "n_total": len(rows), "n_settled": n, "wins": wins, "losses": n - wins,
        "win_rate": wins / n if n else None,
        "priced_n": pn, "unpriced_n": n - pn, "priced_wins": priced_wins,
        "priced_losses": pn - priced_wins, "priced_win_rate": rate,
        "breakeven": breakeven,
        "edge": rate - breakeven if pn else None,
        "units": sum((1 if r["won"] else 0) - cost for (r, _), cost in zip(priced, costs)) if pn else None,
    }


def verify_summary(summary, rows, label="paper", price_field="ask"):
    expected = expected_metrics(rows, price_field)
    if expected["n_settled"] == 0:
        # Empty/pending-only totals may legitimately omit performance fields.
        if summary is None:
            return
        if not isinstance(summary, dict):
            raise ValueError(f"{label}: report is not an object")
        for key, value in expected.items():
            if key in summary and summary[key] != value:
                raise ValueError(f"{label}.{key}: empty record mismatch")
        return
    if not isinstance(summary, dict):
        raise ValueError(f"{label}: settled rows have no published report")
    for key, value in expected.items():
        actual = summary.get(key)
        if key in {"win_rate", "priced_win_rate", "breakeven", "edge", "units"}:
            valid = actual is None if value is None else (
                not isinstance(actual, bool) and isinstance(actual, (int, float))
                and math.isfinite(actual) and abs(actual - value) <= .000051)
        else:
            valid = not isinstance(actual, bool) and actual == value
        if not valid:
            raise ValueError(f"{label}.{key}: published {actual!r}, independent expectation {value!r}")


def load_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r.get("id") for r in rows]
    if any(i is None for i in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path.name}: missing or duplicate record IDs")
    return rows


def verify_board(board, open_rows, totals_rows):
    verify_summary(board.get("open_lane"), open_rows, "open_lane")
    verify_summary((board.get("totals") or {}).get("paper_record"), totals_rows, "totals.paper_record", "market_prob")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, default=ROOT / "public/sports.json")
    args = parser.parse_args()
    board = json.loads(args.board.read_text(encoding="utf-8"))
    verify_board(board, load_rows(ROOT / "data/processed/open_ledger.jsonl"),
                 load_rows(ROOT / "data/processed/totals_ledger.jsonl"))
    print("Published paper reports verified against canonical price-matched rows; no records changed.")


if __name__ == "__main__":
    main()
