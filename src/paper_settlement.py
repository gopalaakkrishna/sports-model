"""Paper-only settlement state/health. Never changes an original prediction."""
from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VERSION = "14.7.6"


def pending(row):
    return row.get("outcome") is None and row.get("won") is None and not row.get("voided")


def block(row, code, detail, evidence=None):
    row["settlement_status"] = code
    row["settlement_blocker"] = {"code": code, "detail": detail}
    if evidence:
        row["settlement_blocker"]["evidence"] = evidence
    row["settlement_version"] = VERSION


def market_result(market, yes_outcome, no_outcome, *, selected_no=False):
    """Return a final resolution only; scalar payout is not a refund or W/L."""
    if market.get("status") not in ("finalized", "settled"):
        return None
    result = str(market.get("result", "")).lower()
    evidence = {"source": "kalshi", "ticker": market.get("ticker"),
                "event_ticker": market.get("event_ticker"), "result": result,
                "status": market.get("status")}
    if result == "scalar":
        try:
            payout = float(market["settlement_value_dollars"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(payout) or not 0 <= payout <= 1:
            return None
        evidence["market_yes_payout_dollars"] = payout
        evidence["payout_dollars"] = round(1 - payout, 8) if selected_no else payout
        return "SCALAR", f"Finalized partial payout ${evidence['payout_dollars']:.4f}; excluded from binary W/L", evidence
    if result in ("yes", "no"):
        return (yes_outcome if result == "yes" else no_outcome), f"Kalshi finalized: {result}", evidence
    return None


def atomic_save(path, rows):
    """Same-directory replacement: never truncate the only paper ledger."""
    path = Path(path)
    validate(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as f:
            name = f.name
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        name = None
    finally:
        if name:
            Path(name).unlink(missing_ok=True)


def validate(rows):
    ids = [str(row.get("id")) for row in rows]
    if any(row.get("id") is None for row in rows) or len(ids) != len(set(ids)):
        raise ValueError("Paper ledger has missing or duplicate record IDs; refusing to overwrite evidence")


def load(path):
    path = Path(path)
    if not path.exists():
        return []
    # A corrupt line is a failed load, never permission to drop that record.
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate(rows)
    return rows


def health(rows, now=None):
    now = now or datetime.now(timezone.utc)
    today = now.date().isoformat()
    unresolved = [r for r in rows if pending(r)]
    future = [r for r in unresolved if str(r.get("start") or "")[:10] > today]
    overdue = [r for r in unresolved if str(r.get("start") or "")[:10] < today]
    codes = Counter((r.get("settlement_blocker") or {}).get("code", "unclassified") for r in overdue)
    errors = [r for r in unresolved if r.get("settlement_status") == "source-error"]
    return {"version": VERSION, "status": "attention-required" if overdue or errors else "ok",
            "recorded": len(rows), "binary_settled": sum(r.get("won") is not None and not r.get("voided") for r in rows),
            "non_binary_resolved": sum(r.get("outcome") == "SCALAR" for r in rows),
            "cancelled_resolved": sum(r.get("outcome") == "CANCELLED" for r in rows),
            "pending": len(unresolved), "future_pending": len(future),
            "past_date_pending": len(overdue), "source_errors": len(errors),
            "blockers": dict(codes),
            "attention": [{"id": r.get("id"), "event": r.get("event"), "start": r.get("start"),
                           "blocker": r.get("settlement_blocker") or {"code": "unclassified", "detail": "No verified settlement match"}}
                          for r in overdue[:25]]}
