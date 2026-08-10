"""Mirror the prediction ledger to Supabase so the record is not tied to one machine.

WHY THIS EXISTS

The ledger is the prediction record — every locked pick and every settlement.
Until now it lived in exactly one place: a file on one laptop. Committing it to
git (see auto_update.publish_self_state) made it durable, but git is a poor
shared-state medium for something several writers touch: a scheduled GitHub
Actions run and a local run can both settle games, and whoever pushes second
hits a conflict or clobbers the other.

Supabase is already this project's shared-state layer — the BTC side keeps its
own trade record at `memory/taraCallLog` in the same table. The sports ledger
gets the same treatment at `memory/sportsLedger`, so every device and every CI
run reads and writes one converging record.

MERGE, NEVER REPLACE

The single most important property here. Two runners can each hold rows the
other has not seen, so a plain overwrite silently destroys predictions —
exactly the kind of quiet data loss that makes a record untrustworthy. Every
sync is a union keyed on `id`, and on conflict the row that is SETTLED wins
(an unsettled row is always the older view of the same prediction). The BTC
side arrived at the same rule for the same reason.

SIZE

At ~30KB this is trivial next to `memory/taraCallLog` (~2.6MB). That matters:
the BTC log had to be excluded from Supabase realtime because rebroadcasting
2.6MB on every change cost ~157MB/day of egress and twice blew the project's
quota. Thirty kilobytes can ride realtime without concern.

CREDENTIALS

Reads SUPABASE_URL / SUPABASE_ANON_KEY from the environment. The anon key is
not a secret in any meaningful sense — it is already embedded in tara-app's
shipped JavaScript, so anyone who can load the site already has it. The real
access boundary is RLS, not the key.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "processed" / "ledger.jsonl"
DOC_PATH = "memory/sportsLedger"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")


class SupabaseUnavailable(RuntimeError):
    """Raised when Supabase is not configured or not reachable.

    Callers treat this as non-fatal: the ledger on disk and in git is still
    the authoritative copy, so a Supabase outage must never stop settlements
    or block a publish.
    """


def _headers() -> dict:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SupabaseUnavailable(
            "SUPABASE_URL / SUPABASE_ANON_KEY not set in the environment")
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def _request(method: str, url: str, **kw) -> requests.Response:
    """One retry pass with backoff. A transient 5xx should not lose a sync."""
    last = None
    for attempt in range(3):
        try:
            r = requests.request(method, url, timeout=30, **kw)
            if r.status_code < 500:
                return r
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(1.5 * (attempt + 1))
    raise SupabaseUnavailable(f"{method} {url} failed after 3 tries — {last}")


def load_local() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in
            LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


def save_local(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r.get("id", 0))
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                      encoding="utf-8")


def fetch_remote() -> list[dict]:
    r = _request("GET", f"{SUPABASE_URL}/rest/v1/tara_state",
                 headers=_headers(),
                 params={"doc_path": f"eq.{DOC_PATH}", "select": "data"})
    if r.status_code == 404 or not r.ok:
        raise SupabaseUnavailable(f"read failed — HTTP {r.status_code}: {r.text[:200]}")
    body = r.json()
    if not body:
        return []
    data = body[0].get("data") or {}
    entries = data.get("entries", [])
    return entries if isinstance(entries, list) else []


def merge(local: list[dict], remote: list[dict]) -> tuple[list[dict], int, int]:
    """Union on id. A settled row beats an unsettled one for the same id.

    Returns (merged, n_gained_from_remote, n_updated_by_remote).

    The settled-wins rule is not arbitrary: settling is one-way in this system
    (see ledger.cmd_unsettle, which exists only to repair corrupt data and
    records every reversal). So between two versions of one prediction, the
    settled one is strictly newer information.
    """
    by_id: dict = {}
    for r in local:
        rid = r.get("id")
        if rid is not None:
            by_id[rid] = r

    gained = updated = 0
    for r in remote:
        rid = r.get("id")
        if rid is None:
            continue
        cur = by_id.get(rid)
        if cur is None:
            by_id[rid] = r
            gained += 1
            continue
        # Same prediction, two views — prefer whichever has an outcome.
        cur_settled = cur.get("outcome") is not None
        rem_settled = r.get("outcome") is not None
        if rem_settled and not cur_settled:
            by_id[rid] = r
            updated += 1
    return sorted(by_id.values(), key=lambda r: r.get("id", 0)), gained, updated


def push(rows: list[dict]) -> None:
    payload = {
        "doc_path": DOC_PATH,
        "data": {"entries": rows, "count": len(rows)},
        "updated_by": os.environ.get("SYNC_SOURCE", "local"),
    }
    r = _request(
        "POST", f"{SUPABASE_URL}/rest/v1/tara_state",
        headers={**_headers(), "Prefer": "resolution=merge-duplicates"},
        params={"on_conflict": "doc_path"},
        data=json.dumps(payload),
    )
    if not r.ok:
        raise SupabaseUnavailable(
            f"write failed — HTTP {r.status_code}: {r.text[:200]}")


def sync() -> str:
    """Pull, merge, save, push. Returns a one-line human summary."""
    local = load_local()
    remote = fetch_remote()
    merged, gained, updated = merge(local, remote)
    if gained or updated:
        save_local(merged)
    if len(merged) != len(remote) or gained or updated:
        push(merged)
        return (f"synced {len(merged)} rows "
                f"(+{gained} new from cloud, {updated} settled by cloud, "
                f"pushed {len(merged) - len(remote):+d} vs cloud)")
    return f"already in sync ({len(merged)} rows)"


def main() -> int:
    try:
        print(sync())
        return 0
    except SupabaseUnavailable as e:
        # Non-fatal by design — the file and git copy remain authoritative.
        print(f"supabase sync skipped: {e}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
