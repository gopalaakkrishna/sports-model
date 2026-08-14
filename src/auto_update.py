"""One command that keeps the Sports board current, meant to run on a timer.

    python src/auto_update.py --mode light    # settle, re-price, publish
    python src/auto_update.py --mode full     # + refresh data and re-predict

WHY TWO MODES

The chain has two halves with very different costs. Settling a finished game and
re-pricing against Kalshi takes under a minute; refetching every historical
source and refitting the models takes many minutes and hammers other people's
APIs. Running the heavy half every 30 minutes would be rude and pointless —
match archives do not change intraday.

    light  settle -> export (auto-locks TAKE calls) -> publish
    full   refresh sources -> re-predict every sport -> then the light chain

PUBLISHING ONLY WHEN SOMETHING CHANGED

Every run rewrites sports.json, and every run stamps a fresh `generated` time.
Committing that unconditionally would mean a commit and a Vercel rebuild every
30 minutes forever, burning build minutes to publish an identical board. So the
payload is compared with `generated` and the diagnostics excluded: if nothing
that a reader would notice has changed, the run publishes nothing and says so.

WHAT THIS CANNOT DO

It runs on this machine, so it only runs while this machine is awake. That is
the honest limit of a local scheduler; moving it to CI would need the model code
and data in a repo with credentials, which is a bigger change than a timer.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).parent
PY = sys.executable
# Locally, tara-app is a sibling folder on disk. In GitHub Actions there is no
# such sibling — the workflow checks out a second copy of tara-app into a
# path inside this repo's workspace and points here via TARA_APP_DIR, so the
# same script runs unmodified in both places.
APP = Path(os.environ.get("TARA_APP_DIR", str(ROOT.parent / "tara-app")))
PAYLOAD = APP / "public" / "sports.json"
LOG = ROOT / "reports" / "auto_update.log"
LOCK = ROOT / "reports" / "auto_update.lock"
# Files that must survive between runs but are too small/important for cache
# eviction risk: the ledger is the entire prediction record, and the
# calibration jsons are cheap config the predict scripts read every run.
# Everything else that changes per run (data/raw/*.parquet) is either
# regenerable from a public API (GitHub Actions cache, see full.yml) or
# static with no fetcher (committed once, see .gitignore).
SELF_STATE_PATHS = [
    "data/processed/ledger.jsonl",
    "data/processed/calibration.json",
    "data/processed/mlb_calibration.json",
    "data/processed/inplay_calibration.json",
    "data/processed/best_params.json",
    # Simultaneous Kalshi/bookmaker price snapshots and their settlement
    # cache. Append-only measurement data — the whole point is that it
    # accumulates across CI runs, so it must ride git like the ledger does.
    "data/live_pairs.csv",
    "data/live_pairs_results.csv",
    # The published board itself. This repo is public, so the app fetches it
    # straight from raw.githubusercontent.com at runtime rather than having it
    # baked into a Vercel build — which means an update is visible to every
    # viewer within one CDN cache window (~5 min) instead of requiring a
    # redeploy, and no cross-repo token is involved.
    "public/sports.json",
]
# Canonical published location inside THIS repo, served publicly at
# https://raw.githubusercontent.com/gopalaakkrishna/sports-model/main/public/sports.json
SELF_BOARD = ROOT / "public" / "sports.json"

# Windows Task Scheduler's MultipleInstances=IgnoreNew only stops the SAME
# scheduled task overlapping itself. It does nothing when a manual invocation
# and a scheduled run happen at once, or when light and full overlap. That gap
# is exactly what corrupted today's board: a manual run and the scheduler's
# catch-up run both had mlb_predict.py and wnba_kalshi.py writing report CSVs
# at the same moment, the two writes interleaved on disk, and every MLB/WNBA
# match ended up with 4 rows instead of 2 — which silently HALVED every
# normalized market price (a real 0.59 landed on the board as 0.29). Nothing
# in the chain raised an error; the corruption was only visible by eye.
#
# A simple age-based lock file closes the gap without adding a dependency for
# PID liveness checks. A lock older than the longest plausible run means its
# owner died without cleaning up, so it is taken over rather than left stuck
# forever requiring a manual delete.
_LOCK_MAX_AGE_S = 45 * 60


class _AlreadyRunning(RuntimeError):
    pass


@contextlib.contextmanager
def _locked():
    if LOCK.exists():
        age = time.time() - LOCK.stat().st_mtime
        if age < _LOCK_MAX_AGE_S:
            raise _AlreadyRunning(
                f"another auto_update started {age / 60:.0f} min ago "
                f"is still marked running ({LOCK})")
        log(f"  stale lock ({age / 60:.0f} min old) — previous run did not "
            f"clean up; taking over")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()}",
                    encoding="utf-8")
    try:
        yield
    finally:
        try:
            LOCK.unlink()
        except OSError:
            pass

# Scripts that regenerate today's predictions. Each writes its own dated CSV
# into reports/; export_tara reads whatever is there.
PREDICT = [
    ("soccer", "kalshi_edge.py"),
    ("mlb", "mlb_predict.py"),
    ("wnba", "wnba_kalshi.py"),
    ("cricket", "hundred_predict.py"),
]


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z  {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def run(script: str, args: list[str] | None = None, timeout: int = 1800) -> bool:
    """Run a step. A failure is logged and reported, never fatal.

    One dead API must not stop the rest of the chain — a Kalshi outage should
    still let finished games settle and the board republish.
    """
    t0 = time.time()
    try:
        r = subprocess.run([PY, "-u", str(SRC / script)] + (args or []),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"  {script:<22} TIMEOUT after {timeout}s")
        return False
    except OSError as e:
        log(f"  {script:<22} FAILED to launch: {e}")
        return False
    ok = r.returncode == 0
    log(f"  {script:<22} {'ok ' if ok else 'FAILED'} {time.time() - t0:5.1f}s")
    if not ok:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        for t in tail:
            log(f"      {t}")
    return ok


def signature(path: Path) -> str | None:
    """Hash of the payload with volatile fields removed.

    `generated` changes every run by definition, and the diagnostics block
    counts API calls. Neither is something a reader of the board can see, so
    neither should trigger a publish.
    """
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    d.pop("generated", None)
    d.pop("_start_times", None)
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git(cwd: Path, *args: str) -> tuple[int, str]:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def _commit_and_push(cwd: Path, paths: list[str], message: str, dry: bool,
                     label: str) -> bool:
    """Stage exactly `paths`, commit only if something changed, push.

    Shared by publish() (tara-app's sports.json) and publish_self_state()
    (this repo's ledger + calibration). Same shape, same care: never commits
    an empty diff, never pushes on --dry-run.
    """
    existing = [p for p in paths if (cwd / p).exists()]
    if not existing:
        return False
    code, out = git(cwd, "status", "--porcelain", "--", *existing)
    if code != 0:
        log(f"  [{label}] git status failed: {out}")
        return False
    if not out.strip():
        log(f"  [{label}] nothing changed")
        return False
    if dry:
        log(f"  [{label}] --dry-run: not committing")
        return False
    code, out = git(cwd, "add", "--", *existing)
    if code != 0:
        log(f"  [{label}] git add failed: {out}")
        return False
    code, out = git(cwd, "commit", "-q", "-m", message)
    if code != 0:
        log(f"  [{label}] git commit failed: {out}")
        return False
    code, out = git(cwd, "push", "-q", "origin", "main")
    if code != 0:
        log(f"  [{label}] git push failed: {out}")
        return False
    log(f"  [{label}] published")
    return True


def publish(dry: bool) -> bool:
    stamp = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%MZ}"
    ok = _commit_and_push(APP, ["public/sports.json"],
                          f"Sports data refresh ({stamp})", dry, "tara-app")
    if ok:
        log("  tara-app: published — Vercel will redeploy")
    return ok


def publish_self_state(dry: bool) -> bool:
    """Commit this repo's own durable state — chiefly the ledger — back to
    itself. Required in CI: every scheduled run starts from a fresh checkout
    of whatever was last committed, so a lock that isn't pushed back here is a
    lock that vanishes the moment the job ends. Locally this is a harmless
    no-op most runs (nothing to commit) since disk state already persists."""
    stamp = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%MZ}"
    return _commit_and_push(ROOT, SELF_STATE_PATHS,
                            f"Ledger/state update ({stamp})", dry, "sports-model")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("light", "full"), default="light")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the chain but never commit or push")
    ap.add_argument("--no-publish", action="store_true")
    args = ap.parse_args()

    try:
        with _locked():
            return _run(args)
    except _AlreadyRunning as e:
        log(f"skipping — {e}")
        return 0


def _run(args) -> int:
    t0 = time.time()
    log(f"=== auto_update --mode {args.mode}")
    before = signature(PAYLOAD)

    if args.mode == "full":
        log("refreshing sources")
        run("refresh_all.py", timeout=3600)
        # Daily scoring of the simultaneous price pairs: pulls settlement
        # results for finished games into the cache and logs the current
        # verdict. Pure measurement, non-fatal.
        log("scoring kalshi/book price pairs")
        run("pair_analysis.py", timeout=600)

    # Re-price on EVERY run, not just the full one. The first version only
    # re-predicted in full mode, so the board carried prices up to 24h old while
    # claiming a 30-minute cadence — the exact staleness this whole schedule was
    # built to remove. Measured cost of the four scripts is ~2 minutes, which is
    # nothing against a 30-minute interval.
    log("re-pricing against the live market")
    for name, script in PREDICT:
        run(script, timeout=1800)

    # Snapshot Kalshi and DraftKings prices for the same games at the same
    # moment. Pure measurement — feeds pair_analysis.py, touches nothing
    # else — so a failure here must never block settling or the board.
    log("snapshotting kalshi/book price pairs")
    run("fetch_live_pairs.py", timeout=600)

    # Pull the shared ledger BEFORE settling, so this run sees predictions
    # another runner locked that this checkout does not have yet — otherwise
    # settle.py cannot resolve them and they sit open forever.
    log("syncing ledger with Supabase (pre-settle)")
    run("supabase_sync.py", timeout=180)

    log("settling finished games")
    run("settle.py", timeout=900)

    # Push again after settling so the outcomes this run resolved are shared
    # immediately, rather than waiting for the next run's pre-settle pull.
    log("syncing ledger with Supabase (post-settle)")
    run("supabase_sync.py", timeout=180)

    log("exporting board")
    if not run("export_tara.py", timeout=900):
        log("export failed — nothing published")
        return 1

    # Mirror the freshly exported board into this repo's public/ so it is
    # served from the canonical raw.githubusercontent.com URL the app reads.
    # When running in CI, TARA_APP_DIR already points here and this is a
    # no-op self-copy; locally it keeps the public copy in step with the one
    # written into tara-app.
    try:
        if PAYLOAD.exists() and PAYLOAD.resolve() != SELF_BOARD.resolve():
            SELF_BOARD.parent.mkdir(parents=True, exist_ok=True)
            SELF_BOARD.write_bytes(PAYLOAD.read_bytes())
    except OSError as e:
        log(f"  could not mirror board into public/: {e}")

    # Publish the ledger back to THIS repo unconditionally (cheap no-op via
    # git status if nothing changed). This is decoupled from the board-changed
    # check below on purpose: a settlement can update the ledger without the
    # visible board necessarily changing shape, and in CI a lock that is not
    # pushed back here vanishes the moment the job's checkout is discarded —
    # the next scheduled run starts fresh and would relock or lose it.
    if not args.no_publish:
        publish_self_state(args.dry_run)

    after = signature(PAYLOAD)
    if before is not None and before == after:
        log(f"no material board change — not publishing  ({time.time() - t0:.0f}s)")
        return 0

    if args.no_publish:
        log("board changed; --no-publish set")
        return 0
    log("board changed — publishing")
    publish(args.dry_run)
    log(f"=== done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
