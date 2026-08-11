"""One command to refresh every data source the project uses.

Replaces the piecemeal fetching that accumulated during development. Sources are
independent, so they run in parallel; each is cached, so a re-run only pulls what
has changed.

    python src/refresh_all.py              # everything
    python src/refresh_all.py --only mlb,kalshi
    python src/refresh_all.py --inventory  # report state, fetch nothing

An honest limit: this covers what the project KNOWS it needs. Data requirements
discovered mid-analysis — the Leagues Cup bridge existed only because the
MLS/Liga MX rating graphs turned out to be disconnected — cannot be pre-fetched.
New sources get added here as they are found, so the list grows but the number
of commands stays at one.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).parent
PY = sys.executable

# name -> (script, output file, max age in days before considered stale)
SOURCES = {
    "euro_soccer": ("fetch_data.py", "data/raw/football_data_raw.parquet", 1),
    "year_round": ("fetch_new_leagues.py", "data/raw/new_leagues_raw.parquet", 1),
    "mlb": ("fetch_mlb.py", "data/raw/mlb_games.parquet", 1),
    "mlb_innings": ("fetch_mlb_innings.py", "data/raw/mlb_inplay_states.parquet", 2),
    "nfl_wnba": ("fetch_nfl_wnba.py", "data/raw/nfl_games.parquet", 1),
    "kalshi_closes": ("fetch_kalshi_closes.py", "data/raw/kalshi_closes.parquet", 1),
    "kalshi_mlb_closes": ("fetch_kalshi_mlb.py", "data/raw/kalshi_mlb_closes.parquet", 1),
}


def age_days(p: Path) -> float | None:
    if not p.exists():
        return None
    return (time.time() - p.stat().st_mtime) / 86400.0


def inventory() -> None:
    print(f"{'source':<22}{'file':<44}{'rows':>10}{'age':>10}  state")
    for name, (_, rel, max_age) in SOURCES.items():
        p = ROOT / rel
        a = age_days(p)
        if a is None:
            print(f"{name:<22}{rel:<44}{'-':>10}{'-':>10}  MISSING")
            continue
        try:
            n = len(pd.read_parquet(p))
        except Exception:
            n = -1
        state = "stale" if a > max_age else "ok"
        print(f"{name:<22}{rel:<44}{n:>10,}{a:>9.1f}d  {state}")

    # Sources held in code rather than downloaded.
    print(f"\n{'in-repo data':<22}")
    try:
        sys.path.insert(0, str(SRC))
        import leagues_cup_data as LC
        d = LC.load()
        print(f"  leagues_cup_data.py  {len(d)} matches "
              f"({d['Date'].min().date()} .. {d['Date'].max().date()})")
    except Exception as e:
        print(f"  leagues_cup_data.py  unavailable: {e}")

    print(f"\n{'known gaps':<22}")
    for gap in [
        "2024 Leagues Cup scores (extraction unreliable; 2023 was garbage)",
        "CONCACAF Champions Cup results (more MLS/LigaMX bridge matches)",
        "Kalshi closing lines for SOCCER (only MLB is collected)",
        "Understat xG — accessible ONLY via browser navigation, not requests;"
        " see fetch_understat.py. FBref and FootyStats return 403.",
        "MLB play-by-play — needed for baserunners/bullpen in the in-play model,"
        " which is its single biggest blind spot",
        "WNBA closing lines (Basketball Reference carries no odds)",
        "Lineups / team news — no automated source identified",
    ]:
        print(f"  - {gap}")


def run_one(name: str) -> tuple[str, bool, str]:
    script, rel, _ = SOURCES[name]
    t0 = time.time()
    try:
        r = subprocess.run([PY, "-u", str(SRC / script)],
                           capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return name, False, "timed out"
    dt = time.time() - t0
    ok = r.returncode == 0 and (ROOT / rel).exists()
    tail = (r.stdout or "").strip().splitlines()
    msg = tail[-1] if tail else (r.stderr or "")[-160:]
    # Surface integrity warnings rather than burying them in the log.
    warn = [l for l in tail if "!" in l or "WARNING" in l]
    if warn:
        msg += "  || " + " | ".join(w.strip() for w in warn[:3])
    return name, ok, f"{dt:.0f}s  {msg[:150]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of: " + ",".join(SOURCES))
    ap.add_argument("--inventory", action="store_true",
                    help="report current state without fetching")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.inventory:
        inventory()
        return

    names = args.only.split(",") if args.only else list(SOURCES)
    names = [n.strip() for n in names if n.strip() in SOURCES]
    if not names:
        print("nothing to do")
        return

    print(f"refreshing {len(names)} source(s) in parallel: {', '.join(names)}\n")
    t0 = time.time()
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, n): n for n in names}
        for fut in cf.as_completed(futs):
            name, ok, msg = fut.result()
            print(f"  [{'OK ' if ok else 'FAIL'}] {name:<20} {msg}")
            results.append((name, ok))

    bad = [n for n, ok in results if not ok]
    print(f"\ntotal {time.time() - t0:.0f}s   "
          f"{len(results) - len(bad)}/{len(results)} succeeded")
    if bad:
        print(f"  FAILED: {', '.join(bad)}")
        print("  Re-run just those with --only, and read the source's own log")
        print("  in reports/ — a fetch that exits cleanly can still have")
        print("  silently truncated its data.")
    print()
    inventory()


if __name__ == "__main__":
    main()
