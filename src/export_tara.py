"""Export the prediction record as JSON for the Tara app to read.

Tara is deployed on Vercel and cannot reach this machine, so the data has to
travel as a committed artefact rather than a live query. This writes a single
static file into the app's public directory:

    tara-app/public/sports.json

Static on purpose. It carries no credentials, needs no serverless function, and
is cacheable — the app just fetches it. The cost is that it is only as fresh as
the last run, which is why `generated` is in the payload and the view shows it.

Everything here reuses dashboard.py rather than recomputing, so the app and the
HTML dashboard cannot drift into disagreeing about the same record.

    python export_tara.py
    python export_tara.py --out ../../tara-app/public/sports.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import dashboard as D
import leagues as LG
from start_times import StartTimes

# Series whose fixtures reach the app without a clock in their source file.
_CLOCK_SERIES = ["KXWNBAGAME", "KXNFLGAME", "KXMLSGAME", "KXLIGAMXGAME"]


def split_fixture(text: str) -> tuple[str, str]:
    """'Away @ Home' or 'Home v Away' -> the two names, order-insensitive."""
    t = re.sub(r"\s*\(\d{4}-\d{2}-\d{2}\)\s*$", "", str(text)).strip()
    for sep in (" @ ", " vs ", " v "):
        if sep in t:
            a, b = t.split(sep, 1)
            return a.strip(), b.strip()
    return t, ""


def fill_start(rows: list[dict], st: StartTimes, key: str) -> int:
    """Attach a real kick-off time to any row that only has a date.

    A row keeps its date-only value when nothing matches. A missing clock shows
    as 'all day', which is honest; a WRONG clock would bucket the fixture under
    the wrong day and could mark a live game as still upcoming.
    """
    filled = 0
    for r in rows:
        s = r.get("start")
        if s and ("+" in str(s)[10:] or "Z" in str(s)[10:]):
            continue                      # already carries a timezone => timed
        date = str(s)[:10] if s else None
        a, b = split_fixture(r.get(key) or "")
        if not b:
            continue
        when = st.lookup(a, b, date)
        if when is not None:
            r["start"] = when.astimezone(timezone.utc).isoformat()
            filled += 1
    return filled

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT.parent / "tara-app" / "public" / "sports.json"


def clean(x):
    """JSON cannot hold NaN/NaT. Absent means absent, not 'NaN'."""
    # NaT is its own type, not a Timestamp, so it slips past the Timestamp
    # branch below and reaches the sort as a NaTType that will not compare
    # against a string.
    if x is None or x is pd.NaT:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if x != x else round(float(x), 4)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        if x is pd.NaT:
            return None
        # Naive means date-only. Keep it that way; the app must not shift it
        # into a timezone and land a day early, which is exactly what happened
        # when bare dates were stamped as UTC midnight.
        return x.isoformat() if x.tzinfo is None else x.tz_convert("UTC").isoformat()
    return x


def clean_dict(d: dict) -> dict:
    """Run every value through clean(). Selectively cleaning a few fields is
    how a NaN reaches json.dumps — a missing `outcome` on an open row is NaN,
    not None, and looks nothing like a number until it is serialised."""
    return {k: clean(v) for k, v in d.items()}


def score_row(label: str, sport: str, g: pd.DataFrame) -> dict:
    s = D.score(g)
    return {"label": label, "sport": sport, "n": s["n"], "wins": s["wins"],
            "losses": s["n"] - s["wins"], "win_rate": clean(s["win_rate"]),
            "ll": clean(s["ll"]), "vs_market": clean(s.get("vs_market")),
            "roi": clean(s.get("roi"))}


def autolock(data: list[dict], hours: float) -> list[str]:
    """Lock ALIGNED-grade calls that start inside the actionable window.

    This is the fix for the failure that keeps recurring: a call gets surfaced
    to the user, never written to the ledger, and then quietly does not exist
    when it loses. Toronto and Seattle were both given as HIGH on 2026-08-06,
    both lost, and neither was in the record until they were backfilled by hand.

    So surfacing and recording are now one action. Exporting the board IS the
    act of showing the calls, therefore exporting also commits them.

    Only TAKE, and only inside the window — locking a soccer fixture three weeks
    out would bloat the ledger with rows whose prices will have moved long
    before kick-off, and the price recorded at lock time is the whole point of
    the row.
    """
    import ledger as L
    if not data:
        return []
    now = pd.Timestamp.now(tz="UTC")
    locked = []
    for r in data:
        if r.get("advice") != "ALIGNED":
            continue
        start = D.to_utc(r.get("start"))
        if start is pd.NaT:
            continue
        if start.tzinfo is None:
            # Date-only. Skipping these is how INDIANA FEVER — a TAKE call for
            # TODAY — sat untracked while every other one of today's calls was
            # locked: the window test needed a clock the row never had. A date
            # of today or tomorrow is inside a 36h window whatever the clock
            # turns out to be, so judge on the date instead.
            day = start.date()
            today = now.tz_convert(D.ET).date()
            if not (today <= day <= today + timedelta(days=int(hours // 24))):
                continue
            ev_date = day
        else:
            age = (start - now).total_seconds() / 3600.0
            if age < 0 or age > hours:
                continue
            ev_date = start.tz_convert(D.ET).date()
        # The LEDGER stores the side, not the team name. settle.py resolves a
        # fixture to HOME/AWAY/DRAW and compares it to the stored pick, so a row
        # locked as "BOCHUM" could never match its own result and would settle
        # as a loss whatever happened. Refuse rather than guess.
        side = str(r.get("side") or "").upper()
        if side not in ("HOME", "AWAY", "DRAW"):
            continue
        ev = f"{r['match']} ({ev_date:%Y-%m-%d})"
        a = SimpleNamespace(
            sport=D.canon(r["sport"]), league=r["league"], event=ev,
            market="1X2" if D.canon(r["sport"]) == "soccer" else "ML",
            pick=side, model_prob=float(r["model"]),
            market_prob=float(r["mkt"]) if r["mkt"] == r["mkt"] else None,
            odds=None, venue="kalshi", stake_units=1.0,
            notes="auto-locked by export_tara (ALIGNED, within window)",
            backfill_at=None, backfill_reason=None)
        before = len(L._load())
        L.cmd_lock(a)
        if len(L._load()) > before:
            locked.append(f"{D.name_pick(side, r['match'])} — {r['match']}")
    return locked


def resolve_clocks(now: datetime) -> tuple[list[dict], StartTimes]:
    """Today's model output with real kick-off times attached.

    Runs BEFORE anything else needs them, because auto-locking has to know when
    a fixture starts to decide whether it is inside the actionable window. The
    first version locked after this step and silently skipped every WNBA call —
    those arrive date-only, so the window test could never pass.
    """
    st = StartTimes()
    st.load_kalshi(_CLOCK_SERIES)
    st.load_mlb((now - timedelta(days=10)).strftime("%Y-%m-%d"),
                (now + timedelta(days=10)).strftime("%Y-%m-%d"))
    rows = D.upcoming_from_reports()
    st.stats["filled_upcoming"] = fill_start(rows, st, "match")
    return rows, st


def build(upcoming: list[dict], st: StartTimes) -> dict:
    df = D.load()
    now = datetime.now(timezone.utc)
    out = {"generated": now.isoformat(timespec="seconds"),
           "record": None, "by_sport": [], "by_league": [], "open": [],
           "settled": [], "upcoming": [], "calibration": [],
           "inplay_grid": D.inplay_grid(), "disclosures": []}
    if df.empty:
        return out

    voided = df[df.get("voided", pd.Series(False, index=df.index)).fillna(False)]
    live = df[~df.index.isin(voided.index)].copy()
    live["sport"] = live["sport"].map(D.canon)
    live["league_name"] = live["league"].map(LG.pretty)
    settled = live[live["outcome"].notna()].copy()
    open_ = live[live["outcome"].isna()].copy()

    if len(settled):
        s = D.score(settled)
        out["record"] = {"n": s["n"], "wins": s["wins"],
                         "losses": s["n"] - s["wins"],
                         "win_rate": clean(s["win_rate"]), "ll": clean(s["ll"]),
                         "mkt_ll": clean(s.get("mkt_ll")),
                         "n_mkt": s.get("n_mkt"),
                         "vs_market": clean(s.get("vs_market")),
                         "pnl": clean(s.get("pnl")), "roi": clean(s.get("roi"))}
        out["by_sport"] = [score_row(D.label(k), k, g)
                           for k, g in settled.groupby("sport")]
        out["by_league"] = [score_row(lg, sp, g) for (sp, lg), g
                            in settled.groupby(["sport", "league_name"])]

        band = pd.cut(settled["model_prob"], [0, .4, .55, .7, .85, 1.0],
                      labels=["<40%", "40-55%", "55-70%", "70-85%", ">85%"])
        for k, g in settled.groupby(band, observed=True):
            out["calibration"].append({
                "band": str(k), "n": len(g),
                "said": clean(float(g["model_prob"].mean())),
                "actual": clean(float(g["won"].astype(int).mean()))})

    def ledger_row(r) -> dict:
        mp = r.get("market_prob")
        has = mp == mp and mp is not None
        gap = abs(r["model_prob"] - mp) if has else float("nan")
        tier = D._tier(gap)
        start = D.date_in_event(r["event"])
        return clean_dict({"id": int(r["id"]), "sport": r["sport"],
                "sport_label": D.label(r["sport"]), "league": r["league_name"],
                "event": r["event"], "pick": D.name_pick(r["pick"], r["event"]),
                "market": r.get("market"), "model": clean(r["model_prob"]),
                "mkt": clean(mp) if has else None, "tier": tier,
                "advice": D.advice(tier), "label": D.human_label(gap),
                "start": clean(start),
                # A row whose fixture is long past but still unsettled is
                # waiting on a data source, not on the game. Boca v Estudiantes
                # sat open for two days looking like an upcoming pick because
                # football-data.co.uk had not published 2026-08-05 yet. Say so
                # rather than let it look like a live call.
                # pd.isna, not `is None`: an unsettled outcome in a DataFrame
                # row is NaN, and `NaN is None` is False — the same trap that
                # silently disabled this check the first time.
                "stale_days": (
                    int((pd.Timestamp.now().normalize() - start).days)
                    if pd.isna(r.get("outcome")) and start is not pd.NaT
                    and getattr(start, "tzinfo", None) is None
                    and (pd.Timestamp.now().normalize() - start).days >= 1
                    else None),
                "outcome": r.get("outcome"),
                "won": clean(r.get("won")),
                "backfilled": bool(r.get("backfilled", False) is True)})

    out["open"] = sorted((ledger_row(r) for _, r in open_.iterrows()),
                         key=lambda x: (x["start"] is None, x["start"] or ""))
    out["settled"] = sorted((ledger_row(r) for _, r in settled.iterrows()),
                            key=lambda x: (x["start"] or ""), reverse=True)

    up = pd.DataFrame(upcoming)
    if len(up):
        up["_k"] = up["start"].map(D.sort_key)
        up = up.sort_values("_k")
        for _, r in up.iterrows():
            out["upcoming"].append(clean_dict({
                "sport": D.canon(r["sport"]),
                "sport_label": D.label(r["sport"]), "league": r["league"],
                "match": r["match"],
                "pick": D.name_pick(r["pick"], r["match"]),
                "model": clean(r["model"]), "mkt": clean(r["mkt"]),
                "tier": r["tier"], "advice": r["advice"],
                "label": r.get("label"), "side": r.get("side"),
                "start": clean(r["start"])}))

    # ── ONE board, not two overlapping lists ────────────────────────────
    # "Upcoming" and "Open" were 94% the same rows: 17 of 18 open predictions
    # also appeared in upcoming, the same pick in two tabs with two framings.
    # Tracked-or-not is a PROPERTY of a row, not a category of row, so it is now
    # a badge. The only thing the Open tab held uniquely was a fixture whose
    # result feed has stalled, which must survive the merge — it is in the past,
    # so no model run will re-emit it.
    def _teams(text: str) -> frozenset:
        t = re.sub(r"\s*\(\d{4}-\d{2}-\d{2}\)\s*$", "", str(text)).strip()
        for sep in (" @ ", " vs ", " v "):
            if sep in t:
                a, b = t.split(sep, 1)
                return frozenset({a.strip().lower(), b.strip().lower()})
        return frozenset({t.lower()})

    def _et_date(start) -> str | None:
        if not start:
            return None
        try:
            t = datetime.fromisoformat(str(start))
        except ValueError:
            return None
        if t.tzinfo is not None:
            t = t.astimezone(D.ET)
        return t.date().isoformat()

    def fkey(teams: frozenset, date: str | None) -> tuple:
        # The date MUST be part of this key. Teams play multi-game series —
        # the same "Away @ Home" pairing legitimately recurs on consecutive
        # days (Mets @ Braves, back-to-back). Matching on teams alone once
        # attached a "tracked" badge and a locked price from one day's game
        # onto a DIFFERENT day's game with the same two teams — the ledger's
        # own _ticker_date docstring warns about exactly this failure for
        # Kalshi matching, and it had not been applied here.
        return (teams, date)

    def _event_date(event: str) -> str | None:
        m = re.search(r"\((\d{4}-\d{2}-\d{2})\)", str(event))
        return m.group(1) if m else None

    by_fixture = {fkey(_teams(o["event"]), _event_date(o["event"])): o
                 for o in out["open"]}
    board = []
    for r in out["upcoming"]:
        o = by_fixture.pop(fkey(_teams(r["match"]), _et_date(r.get("start"))), None)
        if o is None:
            board.append({**r, "tracked": False, "ledger_id": None,
                          "stale_days": None, "mkt_now": r.get("mkt")})
            continue
        # A tracked row shows what was LOCKED, not today's view of the fixture.
        # Merging on teams alone put the badge on whichever side the model
        # currently favours: #5 displayed CHICAGO FIRE carrying "tracked", while
        # the ledger had PORTLAND TIMBERS. Three rows were backing the opposite
        # team to the one being scored — the same wrong-side error as the
        # Detroit/Seattle call. The record is the truth here; the live price is
        # context, carried separately as mkt_now.
        lock_gap = (abs(o["model"] - o["mkt"])
                    if o.get("mkt") is not None and o.get("model") is not None
                    else float("nan"))
        board.append({**r,
                      "pick": o["pick"], "model": o["model"], "mkt": o["mkt"],
                      "tier": D._tier(lock_gap),
                      "advice": D.advice(D._tier(lock_gap)),
                      "label": D.human_label(lock_gap),
                      "mkt_now": r.get("mkt"),
                      "tracked": True, "ledger_id": o["id"],
                      "stale_days": o.get("stale_days")})
    # Whatever is left is tracked but no longer in any model run.
    for o in by_fixture.values():
        board.append({
            "sport": o["sport"], "sport_label": o["sport_label"],
            "league": o["league"], "match": o["event"], "pick": o["pick"],
            "model": o["model"], "mkt": o["mkt"], "tier": o["tier"],
            "advice": o["advice"], "label": o.get("label"),
            "start": o["start"], "side": o.get("market"),
            "tracked": True, "ledger_id": o["id"],
            "stale_days": o.get("stale_days")})
    board.sort(key=lambda x: (x["start"] is None, str(x["start"] or "")))
    out["board"] = board
    out["board_counts"] = {"total": len(board),
                           "tracked": sum(1 for b in board if b["tracked"])}

    # ── the verdict, computed once and shipped ──────────────────────────
    # The board's most important number is not the record, it is whether the
    # gap to the market has cleared the noise floor. Computing it here means
    # the app cannot present a stale or flattering version of it.
    have = settled[settled["market_prob"].notna()] if len(settled) else settled
    if len(have) >= 3:
        ys = have["won"].astype(int).to_numpy()
        pm = have["model_prob"].to_numpy(float)
        qm = have["market_prob"].to_numpy(float)
        cl = lambda v: np.clip(v, 1e-15, 1 - 1e-15)
        diff = (-(ys * np.log(cl(pm)) + (1 - ys) * np.log(cl(1 - pm)))
                + (ys * np.log(cl(qm)) + (1 - ys) * np.log(cl(1 - qm))))
        rng = np.random.default_rng(0)
        boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                         for _ in range(4000)])
        lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        gap = float(diff.mean())
        worse, better = lo > 0, hi < 0
        out["verdict"] = clean_dict({
            "n": len(have), "gap": gap, "lo": lo, "hi": hi,
            "significant_worse": worse,
            "headline": ("Measurably worse than the market." if worse else
                         "Measurably better than the market." if better else
                         "Indistinguishable from the market so far."),
            "detail": (f"Model log loss is {gap:+.3f} against the market's on the "
                       f"{len(have)} predictions carrying a logged price "
                       f"(95% CI [{lo:+.3f}, {hi:+.3f}]). "
                       + ("The interval sits entirely above zero, so this is a "
                          "real deficit rather than a small sample."
                          if worse else
                          "The interval spans zero, so nothing is proven either way."
                          if not better else
                          "The interval sits entirely below zero.")),
        })

    # Upcoming clocks were resolved before locking; the ledger rows still need
    # theirs, and they only exist as "Home v Away (date)" strings.
    n_open = fill_start(out["open"], st, "event")
    n_set = fill_start(out["settled"], st, "event")
    out["_start_times"] = {"filled_open": n_open, "filled_settled": n_set,
                           **st.stats}
    # Re-sort: a row that just gained a clock may now belong elsewhere.
    out["upcoming"].sort(key=lambda x: (x["start"] is None, x["start"] or ""))
    out["open"].sort(key=lambda x: (x["start"] is None, x["start"] or ""))
    out["settled"].sort(key=lambda x: (x["start"] or ""), reverse=True)

    for _, r in voided.iterrows():
        out["disclosures"].append({"id": int(r["id"]), "kind": "voided",
                                   "event": r["event"],
                                   "reason": r.get("void_reason")})
    if "backfilled" in df:
        for _, r in df[df["backfilled"].fillna(False)].iterrows():
            out["disclosures"].append({"id": int(r["id"]), "kind": "backfilled",
                                       "event": r["event"],
                                       "reason": r.get("backfill_reason")})
    for _, r in df.iterrows():
        # Absent in a DataFrame column is NaN, not None — `or []` does not
        # catch it because NaN is truthy.
        hist = r.get("correction_history")
        for h in (hist if isinstance(hist, list) else []):
            out["disclosures"].append({
                "id": int(r["id"]), "kind": "corrected", "event": r["event"],
                "reason": f"was '{h.get('reverted_outcome')}' — {h.get('reason')}"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--lock-hours", type=float, default=36.0,
                    help="lock ALIGNED calls starting within this many hours")
    ap.add_argument("--no-lock", action="store_true",
                    help="export without committing anything to the ledger")
    args = ap.parse_args()

    upcoming, st = resolve_clocks(datetime.now(timezone.utc))
    if not args.no_lock:
        newly = autolock(upcoming, args.lock_hours)
        print(f"auto-lock: {len(newly)} new ALIGNED call(s) committed to the ledger"
              f" (window {args.lock_hours:.0f}h)")
        for s in newly:
            print(f"    + {s}")
        if newly:
            print()

    data = build(upcoming, st)
    out = Path(args.out)
    if not out.parent.exists():
        print(f"target directory does not exist: {out.parent}")
        raise SystemExit(1)
    out.write_text(json.dumps(data, allow_nan=False, separators=(",", ":")),
                   encoding="utf-8")
    rec = data["record"] or {}
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
    print(f"  record   {rec.get('wins', 0)}-{rec.get('losses', 0)}")
    print(f"  open     {len(data['open'])}")
    print(f"  upcoming {len(data['upcoming'])}")
    print(f"  grid     {len(data['inplay_grid'])} states")
    s = data.get("_start_times") or {}
    print(f"  clocks   filled {s.get('filled_upcoming',0)} upcoming, "
          f"{s.get('filled_open',0)} open, {s.get('filled_settled',0)} settled"
          f"  (from {s.get('kalshi_markets',0)} Kalshi markets, "
          f"{s.get('mlb_games',0)} MLB games, {s.get('errors',0)} errors)")
    timed = sum(1 for r in data["upcoming"]
                if r.get("start") and ("+" in str(r["start"])[10:]
                                       or "Z" in str(r["start"])[10:]))
    print(f"           {timed}/{len(data['upcoming'])} upcoming now have a "
          f"kick-off time")


if __name__ == "__main__":
    main()
