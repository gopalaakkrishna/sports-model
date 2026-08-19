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
import os
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

# Minimum model probability for a pick to enter the record. See the long note
# in autolock() — chiefly a fee argument, since Kalshi's fee peaks at a coin
# flip and a 52% pick is one we have said we cannot call.
#
# Raised 0.55 -> 0.60 on 2026-08-11. Settled ALIGNED picks by conviction:
#
#   >= 50%   24 picks   12-12   50.0%
#   >= 55%   17 picks   11-6    64.7%
#   >= 60%   10 picks    8-2    80.0%
#
# The 80% is NOT the expectation. For a calibrated 65% model, 8-of-10 happens
# 26% of the time — unremarkable. The honest forecast for this floor is ~65%,
# and anyone quoting the 80% is quoting noise.
#
# Cost of the raise: roughly 1.7 picks/day becomes 1.0. That is the trade —
# fewer calls, each one the model actually has something to say about.
MIN_CONVICTION = 0.60


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
# Where the board gets written. Locally that is the sibling tara-app checkout;
# in CI there is no sibling, so the workflow sets TARA_APP_DIR to the repo
# itself and the board lands in this repo's public/ (which is what the app now
# fetches at runtime).
#
# auto_update.py already honoured TARA_APP_DIR but this script did not, so in
# CI it fell back to ../tara-app/public — a path that does not exist there —
# tripped the "target directory does not exist" guard, and exited 1. That is
# the single step auto_update treats as fatal, so every cloud run failed at
# "Run the light chain" with nothing else wrong.
_APP_DIR = os.environ.get("TARA_APP_DIR")
DEFAULT_OUT = ((Path(_APP_DIR) if _APP_DIR else ROOT.parent / "tara-app")
               / "public" / "sports.json")


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
        # CONVICTION FLOOR — skip the coin-flip dead zone.
        #
        # Two independent reasons, and the structural one is the stronger:
        #
        # 1. Fees. Kalshi charges 0.07*p*(1-p), which peaks at a coin flip. At
        #    52c that is 1.75c on a 52c stake — you need a 3.4% edge just to
        #    break even. At 65c the same fee is 2.5% of stake. A pick we price
        #    at 52% is one we have explicitly said we cannot separate, so
        #    paying peak fees for it is structurally losing.
        #
        # 2. The record, weakly. Settled picks in the 50-55% band went 1-8,
        #    against a stated 53%. That is a 1.2% outcome on its own, but 7.3%
        #    across the six bands examined — suggestive, NOT conclusive, and
        #    not load-bearing here. Every other band is well calibrated
        #    (57%->58%, 63%->64%), so this is not a broken model, just a zone
        #    where it has nothing to say.
        #
        # Tara's BTC side gated the identical 50-55c zone in v13.4.144 for the
        # same fee reason, arrived at separately.
        #
        # This changes what gets TRACKED, not what gets shown — the board still
        # displays these, they simply stop entering the record as if they were
        # calls worth making.
        if float(r.get("model") or 0) < MIN_CONVICTION:
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
        # Split the record by STRATEGY. One number was hiding two different
        # experiments: an early value-betting run (deliberately backing
        # underdogs where the model disagreed with the market — the exact
        # hypothesis the 68k-match backtest disproved) and the current system
        # (only picking where the model AGREES with the market). Reporting
        # them together makes both unreadable.
        #
        # Those losing experiment picks stay in the record. Deleting picks
        # because they lost is precisely the flattering scoreboard this ledger
        # exists to prevent — they are labelled, not removed.
        def _strategy(note) -> str:
            return ("aligned" if "auto-locked" in str(note or "")
                    else "value_experiment")

        strat = settled["notes"].map(_strategy)
        out["by_strategy"] = []
        for name, label in (("aligned", "Aligned (current system)"),
                            ("value_experiment", "Value-bet experiment (abandoned)")):
            g = settled[strat == name]
            if not len(g):
                continue
            sc = D.score(g)
            out["by_strategy"].append(clean_dict({
                "key": name, "label": label, "n": sc["n"], "wins": sc["wins"],
                "losses": sc["n"] - sc["wins"], "win_rate": sc["win_rate"],
                "ll": sc["ll"], "vs_market": sc.get("vs_market"),
                "current": name == "aligned",
            }))

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
    # MAJORS ONLY (user decision, 2026-08-14). Narrowing kalshi_edge stopped
    # NEW minor-league pricing, but stale prediction reports from earlier full
    # runs kept resurrecting J-League/2. Bundesliga rows onto the board, so the
    # cut is enforced here — the one place every row passes through. Tracked
    # rows are exempt: a pick already committed to the record stays visible
    # until it settles, whatever league it is in.
    MAJORS = {
        "MLB", "NFL", "WNBA", "USA MLS", "Mexico Liga MX",
        "England Premier League", "Spain La Liga", "Italy Serie A",
        "Germany Bundesliga", "France Ligue 1", "UEFA Champions League",
        "The Hundred",  # short season, ends Aug 31 — drops off naturally
    }
    board = [b for b in board if b.get("tracked") or b.get("league") in MAJORS]
    board.sort(key=lambda x: (x["start"] is None, str(x["start"] or "")))
    # A row is "high conviction" when the model both agrees with the market
    # (ALIGNED — the band where it has tracked the closing line) AND clears the
    # coin-flip floor. Those two together are what now enters the record, so
    # the app can show exactly the set being tracked rather than approximating
    # it with its own filter and drifting out of step with the pipeline.
    for b in board:
        b["high_conviction"] = bool(
            str(b.get("advice") or "") == "ALIGNED"
            and float(b.get("model") or 0) >= MIN_CONVICTION)
    # Ship the board itself. This assignment was dropped when the
    # high_conviction flag was added, so for several runs the board was built,
    # counted, and then thrown away — board_counts said 183 while the array
    # shipped empty and the app had nothing to render. The counts being right
    # is what made it survive review: the summary looked healthy.
    out["board"] = board
    out["min_conviction"] = MIN_CONVICTION
    out["board_counts"] = {
        "total": len(board),
        "tracked": sum(1 for b in board if b["tracked"]),
        "high_conviction": sum(1 for b in board if b["high_conviction"]),
    }

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
        # ── READINESS: is there enough evidence to actually bet? ──────────
        #
        # Win rate on its own is gameable and therefore misleading. Picking
        # bigger favourites raises the win rate, but it raises the PRICE you
        # pay by the same amount — a 70% win rate on 70c favourites loses
        # money after fees. So the bar that matters is win rate minus
        # breakeven, where breakeven is the average price paid plus Kalshi's
        # 0.07*p*(1-p).
        #
        # And a point estimate is not evidence. At n=11 an 82% run carries a
        # +/-23% interval — indistinguishable from a coin flip. This reports
        # the 95% lower bound against breakeven, and how many more settled
        # picks are needed before that bound could clear it.
        cur = settled[settled["notes"].astype(str).str.contains("auto-locked")]
        cur = cur[cur["model_prob"] >= MIN_CONVICTION]
        cur = cur[cur["market_prob"].notna()]
        if len(cur) >= 3:
            y = cur["won"].astype(int).to_numpy()
            px = cur["market_prob"].to_numpy(float)
            wr = float(y.mean())
            fee = float(np.mean(0.07 * px * (1 - px)))
            be = float(px.mean()) + fee
            n = len(cur)
            se = float(np.sqrt(max(wr * (1 - wr), 1e-9) / n))
            # NOT named `lo`. The verdict block below reads the bootstrap
            # `lo`/`hi` computed further up, and naming this one `lo` silently
            # overwrote it — the board shipped "95% CI [+0.603, +0.034]", an
            # interval whose lower bound sits above its upper because that
            # lower bound was actually this win rate. The headline stayed
            # correct (computed before the clobber), so only the printed
            # interval was wrong — exactly the kind of error that survives
            # review because everything around it looks healthy.
            wr_lo = wr - 1.96 * se
            # Picks needed for a lower bound at this win rate to clear
            # breakeven. Infinite (reported as None) when the point estimate
            # is already below the bar — no amount of data rescues that.
            need = None
            if wr > be:
                need = int(np.ceil(wr * (1 - wr) * (1.96 / (wr - be)) ** 2))
            out["readiness"] = clean_dict({
                "n": n, "wins": int(y.sum()), "losses": int(n - y.sum()),
                "win_rate": wr, "breakeven": be, "edge": wr - be,
                "ci_low": wr_lo, "clears": bool(wr_lo > be),
                "needed": need, "floor": MIN_CONVICTION,
                "verdict": ("enough evidence — lower bound clears breakeven"
                            if wr_lo > be else
                            f"not yet — need about {need} settled picks"
                            if need else
                            "win rate is below breakeven; more data will not fix that"),
            })

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

    # ── the edge lab: live status of the pair experiment ────────────────
    # fetch_live_pairs.py snapshots Kalshi and DraftKings simultaneously;
    # pair_analysis.py settles them nightly. This block is a read-only
    # summary so the user can watch the verdict form on the site instead of
    # asking. It never fetches anything — files may simply not exist yet.
    try:
        lp = pd.read_csv(ROOT / "data" / "live_pairs.csv")
        lp["ts_utc"] = pd.to_datetime(lp["ts_utc"], errors="coerce")
        last = lp.sort_values("ts_utc").groupby("k_ticker", as_index=False).last()
        gap = (last["k_mid"] - last["book_prob"]).abs()
        fee = 0.07 * last["k_ask"] * (1 - last["k_ask"])
        tradeable = int(((last["book_prob"] - last["k_ask"]) > fee + 0.02).sum())
        settled_n, edge_units = 0, None
        rp = ROOT / "data" / "live_pairs_results.csv"
        if rp.exists():
            rr = pd.read_csv(rp, dtype=str)
            j = last.merge(rr, on="k_ticker", how="inner")
            settled_n = len(j)
            t = j[(j["book_prob"] - j["k_ask"])
                  > 0.07 * j["k_ask"] * (1 - j["k_ask"]) + 0.02]
            if len(t):
                won = (t["result"] == "yes").astype(int)
                edge_units = float((won - t["k_ask"]
                                    - 0.07 * t["k_ask"] * (1 - t["k_ask"])).sum())
        # ── the verdict, once the sample is in ────────────────────────────
        # The experiment was pre-registered to conclude at 150 settled legs.
        # Reporting "collecting..." past that point would be the same
        # goalpost-moving the readiness panel exists to prevent, so the
        # conclusion is computed and shipped the moment the sample arrives.
        target, concluded, verdict = 150, False, None
        n_trades = int(len(t)) if settled_n else 0
        if settled_n >= target:
            concluded = True
            spread_pts = float(gap.mean())
            wide = float((gap > 0.05).mean())
            verdict = (
                f"No edge. Across {settled_n} settled legs priced at the same "
                f"instant, Kalshi and the bookmaker differ by "
                f"{spread_pts * 100:.1f}c on average and never by more than 5c "
                f"({wide:.0%} of legs). Only {n_trades} leg"
                f"{'' if n_trades == 1 else 's'} ever cleared spread plus fee, "
                f"so there is no trade here to take. The +14.4% ROI "
                f"the retrospective test showed was entirely the 8-hour "
                f"lookahead in that test, not a real mispricing."
            ) if wide < 0.02 else (
                f"Divergence exists on {wide:.0%} of {settled_n} settled legs. "
                f"See pair_analysis.py for whether it survives cost."
            )
        out["edge_lab"] = clean_dict({
            "legs": int(len(last)), "games": int(lp["espn_id"].nunique()),
            "snapshots": int(len(lp)), "settled": settled_n, "target": target,
            "mean_gap": float(gap.mean()) if len(gap) else None,
            "tradeable_now": tradeable,
            "paper_units": edge_units,
            "trades": n_trades,
            "concluded": concluded,
            "verdict": verdict,
            "last_snapshot": (lp["ts_utc"].max().isoformat(timespec="minutes")
                              if lp["ts_utc"].notna().any() else None),
            "note": ("Kalshi vs DraftKings, priced at the same instant. "
                     "The value lane opens only if the gap survives spread "
                     "and fees at 150 settled legs."),
        })
    except (OSError, ValueError, KeyError):
        pass

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
    # Create the directory rather than refusing. The guard used to exit(1) if
    # the parent was missing, which was meant to catch a typo'd --out but in
    # practice just made the export the single fatal step in a CI run that was
    # otherwise fine. A missing output folder is not a reason to throw away a
    # completed export.
    out.parent.mkdir(parents=True, exist_ok=True)

    # Refuse to publish a board that says it has rows but ships none. The
    # `out["board"] = board` assignment was once dropped by accident: counts
    # kept reporting 183 while the array serialised empty, the export exited 0,
    # and the app rendered nothing for days. Every summary line looked healthy,
    # which is exactly why it went unnoticed — so the check is on the payload
    # itself, not on the variables used to build it.
    n_board = len(data.get("board") or [])
    n_claim = (data.get("board_counts") or {}).get("total", 0)
    if n_claim and not n_board:
        print(f"REFUSING TO WRITE: board_counts says {n_claim} rows but the "
              f"board array is empty — the payload would render a blank app.")
        raise SystemExit(1)
    if n_board and abs(n_board - n_claim) > 0:
        print(f"  warning: board has {n_board} rows but counts say {n_claim}")

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
