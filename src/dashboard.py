"""Build a self-contained HTML dashboard of the prediction record.

Reads the ledger and today's prediction files and writes one HTML file with no
external dependencies — open it directly in a browser, no server needed.

What it deliberately shows, because a dashboard that only shows wins is a
marketing page:

  * log loss against the market on the same predictions, not just win rate.
    Win rate is close to meaningless here: back only 90% favourites and you get
    a beautiful win rate while losing money.
  * voided and corrected rows, with their reasons, in plain sight
  * a sample-size warning that stays up until the record is large enough to
    mean anything
  * calibration — of the calls given N% confidence, how many actually landed

    python dashboard.py            # writes reports/dashboard.html
    python dashboard.py --open     # ...and opens it
"""

from __future__ import annotations

import argparse
import html
import json
import re
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
import leagues as LG

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "processed" / "ledger.jsonl"
OUT = ROOT / "reports" / "dashboard.html"
ET = ZoneInfo("America/New_York")
EPS = 1e-15

SPORT_ICON = {"soccer": "⚽", "baseball": "⚾", "mlb": "⚾", "basketball": "🏀",
              "wnba": "🏀", "nfl": "🏈", "football": "🏈", "cricket": "🏏",
              "motorsports": "🏎️"}


def e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def load() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def logloss(p, y) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def score(g: pd.DataFrame) -> dict:
    y = g["won"].astype(int).to_numpy()
    out = {"n": len(g), "wins": int(y.sum()), "win_rate": float(y.mean()),
           "ll": logloss(g["model_prob"], y),
           "brier": float(((g["model_prob"] - y) ** 2).mean())}
    # Compare on the rows that actually carry a contemporaneous market price.
    # Requiring all of them meant one price-less row blanked the whole column,
    # which hides the only comparison that matters. Both sides are scored on the
    # SAME subset, so the difference stays honest; n_mkt says how many.
    sub = g[g["market_prob"].notna()]
    if len(sub):
        ys = sub["won"].astype(int).to_numpy()
        out["n_mkt"] = len(sub)
        out["mkt_ll"] = logloss(sub["market_prob"], ys)
        out["ll_mkt_subset"] = logloss(sub["model_prob"], ys)
        out["vs_market"] = out["ll_mkt_subset"] - out["mkt_ll"]
    if "pnl_units" in g and g["pnl_units"].notna().any():
        sub = g[g["pnl_units"].notna()]
        staked = float(sub["stake_units"].fillna(0).sum())
        out["pnl"] = float(sub["pnl_units"].sum())
        out["staked"] = staked
        out["roi"] = out["pnl"] / staked if staked else float("nan")
    return out


def pct(x, d=1) -> str:
    return "—" if x is None or x != x else f"{x:.{d}%}"


def date_in_event(event: str):
    """The '(YYYY-MM-DD)' the ledger appends to an event, if it has one.

    Left deliberately tz-NAIVE. Stamping a bare date as UTC midnight and then
    displaying it in Eastern rendered every fixture on the previous evening —
    2026-08-05 came out as "Tue 04 Aug 08:00 PM". A date with no clock is a
    date, and gets shown as one.
    """
    m = re.search(r"\((\d{4}-\d{2}-\d{2})\)", str(event))
    return pd.Timestamp(m.group(1)) if m else pd.NaT


def sort_key(t):
    """A comparable instant for ordering only.

    Start times are a mix: some sources give a real UTC timestamp, others only
    a date, which is kept naive so it is not shifted a day on display. Those
    two cannot be compared directly, so ordering uses a UTC projection while
    the displayed value stays exactly as parsed.
    """
    if t is None or t is pd.NaT or (isinstance(t, float) and t != t):
        return pd.Timestamp.max.tz_localize("UTC")
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def when_et(t) -> str:
    if t is None or t is pd.NaT or (isinstance(t, float) and t != t):
        return '<span class="dim">—</span>'
    try:
        ts = pd.Timestamp(t)
    except (TypeError, ValueError):
        return '<span class="dim">—</span>'
    if ts is pd.NaT:
        return '<span class="dim">—</span>'
    if ts.tzinfo is None:                      # date only — no clock to convert
        return f"{ts:%a %d %b}"
    return f"{ts.tz_convert(ET):%a %d %b · %I:%M %p}"


# The ledger says "mlb", the model runs say "baseball". Same sport; without
# this they render as two, each with its own half of the record.
SPORT_CANON = {"mlb": "baseball", "wnba": "basketball", "football": "nfl"}
SPORT_LABEL = {"baseball": "Baseball", "basketball": "Basketball",
               "soccer": "Soccer", "nfl": "NFL", "cricket": "Cricket",
               "motorsports": "Motorsports"}


def canon(sport) -> str:
    s = str(sport).strip().lower()
    return SPORT_CANON.get(s, s)


def label(sport) -> str:
    s = canon(sport)
    return SPORT_LABEL.get(s, s.title())


def inplay_row(ncols: int, uid: str) -> str:
    """A collapsible in-game panel, attached under a baseball fixture."""
    innings = "".join(f'<option value="{i}"{" selected" if i == 9 else ""}>'
                      f'{i}</option>' for i in range(1, 10))
    leads = "".join(
        f'<option value="{l}"{" selected" if l == 2 else ""}>'
        f'{"home +" + str(l) if l > 0 else "away +" + str(-l) if l < 0 else "level"}'
        f'</option>' for l in range(-6, 7))
    return (f'<tr><td colspan="{ncols}"><details><summary>live situation — '
            f'what is this lead worth?</summary><div class="inplay" id="{e(uid)}">'
            f'<label>entering inning</label><select class="f-inn">{innings}</select>'
            f'<label>score</label><select class="f-lead">{leads}</select>'
            f'<div class="f-out"></div></div></details></td></tr>')


def icon(sport) -> str:
    return SPORT_ICON.get(str(sport).lower(), "•")


def name_pick(pick: str, event: str) -> str:
    """Turn HOME/AWAY into the actual team name.

    A row reading 'Detroit @ Seattle ... HOME' was once acted on as the wrong
    side. The side being backed must never require reading the fixture and
    working out which end of it 'HOME' points at.
    """
    p = str(pick).upper()
    if p not in ("HOME", "AWAY"):
        return p
    ev = re.sub(r"\s*\(\d{4}-\d{2}-\d{2}\)\s*$", "", str(event)).strip()
    for sep, flip in ((" v ", False), (" vs ", False), (" @ ", True)):
        if sep in ev:
            a, b = [x.strip() for x in ev.split(sep, 1)]
            home, away = (b, a) if flip else (a, b)
            return (home if p == "HOME" else away).upper()
    return p


def _tier(gap) -> str:
    if gap != gap:
        return "n/a"
    return "HIGH" if gap < 0.03 else "MEDIUM" if gap < 0.07 else "LOW"


DATE_ONLY = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*$")


def to_utc(x):
    """Parse whatever the source file used for a start time. NaT if absent.

    A bare date is returned tz-NAIVE. Stamping it as UTC midnight and then
    rendering it in Eastern moved every WNBA fixture to 8pm the previous
    evening. A value with no clock in it cannot be converted between zones,
    so it is not converted at all.

    Everything with an actual time is UTC: Kalshi's occurrence_datetime and
    MLB's StatsAPI gameDate are both UTC, which is why a 9:40pm ET first pitch
    shows up as 01:40.
    """
    if x is None or (isinstance(x, float) and x != x) or x is pd.NaT:
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, str) and DATE_ONLY.match(x):
        return pd.Timestamp(x.strip())
    t = pd.to_datetime(x, errors="coerce", utc=True)
    return t if t is not pd.NaT else pd.NaT


def mlb_start(date, clock):
    """Combine MLB's separate date and UTC clock columns.

    The clock is UTC but the date column is the game's LOCAL date, so a 9:40pm
    ET first pitch is stored as date 2026-08-06 with clock 01:40 — which is
    01:40 on the 7th in UTC. Pasting them together put that game at 9:40pm on
    the 5th, a full day early. MLB games run roughly 16:00-04:00 UTC, so a
    clock before 10:00 belongs to the following calendar day.
    """
    if clock is None or (isinstance(clock, float) and clock != clock):
        return to_utc(date)
    d = pd.Timestamp(str(date))
    hh = int(str(clock).split(":")[0])
    if hh < 10:
        d += pd.Timedelta(days=1)
    return pd.to_datetime(f"{d:%Y-%m-%d} {clock}", errors="coerce", utc=True)


# Reliability bands. These describe how far the model sits from the market and
# nothing else:
#
#   ALIGNED  within 3 points of the market. Backtests put the model at or near
#            the closing line in this band. Agreement is NOT edge — by
#            construction there is nothing here to exploit, and after Kalshi's
#            ~1.7c fee near 50c an aligned position is negative EV.
#   WIDE     3-7 points apart. Shown, never tallied.
#   OFF      more than 7 points apart. Large disagreement backtested at 1.0373
#            log loss against the market's 0.9641 — the model being wrong, not
#            an edge. Also OFF on thin data or no market price.
#
# These were called TAKE / CAUTION / SKIP. "ALIGNED" is an instruction, and it was
# read as one: it sat in green next to fixtures whose only distinction was that
# the model AGREED with the price. With the live record now showing the model
# significantly worse than the market (gap +0.111, 95% CI [+0.024, +0.219]),
# labelling agreement as "ALIGNED" was the least accurate thing on the board.
def advice(tier: str, tradeable=None, thin: bool = False) -> str:
    if thin:
        return "OFF"
    if tier == "n/a":
        return "NO PRICE"
    if tier == "HIGH":
        return "OFF — illiquid" if tradeable is False else "ALIGNED"
    if tier == "MEDIUM":
        return "WIDE"
    return "OFF"


def human_label(gap, tradeable=None, thin: bool = False) -> str:
    """A label that explains itself, instead of one that needs a legend.

    ALIGNED / WIDE / OFF are internal band names for one number: how far the
    model sits from the market. Nobody should have to learn three words to read
    that, so the number is shown directly. "Off by 13" says what "OFF" meant and
    also says how far off, which is the part that actually varies.
    """
    if thin:
        return "thin data"
    if gap != gap:
        return "no market price"
    if tradeable is False:
        return "illiquid"
    pts = int(round(abs(gap) * 100))
    if pts < 3:
        return "matches market"
    return f"off by {pts}"


def upcoming_from_reports() -> list[dict]:
    """Whatever today's per-sport runs produced, tagged with sport and league.

    Each source file names its competition differently — Kalshi series ticker,
    football-data division code, or not at all — so every branch resolves
    through leagues.pretty() rather than passing its own string through.
    """
    R = ROOT / "reports"
    out: list[dict] = []

    # How far back the fallback below may reach. The fallback exists so the
    # board does not empty at midnight; three days covers that completely.
    # Unbounded, it did something much worse than emptying: when the soccer
    # parser broke on 2026-08-27 it silently reached back to the Aug 11
    # reports committed in this repo and published three-week-old picks under
    # a fresh timestamp for days. A stale board that admits it is stale is
    # recoverable; one that looks current is not.
    _MAX_FALLBACK_DAYS = 3

    def latest(prefix: str):
        """Today's file, or a recent one if today's is not written yet.

        Requiring today's date emptied the entire board the moment the clock
        passed midnight: the export ran, found no 2026-08-07 files, and shipped
        an upcoming list of zero. The slate does not vanish at midnight, so
        neither should the board — fall back to the newest file, but only
        within _MAX_FALLBACK_DAYS, so a dead predictor shows up as a missing
        section rather than as confident month-old picks.
        """
        cands = sorted(R.glob(f"{prefix}_*.csv"))
        if not cands:
            return None
        newest = cands[-1]
        m = re.search(r"(\d{4}-\d{2}-\d{2})", newest.name)
        if not m:
            return newest
        try:
            age = (pd.Timestamp.now().normalize()
                   - pd.Timestamp(m.group(1))).days
        except ValueError:
            return newest
        if age > _MAX_FALLBACK_DAYS:
            print(f"  {prefix}: newest report is {age}d old "
                  f"({newest.name}) — ignoring rather than publishing it as "
                  f"current. The predictor that writes it is not running.")
            return None
        return newest

    def add(sport, league, match, pick, model, mkt, start=None,
            tradeable=None, thin=False, side=None):
        # `pick` is for DISPLAY (a team name); `side` is HOME/AWAY/DRAW and is
        # what the ledger stores. settle.py resolves a fixture to HOME/AWAY/DRAW
        # and compares that to the stored pick, so a row locked with a team name
        # can never match its own result — it would settle as a loss whatever
        # happened. The two must not be conflated.
        mkt = float(mkt) if mkt == mkt else float("nan")
        gap = abs(model - mkt) if mkt == mkt else float("nan")
        tier = _tier(gap)
        out.append({"sport": sport, "league": LG.pretty(league), "match": match,
                    "pick": pick, "side": side, "model": float(model),
                    "mkt": mkt, "tier": tier, "start": to_utc(start),
                    "tradeable": tradeable, "thin": bool(thin),
                    "advice": advice(tier, tradeable, thin),
                    "label": human_label(gap, tradeable, thin)})

    # Soccer, priced off Kalshi — league comes from the series ticker.
    p = latest("kalshi_edge")
    if p:
        d = pd.read_csv(p)
        if not d.empty:
            d["mkt"] = d.groupby("event")["ask"].transform(
                lambda s: s / s.sum() if s.sum() > 0 else np.nan)
            for _, r in (d.sort_values("model", ascending=False)
                          .groupby("event", as_index=False).first().iterrows()):
                add("soccer", r["series"], r["match"], r["leg"],
                    r["model"], r["mkt"], side=str(r["leg"]).upper(),
                    start=r.get("when"),
                    tradeable=bool(r["tradeable"]) if "tradeable" in r else None,
                    thin=bool(r.get("thin_data", False)))

    # MLB / WNBA / cricket: one league each, market is the ask.
    for sport, league, fname, teamcol in [
            ("baseball", "MLB", "mlb_predictions", "team"),
            ("basketball", "WNBA", "wnba_kalshi", "team")]:
        p = latest(fname)
        if not p:
            continue
        d = pd.read_csv(p)
        if d.empty:
            continue
            # Group by (date, match), not match alone. Teams play multi-game
            # series — the SAME "Away @ Home" string legitimately recurs on
            # consecutive days. Grouping by match alone merged two distinct
            # games' ask prices into one normalization (silently halving a
            # real 59c to a displayed 29c) and then `.first()` kept only ONE
            # of the two games, dropping the other from the board entirely.
            # This is the exact failure the _ticker_date docstring already
            # warns about for Kalshi matching; it was just never applied here.
        d["_gkey"] = d["date"].astype(str) + "|" + d["match"].astype(str)
        d["mkt"] = d.groupby("_gkey")["ask"].transform(
            lambda s: s / s.sum() if s.sum() > 0 else np.nan)
        for _, r in (d.sort_values("model", ascending=False)
                      .groupby("_gkey", as_index=False).first().iterrows()):
            # MLB carries the date and clock in separate columns.
            when = (mlb_start(r["date"], r["start"]) if "start" in r
                    else r.get("date"))
            add(sport, league, r["match"], r[teamcol], r["model"], r["mkt"],
                side=str(r["side"]).upper() if "side" in r else None,
                start=when,
                tradeable=bool(r["tradeable"]) if "tradeable" in r else None)

    p = latest("hundred")
    if p:
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            fav_home = r["model_home"] >= 0.5
            add("cricket", "The Hundred", f"{r['home']} v {r['away']}",
                r["home"] if fav_home else r["away"],
                r["model_home"] if fav_home else 1 - r["model_home"],
                r["mkt_home"] if fav_home else 1 - r["mkt_home"],
                side="HOME" if fav_home else "AWAY", start=r.get("start"))

    # NFL has no Kalshi price in this file, so there is no market to compare
    # against and every row is n/a rather than silently HIGH.
    p = latest("nfl_upcoming")
    if p:
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            fav_home = r["p_home"] >= 0.5
            add("nfl", "NFL", f"{r['away']} @ {r['home']}",
                r["home"] if fav_home else r["away"],
                r["p_home"] if fav_home else 1 - r["p_home"], float("nan"),
                side="HOME" if fav_home else "AWAY",
                start=r.get("date"), thin=bool(r.get("thin", False)))
    return out


def inplay_grid() -> dict:
    """Win rate for the side leading by N entering inning I, from the state data.

    Powers the expandable panel on each baseball fixture. The point is not
    prediction — it is that the numbers move fastest exactly when you are least
    able to look them up, which is during the ninth.
    """
    p = ROOT / "data" / "raw" / "mlb_inplay_states.parquet"
    if not p.exists():
        return {}
    st = pd.read_parquet(p)
    st = st[st["half"] == "top"]
    grid = {}
    for inning in range(1, 10):
        for lead in range(-6, 7):
            sub = st[(st["inning"] == inning) & (st["diff"] == lead)]
            if len(sub) < 40:
                continue
            grid[f"{inning}|{lead}"] = [round(float(sub["home_won"].mean()), 4),
                                        int(len(sub))]
    return grid


INPLAY_JS = """
const G = window.__GRID__ || {};
function fmtPct(x){return (x*100).toFixed(0)+'%';}
function calc(panel){
  const inn=+panel.querySelector('.f-inn').value;
  const lead=+panel.querySelector('.f-lead').value;
  const out=panel.querySelector('.f-out');
  const k=inn+'|'+lead, g=G[k];
  if(!g){out.innerHTML='<span class="dim">Not enough historical games at '+
    'that exact state to quote a rate. Reporting nothing rather than a '+
    'number built on a handful of games.</span>';return;}
  const [pHome,n]=g;
  const leaderHome = lead>0;
  const p = leaderHome ? pHome : 1-pHome;
  if(lead===0){
    out.innerHTML='<b>Level.</b> Home win rate here is '+fmtPct(pHome)+
      ' across '+n.toLocaleString()+' games.';return;}
  const blown = 1/Math.max(1-p,1e-9);
  const risk = p/(1-p);
  out.innerHTML='<b>'+(leaderHome?'Home':'Away')+' leading by '+Math.abs(lead)+
    ' entering inning '+inn+' wins '+fmtPct(p)+'</b> of '+n.toLocaleString()+
    ' such games — blown roughly 1 in '+blown.toFixed(0)+'.'+
    '<div class="mt">At a market price of '+fmtPct(p)+' you would be risking '+
    risk.toFixed(1)+' units to win 1 on the remaining move. The price is your '+
    'own expected value either way, so holding and selling have the same EV — '+
    'what changes is the shape. Kalshi\\'s fee is 0.07·p·(1−p), so the exit is '+
    'cheapest exactly when the position is already won.</div>';
}
document.addEventListener('input',e=>{
  const p=e.target.closest('.inplay'); if(p) calc(p);});
document.addEventListener('toggle',e=>{
  if(e.target.tagName==='DETAILS'){const p=e.target.querySelector('.inplay');
    if(p) calc(p);}},true);
"""

CSS = """
:root{--bg:#fbfbfa;--fg:#1a1a18;--dim:#6b6b66;--line:#e3e3df;--card:#fff;
--good:#1a7f4b;--bad:#b3261e;--warn:#8a6100;--accent:#2f5fb8}
@media (prefers-color-scheme:dark){:root{--bg:#151614;--fg:#e8e8e4;--dim:#9a9a93;
--line:#2c2e2b;--card:#1d1f1c;--good:#4ec27f;--bad:#f2857c;--warn:#d9a441;--accent:#7ba6ee}}
:root[data-theme=dark]{--bg:#151614;--fg:#e8e8e4;--dim:#9a9a93;--line:#2c2e2b;
--card:#1d1f1c;--good:#4ec27f;--bad:#f2857c;--warn:#d9a441;--accent:#7ba6ee}
:root[data-theme=light]{--bg:#fbfbfa;--fg:#1a1a18;--dim:#6b6b66;--line:#e3e3df;
--card:#fff;--good:#1a7f4b;--bad:#b3261e;--warn:#8a6100;--accent:#2f5fb8}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",system-ui,sans-serif;
padding:28px 20px 80px}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.02em}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.1em;color:var(--dim);
margin:36px 0 12px;font-weight:600}
.sub{color:var(--dim);font-size:13px;margin-bottom:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi{font-size:26px;font-weight:650;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.klab{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);margin-top:2px}
.knote{font-size:11px;color:var(--dim);margin-top:6px}
.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:600px}
th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
color:var(--dim);font-weight:600;padding:6px 10px;border-bottom:1px solid var(--line)}
td{padding:7px 10px;border-bottom:1px solid var(--line);
font-variant-numeric:tabular-nums;vertical-align:top}
tr:last-child td{border-bottom:none}
.num{text-align:right}
.pick{font-weight:650;letter-spacing:.01em}
.tag{display:inline-block;font-size:10.5px;font-weight:650;letter-spacing:.05em;
padding:2px 7px;border-radius:5px;border:1px solid var(--line)}
.t-HIGH{color:var(--good);border-color:var(--good)}
.t-MEDIUM{color:var(--warn);border-color:var(--warn)}
.t-LOW,.t-na{color:var(--dim)}
.W{color:var(--good);font-weight:650}.L{color:var(--bad);font-weight:650}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--warn);
border-radius:8px;padding:12px 16px;font-size:13px;color:var(--dim);margin-top:12px}
.note b{color:var(--fg)}
.bar{height:7px;background:var(--line);border-radius:4px;overflow:hidden;min-width:70px}
.bar i{display:block;height:100%;background:var(--accent)}
.nowrap{white-space:nowrap;color:var(--dim);font-size:12.5px}
.empty{color:var(--dim);font-size:13.5px;padding:10px 0}
.sporthead{display:flex;align-items:baseline;gap:10px;margin:26px 0 2px;
font-size:17px;font-weight:650;letter-spacing:-.01em;text-transform:capitalize}
.sporthead span{font-size:11.5px;font-weight:600;letter-spacing:.06em;
text-transform:uppercase;color:var(--dim)}
.leaguehead{font-size:12px;font-weight:600;letter-spacing:.04em;color:var(--dim);
margin:14px 0 4px;padding-left:2px;border-left:3px solid var(--accent);
padding:2px 0 2px 9px}
.a-TAKE{color:var(--good);border-color:var(--good)}
.a-CAUTION{color:var(--warn);border-color:var(--warn)}
.a-SKIP,.a-NOPRICE{color:var(--dim)}
details{margin:0}
details>summary{cursor:pointer;font-size:12px;color:var(--accent);
list-style:none;padding:2px 0}
details>summary::-webkit-details-marker{display:none}
details>summary::before{content:"▸ ";font-size:10px}
details[open]>summary::before{content:"▾ "}
.inplay{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:12px 14px;margin:6px 0 10px;font-size:13px}
.inplay label{font-size:11px;text-transform:uppercase;letter-spacing:.06em;
color:var(--dim);margin-right:6px}
.inplay select{background:var(--bg);color:var(--fg);border:1px solid var(--line);
border-radius:6px;padding:3px 7px;font:inherit;font-size:12.5px;margin-right:16px}
.f-out{margin-top:10px;line-height:1.6}
.mt{margin-top:8px;color:var(--dim);font-size:12.5px}
.dim{color:var(--dim)}
footer{margin-top:44px;padding-top:16px;border-top:1px solid var(--line);
font-size:12px;color:var(--dim)}
"""


def table(headers, rows, cls="") -> str:
    if not rows:
        return '<div class="empty">Nothing here yet.</div>'
    h = "".join(f'<th class="{"num" if isinstance(x, tuple) else ""}">'
                f'{e(x[0] if isinstance(x, tuple) else x)}</th>' for x in headers)
    # A row given as a raw string is emitted verbatim — that is how the
    # full-width in-play drilldown gets attached under its fixture.
    body = "".join(r if isinstance(r, str) else "<tr>" + "".join(r) + "</tr>"
                   for r in rows)
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{h}</tr>' \
           f'</thead><tbody>{body}</tbody></table></div>'


def build(grid: dict) -> str:
    df = load()
    now = datetime.now(ET)
    parts = [f'<div class="wrap"><h1>Prediction record</h1>'
             f'<div class="sub">Generated {now:%a %d %b %Y, %I:%M %p} ET · '
             f'all figures from the locked ledger</div>']

    if df.empty:
        parts.append('<div class="empty">No predictions logged yet.</div></div>')
        return "".join(parts)

    voided = df[df.get("voided", pd.Series(False, index=df.index)).fillna(False)]
    corrected = df[df.get("correction_history").notna()] if "correction_history" in df \
        else df.iloc[0:0]
    live = df[~df.index.isin(voided.index)]
    settled = live[live["outcome"].notna()].copy()
    open_ = live[live["outcome"].isna()].copy()

    # ---- headline -------------------------------------------------------
    if len(settled):
        s = score(settled)
        vs = s.get("vs_market")
        vs_cls = "good" if vs is not None and vs < 0 else "bad"
        vs_txt = f"{vs:+.4f}" if vs is not None else "—"
        cards = [
            (f"{s['wins']}–{s['n'] - s['wins']}", "record", f"{pct(s['win_rate'])} win rate"),
            (f"{s['ll']:.4f}", "model log loss", "lower is better"),
            (f"{s['mkt_ll']:.4f}" if "mkt_ll" in s else "—", "market log loss",
             f"on the {s['n_mkt']} with a logged price" if "mkt_ll" in s
             else "no prices logged"),
            (f'<span class="{vs_cls}">{vs_txt}</span>', "vs market",
             ("beating market" if vs < 0 else "losing to market")
             if vs is not None else "—"),
        ]
        if "roi" in s:
            cards.append((f'<span class="{"good" if s["pnl"] > 0 else "bad"}">'
                          f'{s["pnl"]:+.2f}u</span>', "P&L", f"{pct(s.get('roi'))} ROI"))
    else:
        cards = [("0", "settled", "nothing resolved yet")]
    parts.append('<div class="grid">' + "".join(
        f'<div class="card"><div class="kpi">{k}</div>'
        f'<div class="klab">{e(l)}</div><div class="knote">{e(n)}</div></div>'
        for k, l, n in cards) + "</div>")

    if len(settled) < 100:
        parts.append(
            f'<div class="note"><b>{len(settled)} settled predictions is far too '
            f'few to conclude anything.</b> Separating a real 2% edge from noise '
            f'takes on the order of 1,000 bets. Everything below is bookkeeping, '
            f'not evidence. The log-loss comparison is the number that will '
            f'eventually matter; win rate never will.</div>')

    # Normalise league names before any grouping, or the same competition
    # appears twice under two spellings.
    for d_ in (settled, open_):
        if len(d_):
            d_["league_name"] = d_["league"].map(LG.pretty)
            d_["sport"] = d_["sport"].map(canon)

    # ---- breakdown tables ----------------------------------------------
    def breakdown(g: pd.DataFrame, label: str) -> list[list[str]]:
        s = score(g)
        vs = s.get("vs_market")
        vs_html = (f'<td class="num {"good" if vs < 0 else "bad"}">{vs:+.4f}</td>'
                   if vs is not None else '<td class="num">—</td>')
        return [[f"<td>{label}</td>",
                 f'<td class="num">{s["n"]}</td>',
                 f'<td class="num">{s["wins"]}–{s["n"] - s["wins"]}</td>',
                 f'<td class="num">{pct(s["win_rate"])}</td>',
                 f'<td class="num">{s["ll"]:.4f}</td>', vs_html,
                 f'<td class="num">{pct(s.get("roi")) if "roi" in s else "—"}</td>']]

    hdr = ["", ("n",), ("W–L",), ("win%",), ("log loss",), ("vs market",), ("ROI",)]
    if len(settled):
        parts.append("<h2>Record by sport</h2>")
        rows = []
        for sp, g in settled.groupby("sport"):
            rows += breakdown(g, f'{icon(sp)} {e(label(sp))}')
        parts.append(table(hdr, rows))

        parts.append("<h2>Record by league</h2>")
        rows = []
        for (sp, lg), g in settled.groupby(["sport", "league_name"]):
            rows += breakdown(g, f'{icon(sp)} {e(lg)}')
        parts.append(table(hdr, rows))

    # ---- open, nested sport > league, soonest first ----------------------
    parts.append(f"<h2>Open — awaiting result ({len(open_)})</h2>")
    if not len(open_):
        parts.append('<div class="empty">Nothing open.</div>')
    else:
        open_["start"] = open_["event"].map(date_in_event)
        open_["_k"] = open_["start"].map(sort_key)
        open_ = open_.sort_values("_k")
    for sp, gs in open_.groupby("sport", sort=False):
        parts.append(f'<div class="sporthead">{icon(sp)} {e(label(sp))} '
                     f'<span>{len(gs)}</span></div>')
        for lg, g in gs.groupby("league_name", sort=False):
            parts.append(f'<div class="leaguehead">{e(lg)} · {len(g)}</div>')
            rows = []
            for i, (_, r) in enumerate(g.sort_values("_k").iterrows()):
                mp = r.get("market_prob")
                has = mp == mp and mp is not None
                gap = (r["model_prob"] - mp) if has else None
                t = _tier(abs(gap)) if has else "n/a"
                adv = advice(t)
                rows.append([
                    f'<td class="nowrap">{when_et(r["start"])}</td>',
                    f'<td class="num">#{e(r["id"])}</td>',
                    f'<td><span class="tag a-{adv.split()[0]}">{e(adv)}</span></td>',
                    f'<td><span class="tag t-{t.replace("/", "")}">{e(t)}</span></td>',
                    f'<td class="pick">{e(name_pick(r["pick"], r["event"]))}</td>',
                    f'<td>{e(r["event"])}</td>',
                    f'<td class="num">{pct(r["model_prob"], 0)}</td>',
                    f'<td class="num">{pct(mp, 0) if has else "—"}</td>',
                    f'<td class="num">{f"{gap:+.0%}" if has else "—"}</td>'])
                if str(sp) in ("baseball", "mlb") and grid:
                    rows.append(inplay_row(9, f"o{i}{lg}"))
            parts.append(table([("date",), ("#",), "call", "conf", "bet on",
                                "fixture", ("model",), ("market",), ("gap",)],
                               rows))

    # ---- settled, nested sport > league ---------------------------------
    parts.append(f"<h2>Settled ({len(settled)})</h2>")
    if not len(settled):
        parts.append('<div class="empty">Nothing settled yet.</div>')
    for sp, gs in settled.groupby("sport"):
        s = score(gs)
        parts.append(f'<div class="sporthead">{icon(sp)} {e(label(sp))} '
                     f'<span>{s["wins"]}–{s["n"] - s["wins"]}</span></div>')
        for lg, g in gs.groupby("league_name"):
            sl = score(g)
            parts.append(f'<div class="leaguehead">{e(lg)} · '
                         f'{sl["wins"]}–{sl["n"] - sl["wins"]}</div>')
            rows = []
            g = g.assign(start=g["event"].map(date_in_event))
            g = g.assign(_k=g["start"].map(sort_key))
            for _, r in g.sort_values("_k", ascending=False).iterrows():
                won = bool(r["won"])
                mp = r.get("market_prob")
                has = mp == mp and mp is not None
                t = _tier(abs(r["model_prob"] - mp)) if has else "n/a"
                rows.append([
                    f'<td class="nowrap">{when_et(r["start"])}</td>',
                    f'<td class="num">#{e(r["id"])}</td>',
                    f'<td><span class="tag t-{t.replace("/", "")}">{e(t)}</span></td>',
                    f'<td class="pick">{e(name_pick(r["pick"], r["event"]))}</td>',
                    f'<td>{e(r["event"])}</td>',
                    f'<td class="num">{pct(r["model_prob"], 0)}</td>',
                    f'<td>{e(r["outcome"])}</td>',
                    f'<td class="{"W" if won else "L"}">'
                    f'{"WON" if won else "lost"}</td>'])
            parts.append(table([("date",), ("#",), "conf", "bet on", "fixture",
                                ("model",), "result", "outcome"], rows))

    # ---- calibration ----------------------------------------------------
    if len(settled) >= 5:
        parts.append("<h2>Calibration</h2>")
        parts.append('<div class="sub">Of the calls made at each confidence '
                     'level, how many actually landed. A calibrated model tracks '
                     'the diagonal.</div>')
        band = pd.cut(settled["model_prob"], [0, .4, .55, .7, .85, 1.0],
                      labels=["<40%", "40–55%", "55–70%", "70–85%", ">85%"])
        rows = []
        for k, g in settled.groupby(band, observed=True):
            act = float(g["won"].astype(int).mean())
            exp = float(g["model_prob"].mean())
            rows.append([
                f"<td>{e(k)}</td>", f'<td class="num">{len(g)}</td>',
                f'<td class="num">{exp:.0%}</td>',
                f'<td class="num">{act:.0%}</td>',
                f'<td><div class="bar"><i style="width:{act * 100:.0f}%"></i></div></td>'])
        parts.append(table(["confidence", ("n",), ("said",), ("actual",), "  "], rows))

    # ---- upcoming from today's model runs -------------------------------
    up = pd.DataFrame(upcoming_from_reports())
    if len(up):
        n_high = int((up["tier"] == "HIGH").sum())
        parts.append(f"<h2>Model output today — not yet locked ({len(up)}, "
                     f"{n_high} HIGH)</h2>")
        parts.append('<div class="sub">HIGH means the model sits within 3 points '
                     'of the market. That band is where it has measurably matched '
                     'the closing line; large disagreement has backtested as model '
                     'error, so LOW is a warning, not an opportunity. Rows with no '
                     'market price cannot be tiered at all and show n/a.</div>')
        # Soonest first. Anything without a parsable start time sorts last
        # rather than being dropped or guessed at.
        up["_k"] = up["start"].map(sort_key)
        up = up.sort_values("_k")
        for sp, gs in up.groupby("sport", sort=False):
            h = int((gs["tier"] == "HIGH").sum())
            parts.append(f'<div class="sporthead">{icon(sp)} {e(label(sp))} '
                         f'<span>{h} TAKE of {len(gs)}</span></div>')
            for lg, g in gs.groupby("league", sort=False):
                parts.append(f'<div class="leaguehead">{e(lg)} · {len(g)}</div>')
                rows = []
                for i, (_, r) in enumerate(g.sort_values("_k").iterrows()):
                    t, adv = str(r["tier"]), str(r["advice"])
                    rows.append([
                        f'<td class="nowrap">{when_et(r["start"])}</td>',
                        f'<td><span class="tag a-{adv.split()[0]}">{e(adv)}'
                        f'</span></td>',
                        f'<td><span class="tag t-{t.replace("/", "")}">{e(t)}'
                        f'</span></td>',
                        f'<td class="pick">'
                        f'{e(name_pick(r["pick"], r["match"]))}</td>',
                        f'<td>{e(r["match"])}</td>',
                        f'<td class="num">{pct(r["model"], 0)}</td>',
                        f'<td class="num">{pct(r["mkt"], 0)}</td>'])
                    if str(sp) == "baseball" and grid:
                        rows.append(inplay_row(7, f"u{sp}{lg}{i}"))
                parts.append(table([("start ET",), "call", "conf", "bet on",
                                    "fixture", ("model",), ("market",)], rows))

    # ---- disclosure -----------------------------------------------------
    backfilled = (df[df["backfilled"].fillna(False)] if "backfilled" in df
                  else df.iloc[0:0])
    if len(voided) or len(corrected) or len(backfilled):
        parts.append("<h2>Excluded, corrected and backfilled</h2>")
        parts.append('<div class="sub">Kept visible on purpose. A record that '
                     'quietly drops rows is not a record. Backfilled rows were '
                     'genuinely called at the time but written down afterwards — '
                     'they still count, and they are weaker evidence than a lock '
                     'made before kick-off.</div>')
        rows = []
        for _, r in voided.iterrows():
            rows.append([f'<td class="num">#{e(r["id"])}</td>',
                         "<td>voided</td>", f'<td>{e(r["event"])}</td>',
                         f'<td>{e(r.get("void_reason"))}</td>'])
        for _, r in backfilled.iterrows():
            rows.append([f'<td class="num">#{e(r["id"])}</td>',
                         "<td>backfilled</td>", f'<td>{e(r["event"])}</td>',
                         f'<td>{e(r.get("backfill_reason"))}</td>'])
        for _, r in corrected.iterrows():
            for h in (r.get("correction_history") or []):
                rows.append([f'<td class="num">#{e(r["id"])}</td>',
                             "<td>corrected</td>", f'<td>{e(r["event"])}</td>',
                             f'<td>was &ldquo;{e(h.get("reverted_outcome"))}&rdquo; — '
                             f'{e(h.get("reason"))}</td>'])
        parts.append(table([("#",), "action", "fixture", "reason"], rows))

    parts.append(
        '<footer>Scored by log loss against the market price recorded at lock '
        'time, not by win rate. Voided rows are excluded from every figure above. '
        'Regenerate with <code>python src/dashboard.py</code>.</footer></div>')
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", dest="open_",
                    help="open in the default browser when done")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    grid = inplay_grid()
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>Prediction record</title><style>{CSS}</style></head>'
           f'<body>{build(grid)}'
           f'<script>window.__GRID__={json.dumps(grid)};{INPLAY_JS}</script>'
           f'</body></html>')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out}  ({len(doc):,} bytes)")
    if args.open_:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()

