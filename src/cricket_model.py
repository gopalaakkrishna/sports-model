"""The Hundred: team ratings and win probability.

Structure: 100 balls a side, two innings, mean first-innings total ~139.

Model: runs scored in an innings as batting strength against the opponent's
bowling, fitted by ridge-regularised weighted least squares with time decay —
the same machinery as the NFL/WNBA margin model, since totals are large and
roughly normal rather than Poisson counts.

    runs = mu + bat_team + bowl_opponent + home_bonus

Win probability comes from the expected run difference over the two innings,
scaled by the observed spread of margins.

THIS IS A THIN DATASET AND THE MODEL IS SIZED ACCORDINGLY.
189 men's matches across six seasons — MLB alone has 28,000 games. Regularisation
is heavy on purpose; with ~8 matches per team per season, light shrinkage would
let a good fortnight look like a dynasty.

Two data traps handled explicitly:

* **2026 rebrands.** Oval Invincibles became MI London, Northern Superchargers
  became Sunrisers Leeds, Manchester Originals became Manchester Super Giants.
  Verified against reporting, not inferred from the names — "MI London" sits
  close to "London Spirit" and guessing would have merged two different clubs.
  Unmapped, MI London would look like an expansion side rather than the
  three-time defending champion.

* **Home team is not the listed team.** Cricsheet's team order is arbitrary, and
  the listed-first side wins only 46% of the time. Home status is derived from
  the venue instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

# Verified 2026 franchise rebrands — history carries across.
REBRAND = {
    "Oval Invincibles": "MI London",
    "Northern Superchargers": "Sunrisers Leeds",
    "Manchester Originals": "Manchester Super Giants",
}

# Home grounds. Names in the data carry a stray leading quote.
VENUE_HOME = {
    "Kennington Oval": "MI London",
    "Lord's": "London Spirit",
    "The Rose Bowl": "Southern Brave",
    "Trent Bridge": "Trent Rockets",
    "Headingley": "Sunrisers Leeds",
    "Sophia Gardens": "Welsh Fire",
    "Old Trafford": "Manchester Super Giants",
    "Edgbaston": "Birmingham Phoenix",
}


def canon(team: str) -> str:
    return REBRAND.get(str(team).strip(), str(team).strip())


def venue_home(venue: str) -> str | None:
    v = str(venue).strip().strip('"').strip()
    for k, t in VENUE_HOME.items():
        if k.lower() in v.lower():
            return t
    return None


@dataclass
class CricketFit:
    teams: list[str]
    bat: np.ndarray
    bowl: np.ndarray
    intercept: float
    home_bonus: float
    sigma_margin: float
    n_matches: int
    eff_n: float
    team_eff_n: np.ndarray

    def idx(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}


def build(matches: pd.DataFrame, innings: pd.DataFrame) -> pd.DataFrame:
    """Join match info to innings totals, one row per innings."""
    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m["home_team"] = m["home_team"].map(canon)
    m["away_team"] = m["away_team"].map(canon)
    m["winner"] = m["winner"].map(lambda x: canon(x) if pd.notna(x) else x)
    m["venue_home"] = m["venue"].map(venue_home)

    inn = innings[innings["innings"].isin([1, 2])].copy()
    inn = inn[inn["match_id"] != "all_matches"]
    inn["match_id"] = inn["match_id"].astype(str)
    m["match_id"] = m["match_id"].astype(str)

    rows = []
    for _, g in m.iterrows():
        sub = inn[inn["match_id"] == g["match_id"]]
        if len(sub) != 2:
            continue
        # Cricsheet innings 1 is whichever side batted first; identify it from
        # the toss rather than assuming.
        first = g.get("toss_winner")
        dec = str(g.get("toss_decision", "")).lower()
        if pd.isna(first):
            continue
        first = canon(first)
        bat_first = first if dec == "bat" else (
            g["away_team"] if first == g["home_team"] else g["home_team"])
        bat_second = g["away_team"] if bat_first == g["home_team"] else g["home_team"]
        t1 = float(sub[sub["innings"] == 1]["total"].iloc[0])
        t2 = float(sub[sub["innings"] == 2]["total"].iloc[0])
        for team, opp, runs, order in ((bat_first, bat_second, t1, 1),
                                       (bat_second, bat_first, t2, 2)):
            rows.append({
                "match_id": g["match_id"], "date": g["date"],
                "team": team, "opponent": opp, "runs": runs,
                "innings_order": order,
                "is_home": int(team == g.get("venue_home")),
                "winner": g.get("winner"),
                "season": g.get("season"),
            })
    return pd.DataFrame(rows)


def fit(panel: pd.DataFrame, as_of: pd.Timestamp, xi: float = 0.0012,
        reg: float = 25.0, max_years: float = 6.0) -> CricketFit:
    h = panel[(panel["date"] < as_of) & panel["runs"].notna()]
    h = h[h["date"] >= as_of - pd.Timedelta(days=365.25 * max_years)]
    if len(h) < 60:
        raise ValueError(f"only {len(h)} innings before {as_of}")

    teams = sorted(set(h["team"]) | set(h["opponent"]))
    ti = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    bi = h["team"].map(ti).to_numpy(np.int64)
    oi = h["opponent"].map(ti).to_numpy(np.int64)
    y = h["runs"].to_numpy(float)
    hm = h["is_home"].to_numpy(float)
    w = np.exp(-xi * (as_of - h["date"]).dt.days.to_numpy())

    p = 2 * n + 2
    I_MU, I_H = 2 * n, 2 * n + 1
    X = np.zeros((len(h), p))
    X[np.arange(len(h)), bi] = 1.0
    X[np.arange(len(h)), n + oi] = 1.0
    X[:, I_MU] = 1.0
    X[:, I_H] = hm

    Xw = X * w[:, None]
    A = X.T @ Xw
    b = Xw.T @ y
    pen = np.eye(p) * reg
    pen[I_MU, I_MU] = 0.0
    pen[I_H, I_H] = 0.0
    beta = np.linalg.solve(A + pen, b)

    bat, bowl = beta[:n], beta[n:2 * n]
    mu, hb = float(beta[I_MU]), float(beta[I_H])

    pred = mu + bat[bi] + bowl[oi] + hb * hm
    resid = y - pred
    # Margin variance is roughly twice the innings residual variance.
    sig = float(np.sqrt(2.0) * np.sqrt(np.average(resid ** 2, weights=w)))
    eff = (np.bincount(bi, weights=w, minlength=n)
           + np.bincount(oi, weights=w, minlength=n))
    return CricketFit(teams, bat, bowl, mu, hb, sig, len(h),
                      float(w.sum()), eff)


def predict(f: CricketFit, team_a: str, team_b: str,
            home: str | None = None) -> dict | None:
    ti = f.idx()
    a, b = canon(team_a), canon(team_b)
    if a not in ti or b not in ti:
        return None
    i, j = ti[a], ti[b]
    ha = f.home_bonus if home and canon(home) == a else 0.0
    hb_ = f.home_bonus if home and canon(home) == b else 0.0
    ra = f.intercept + f.bat[i] + f.bowl[j] + ha
    rb = f.intercept + f.bat[j] + f.bowl[i] + hb_
    margin = ra - rb
    p_a = float(norm.cdf(margin / f.sigma_margin))
    return {
        "team_a": a, "team_b": b,
        "exp_runs_a": ra, "exp_runs_b": rb,
        "exp_margin": margin,
        "p_a": p_a, "p_b": 1.0 - p_a,
        "sigma": f.sigma_margin,
        "eff_n_min": float(min(f.team_eff_n[i], f.team_eff_n[j])),
    }
