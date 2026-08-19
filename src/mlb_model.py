"""Poisson run-scoring model for MLB, with a starting-pitcher term.

For a game between home team i and away team j, with starting pitchers p_home
and p_away:

    lambda = exp(off_i + def_j + sp_{p_away} + home_adv)   expected home runs
    mu     = exp(off_j + def_i + sp_{p_home})              expected away runs

The pitcher term is the structural difference from the soccer model. A starting
pitcher throws roughly half a baseball game and is the largest single
game-to-game factor in the sport; soccer has no equivalent. Note the crossing:
the AWAY starter suppresses the HOME team's runs.

Pitchers are numerous (~1,500 across a decade) and many have few starts, so they
carry a heavier ridge than teams. Without it a pitcher with two good outings
would look like an ace — the same failure mode that made a promoted club look
mid-table in the soccer build.

Baseball has no draws. Regulation ties go to extra innings, so the Poisson tie
probability is split between the teams, slightly favouring the home side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_RUNS = 25          # run distributions are effectively zero beyond this
EXTRA_INNINGS_HOME = 0.53   # home team's share of games tied after regulation


@dataclass
class MLBFit:
    teams: list[str]
    pitchers: list[str]
    venues: list[str]
    off: np.ndarray
    dfn: np.ndarray
    sp: np.ndarray
    park: np.ndarray
    home_adv: float
    n_games: int
    eff_n: float
    team_eff_n: np.ndarray
    pitcher_eff_n: np.ndarray
    # Platt scaling on the win probability. Raw Poisson run models are wildly
    # over-confident for baseball: fitted shrinkage is ~0.41, i.e. the raw model
    # overstates its edge by roughly 2.4x. Uncalibrated it scores WORSE than
    # always predicting the base rate.
    calib_a: float = 1.0
    calib_b: float = 0.0

    def t_idx(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}

    def p_idx(self) -> dict[str, int]:
        return {p: i for i, p in enumerate(self.pitchers)}

    def v_idx(self) -> dict[str, int]:
        return {v: i for i, v in enumerate(self.venues)}


def _neg_ll(params, hi, ai, sph, spa, vi, x, y, w, n_t, n_p, n_v,
            reg_team, reg_sp, reg_park):
    off = params[:n_t]
    dfn = params[n_t:2 * n_t]
    sp = params[2 * n_t:2 * n_t + n_p]
    park = params[2 * n_t + n_p:2 * n_t + n_p + n_v]
    hadv = params[-1]

    # The park term lifts or suppresses BOTH sides' runs — Coors Field inflates
    # the whole game, not one team's offence.
    log_lam = off[hi] + dfn[ai] + sp[spa] + park[vi] + hadv
    log_mu = off[ai] + dfn[hi] + sp[sph] + park[vi]
    lam = np.exp(np.clip(log_lam, -4, 3))
    mu = np.exp(np.clip(log_mu, -4, 3))

    ll = w * (-lam + x * log_lam - mu + y * log_mu)
    total = (ll.sum()
             - 0.5 * reg_team * (np.dot(off, off) + np.dot(dfn, dfn))
             - 0.5 * reg_sp * np.dot(sp, sp)
             - 0.5 * reg_park * np.dot(park, park))

    # d/dtheta of (-lam + x*log lam) is (x - lam) for any theta entering log lam
    # with unit coefficient.
    rh = w * (x - lam)
    ra = w * (y - mu)

    g_off = np.bincount(hi, weights=rh, minlength=n_t) + np.bincount(ai, weights=ra, minlength=n_t)
    g_dfn = np.bincount(ai, weights=rh, minlength=n_t) + np.bincount(hi, weights=ra, minlength=n_t)
    g_sp = np.bincount(spa, weights=rh, minlength=n_p) + np.bincount(sph, weights=ra, minlength=n_p)
    g_pk = np.bincount(vi, weights=rh + ra, minlength=n_v)
    g_h = float(rh.sum())

    g_off -= reg_team * off
    g_dfn -= reg_team * dfn
    g_sp -= reg_sp * sp
    g_pk -= reg_park * park

    return -total, -np.concatenate([g_off, g_dfn, g_sp, g_pk, [g_h]])


def fit(games: pd.DataFrame, as_of: pd.Timestamp, xi: float = 0.0025,
        reg_team: float = 2.0, reg_sp: float = 8.0, reg_park: float = 4.0,
        max_years: float = 5.0, calib: tuple[float, float] = (1.0, 0.0)) -> MLBFit:
    """Fit on completed games strictly before `as_of`."""
    h = games[(games["date"] < as_of) & games["final"]
              & games["home_runs"].notna() & games["away_runs"].notna()]
    h = h[h["date"] >= as_of - pd.Timedelta(days=365.25 * max_years)]
    if len(h) < 500:
        raise ValueError(f"only {len(h)} games before {as_of}, need >=500")

    h = h.copy()
    # Games with an unknown starter get a shared "unknown" bucket rather than
    # being dropped — dropping them would bias the sample toward announced
    # matchups.
    h["home_sp"] = h["home_sp"].fillna("__unknown__")
    h["away_sp"] = h["away_sp"].fillna("__unknown__")

    h["venue"] = h["venue"].fillna("__unknown__")

    teams = sorted(set(h["home_team"]) | set(h["away_team"]))
    pitchers = sorted(set(h["home_sp"]) | set(h["away_sp"]))
    venues = sorted(set(h["venue"]))
    ti = {t: i for i, t in enumerate(teams)}
    pi = {p: i for i, p in enumerate(pitchers)}
    vi_map = {v: i for i, v in enumerate(venues)}

    hi = h["home_team"].map(ti).to_numpy(np.int64)
    ai = h["away_team"].map(ti).to_numpy(np.int64)
    sph = h["home_sp"].map(pi).to_numpy(np.int64)
    spa = h["away_sp"].map(pi).to_numpy(np.int64)
    vi = h["venue"].map(vi_map).to_numpy(np.int64)
    x = h["home_runs"].to_numpy(float)
    y = h["away_runs"].to_numpy(float)

    age = (as_of - h["date"]).dt.total_seconds().to_numpy() / 86400.0
    w = np.exp(-xi * age)

    n_t, n_p, n_v = len(teams), len(pitchers), len(venues)
    # Start off/def at log of league average runs per team per game.
    lvl = np.log(max((x.mean() + y.mean()) / 2.0, 0.5)) / 2.0
    p0 = np.concatenate([np.full(n_t, lvl), np.full(n_t, lvl),
                         np.zeros(n_p), np.zeros(n_v), [0.03]])
    bounds = ([(-3, 3)] * n_t + [(-3, 3)] * n_t + [(-1.5, 1.5)] * n_p
              + [(-1.0, 1.0)] * n_v + [(-0.5, 0.5)])

    res = minimize(_neg_ll, p0,
                   args=(hi, ai, sph, spa, vi, x, y, w, n_t, n_p, n_v,
                         reg_team, reg_sp, reg_park),
                   jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500, "ftol": 1e-9})
    p = res.x
    team_eff = (np.bincount(hi, weights=w, minlength=n_t)
                + np.bincount(ai, weights=w, minlength=n_t))
    pit_eff = (np.bincount(sph, weights=w, minlength=n_p)
               + np.bincount(spa, weights=w, minlength=n_p))
    return MLBFit(teams=teams, pitchers=pitchers, venues=venues,
                  off=p[:n_t], dfn=p[n_t:2 * n_t], sp=p[2 * n_t:2 * n_t + n_p],
                  park=p[2 * n_t + n_p:2 * n_t + n_p + n_v],
                  home_adv=float(p[-1]), n_games=len(h), eff_n=float(w.sum()),
                  team_eff_n=team_eff, pitcher_eff_n=pit_eff,
                  calib_a=calib[0], calib_b=calib[1])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict(f: MLBFit, home: str, away: str,
            home_sp: str | None = None, away_sp: str | None = None,
            venue: str | None = None) -> dict | None:
    ti, pi, vmap = f.t_idx(), f.p_idx(), f.v_idx()
    if home not in ti or away not in ti:
        return None
    i, j = ti[home], ti[away]

    def sp_val(name):
        """Unknown or debut starters fall back to league-average (0.0)."""
        if name and name in pi:
            k = pi[name]
            return f.sp[k], float(f.pitcher_eff_n[k])
        return 0.0, 0.0

    sp_h, eff_sph = sp_val(home_sp)
    sp_a, eff_spa = sp_val(away_sp)
    pk = float(f.park[vmap[venue]]) if venue and venue in vmap else 0.0

    lam = float(np.exp(np.clip(f.off[i] + f.dfn[j] + sp_a + pk + f.home_adv, -4, 3)))
    mu = float(np.exp(np.clip(f.off[j] + f.dfn[i] + sp_h + pk, -4, 3)))

    hr = poisson.pmf(np.arange(MAX_RUNS + 1), lam)
    ar = poisson.pmf(np.arange(MAX_RUNS + 1), mu)
    m = np.outer(hr, ar)
    m /= m.sum()

    p_home_reg = float(np.tril(m, -1).sum())
    p_tie = float(np.trace(m))
    p_away_reg = float(np.triu(m, 1).sum())
    # Ties are resolved in extra innings, slightly favouring the home side.
    p_home_raw = p_home_reg + p_tie * EXTRA_INNINGS_HOME
    p_away_raw = p_away_reg + p_tie * (1.0 - EXTRA_INNINGS_HOME)

    # Platt-scale the win probability. Without this the model is worse than
    # simply predicting the league home-win rate.
    z = np.log(max(p_home_raw, 1e-15) / max(1.0 - p_home_raw, 1e-15))
    p_home = float(_sigmoid(f.calib_a * z + f.calib_b))
    p_away = 1.0 - p_home

    total = np.add.outer(np.arange(MAX_RUNS + 1), np.arange(MAX_RUNS + 1))
    return {
        "lambda_home": lam, "lambda_away": mu,
        "p_home": p_home, "p_away": p_away,
        "p_home_uncalibrated": p_home_raw,
        "p_tie_regulation": p_tie,
        "exp_total": lam + mu,
        # DO NOT TRADE THESE. Measured 2026-08-19 on a 6,572-game walk-forward
        # (mlb_totals_bt.parquet): both lines score WORSE than a constant
        # base rate (-0.0247 at 8.5, -0.0245 at 9.5) and are wildly
        # overconfident — at a stated 80-90% the actual rate is 56%, and in
        # the >=60% band the over side states 69.7% against an actual 54.4%.
        #
        # The cause is structural, not a tuning problem: runs are modelled as
        # two independent Poissons, and baseball scoring is heavily
        # overdispersed (one big inning breaks the assumption). The WINNER
        # market survives this because the error largely cancels when you take
        # the difference of two similarly-misspecified distributions; a TOTAL
        # is exposed to it directly. Calibration cannot rescue a variance
        # misspecification of this size — it only rescales.
        #
        # Soccer totals DID pass the same gate (see totals_predict.py), so the
        # tempting inference "totals worked there, ship them here" is exactly
        # the mistake this comment exists to stop.
        "p_over_8_5": float(m[total > 8.5].sum()),
        "p_over_9_5": float(m[total > 9.5].sum()),
        "eff_n_home": float(f.team_eff_n[i]),
        "eff_n_away": float(f.team_eff_n[j]),
        "eff_n_sp_home": eff_sph,
        "eff_n_sp_away": eff_spa,
    }
