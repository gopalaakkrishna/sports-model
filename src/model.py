"""Dixon-Coles bivariate Poisson goal model with exponential time decay.

Each team gets an attack and a defence rating. For a match between home i and
away j:

    lambda = exp(attack_i + defence_j + home_adv_division)   # home goals
    mu     = exp(attack_j + defence_i)                       # away goals

Goals are Poisson around those means, with the Dixon-Coles `tau` correction that
fixes the well-known underprediction of 0-0/1-0/0-1/1-1 by the independent
Poisson model.

Two deliberate design choices:

* Fits are done per COUNTRY, pooling all divisions together. Promotion and
  relegation link the divisions into one connected graph, so a newly promoted
  side arrives with a real rating instead of a cold start. A separate home
  advantage is fitted per division.
* An L2 ridge on the ratings both resolves the additive identifiability of
  attack/defence and shrinks thin-sample teams toward average.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# Divisions grouped by country so promoted/relegated teams share a rating scale.
COUNTRY_GROUPS = {
    "England": ["E0", "E1", "E2", "E3"],
    "Scotland": ["SC0", "SC1", "SC2", "SC3"],
    "Germany": ["D1", "D2"],
    "Italy": ["I1", "I2"],
    "Spain": ["SP1", "SP2"],
    "France": ["F1", "F2"],
    "Netherlands": ["N1"],
    "Belgium": ["B1"],
    "Portugal": ["P1"],
    "Turkey": ["T1"],
    "Greece": ["G1"],
}
LEAGUE_TO_COUNTRY = {lg: c for c, lgs in COUNTRY_GROUPS.items() for lg in lgs}

# Year-round / summer leagues, which use "CODE:League Name" division ids and run
# through the European off-season. Groups for these are derived from the data
# (see data.country_groups) since a country can carry more than one competition.
NEW_LEAGUE_COUNTRIES = {
    "ARG": "Argentina", "BRA": "Brazil", "MEX": "Mexico", "USA": "USA",
    "JPN": "Japan", "CHN": "China", "NOR": "Norway", "SWE": "Sweden",
    "FIN": "Finland", "IRL": "Ireland", "DNK": "Denmark", "POL": "Poland",
    "ROU": "Romania", "RUS": "Russia", "AUT": "Austria", "SWZ": "Switzerland",
}

MAX_GOALS = 15  # score matrix truncation; P(>15 goals) is ~0


@dataclass
class FitResult:
    teams: list[str]
    divisions: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home_adv: np.ndarray  # one per division
    rho: float
    n_matches: int
    eff_n: float  # sum of time weights — the "effective" sample size
    # Per-team effective sample. A promoted club with 3 recent games can land a
    # mid-table rating purely on which opponents it happened to face, so any
    # disagreement with the market involving a thin-data team is far more likely
    # to be model error than edge. Callers should check this before acting.
    team_eff_n: np.ndarray = None
    # Per-team deviation from its division's home advantage. Zero unless the fit
    # was run with reg_home > 0. Captures ground-specific effects the division
    # average cannot — altitude above all.
    home_team: np.ndarray = None

    def team_index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}

    def div_index(self) -> dict[str, int]:
        return {d: i for i, d in enumerate(self.divisions)}


def _tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction."""
    out = np.ones_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    out[m00] = 1.0 - lam[m00] * mu[m00] * rho
    out[m01] = 1.0 + lam[m01] * rho
    out[m10] = 1.0 + mu[m10] * rho
    out[m11] = 1.0 - rho
    return out


def _tau_grads(x, y, lam, mu, rho):
    """Partial derivatives of tau wrt lambda, mu and rho."""
    dl = np.zeros_like(lam)
    dm = np.zeros_like(lam)
    dr = np.zeros_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    dl[m00] = -mu[m00] * rho
    dm[m00] = -lam[m00] * rho
    dr[m00] = -lam[m00] * mu[m00]
    dl[m01] = rho
    dr[m01] = lam[m01]
    dm[m10] = rho
    dr[m10] = mu[m10]
    dr[m11] = -1.0
    return dl, dm, dr


def _neg_ll(params, hi, ai, di, x, y, w, n_teams, n_div, reg, reg_home=0.0):
    """Weighted negative log-likelihood and its analytic gradient.

    When reg_home > 0 the parameter vector carries a per-team home-advantage
    deviation after rho. One number per division cannot represent a ground like
    Toluca's Estadio Nemesio Diez at 2,670m, where altitude is a large and
    documented edge. The deviations are heavily shrunk, so a team only moves off
    its division's average with real evidence.
    """
    atk = params[:n_teams]
    dfn = params[n_teams : 2 * n_teams]
    hadv = params[2 * n_teams : 2 * n_teams + n_div]
    if reg_home > 0:
        rho = params[2 * n_teams + n_div]
        hteam = params[2 * n_teams + n_div + 1 :]
    else:
        rho = params[-1]
        hteam = None

    log_lam = atk[hi] + dfn[ai] + hadv[di]
    if hteam is not None:
        log_lam = log_lam + hteam[hi]
    log_mu = atk[ai] + dfn[hi]
    lam = np.exp(np.clip(log_lam, -10, 4))
    mu = np.exp(np.clip(log_mu, -10, 4))

    tau = _tau(x, y, lam, mu, rho)
    # tau can go non-positive for extreme rho; clip and let the optimiser back off.
    tau = np.clip(tau, 1e-10, None)

    ll = w * (np.log(tau) - lam + x * log_lam - mu + y * log_mu)
    total = ll.sum() - 0.5 * reg * (np.dot(atk, atk) + np.dot(dfn, dfn))
    if hteam is not None:
        total -= 0.5 * reg_home * np.dot(hteam, hteam)

    dtl, dtm, dtr = _tau_grads(x, y, lam, mu, rho)
    # d(loglik)/d(lambda) * lambda  -> chain rule straight onto the log-scale params
    g_lam = w * ((-1.0 + x / lam) + dtl / tau) * lam
    g_mu = w * ((-1.0 + y / mu) + dtm / tau) * mu
    g_rho = float((w * dtr / tau).sum())

    g_atk = np.bincount(hi, weights=g_lam, minlength=n_teams) + np.bincount(
        ai, weights=g_mu, minlength=n_teams
    )
    g_dfn = np.bincount(ai, weights=g_lam, minlength=n_teams) + np.bincount(
        hi, weights=g_mu, minlength=n_teams
    )
    g_hadv = np.bincount(di, weights=g_lam, minlength=n_div)

    g_atk -= reg * atk
    g_dfn -= reg * dfn

    if hteam is not None:
        # Per-team home effect enters log_lam exactly like attack does for the
        # home side, so it takes the same gradient contribution.
        g_hteam = np.bincount(hi, weights=g_lam, minlength=n_teams) - reg_home * hteam
        grad = np.concatenate([g_atk, g_dfn, g_hadv, [g_rho], g_hteam])
    else:
        grad = np.concatenate([g_atk, g_dfn, g_hadv, [g_rho]])
    return -total, -grad


def fit(
    matches: pd.DataFrame,
    as_of: pd.Timestamp,
    xi: float = 0.0018,
    reg: float = 2.0,
    max_years: float = 6.0,
    reg_home: float = 0.0,
    count_cols: tuple[str, str] = ("FTHG", "FTAG"),
) -> FitResult:
    """Fit the model on everything strictly before `as_of`.

    xi is the exponential time-decay rate per day (0.0018 ~= 1 year half-life).

    count_cols selects which counts to model. The default is goals. Passing
    ("HST", "AST") fits the identical structure to shots on target, which turns
    out to carry information goals alone do not — see model_shots.py.
    """
    hc, ac = count_cols
    hist = matches[
        (matches["Date"] < as_of) & matches[hc].notna() & matches[ac].notna()
    ]
    cutoff = as_of - pd.Timedelta(days=365.25 * max_years)
    hist = hist[hist["Date"] >= cutoff]
    if len(hist) < 100:
        raise ValueError(f"only {len(hist)} matches before {as_of}, need >=100")

    teams = sorted(set(hist["HomeTeam"]) | set(hist["AwayTeam"]))
    divisions = sorted(hist["Div"].unique())
    t_idx = {t: i for i, t in enumerate(teams)}
    d_idx = {d: i for i, d in enumerate(divisions)}

    hi = hist["HomeTeam"].map(t_idx).to_numpy(np.int64)
    ai = hist["AwayTeam"].map(t_idx).to_numpy(np.int64)
    di = hist["Div"].map(d_idx).to_numpy(np.int64)
    x = hist[hc].to_numpy(float)
    y = hist[ac].to_numpy(float)

    age_days = (as_of - hist["Date"]).dt.total_seconds().to_numpy() / 86400.0
    w = np.exp(-xi * age_days)

    n_teams, n_div = len(teams), len(divisions)
    p0 = np.concatenate(
        [np.zeros(n_teams), np.zeros(n_teams), np.full(n_div, 0.25), [-0.03]]
    )
    bounds = (
        [(-3, 3)] * n_teams + [(-3, 3)] * n_teams + [(-1, 1)] * n_div + [(-0.2, 0.2)]
    )
    if reg_home > 0:
        p0 = np.concatenate([p0, np.zeros(n_teams)])
        bounds = bounds + [(-0.8, 0.8)] * n_teams

    res = minimize(
        _neg_ll,
        p0,
        args=(hi, ai, di, x, y, w, n_teams, n_div, reg, reg_home),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    p = res.x
    home_team = (p[2 * n_teams + n_div + 1 :] if reg_home > 0
                 else np.zeros(n_teams))
    team_eff = (np.bincount(hi, weights=w, minlength=n_teams)
                + np.bincount(ai, weights=w, minlength=n_teams))
    return FitResult(
        teams=teams,
        divisions=divisions,
        attack=p[:n_teams],
        defence=p[n_teams : 2 * n_teams],
        home_adv=p[2 * n_teams : 2 * n_teams + n_div],
        rho=float(p[2 * n_teams + n_div]) if reg_home > 0 else float(p[-1]),
        n_matches=len(hist),
        eff_n=float(w.sum()),
        team_eff_n=team_eff,
        home_team=home_team,
    )


def score_matrix(lam: float, mu: float, rho: float) -> np.ndarray:
    """Joint distribution over (home goals, away goals)."""
    hg = poisson.pmf(np.arange(MAX_GOALS + 1), lam)
    ag = poisson.pmf(np.arange(MAX_GOALS + 1), mu)
    m = np.outer(hg, ag)
    m[0, 0] *= 1.0 - lam * mu * rho
    m[0, 1] *= 1.0 + lam * rho
    m[1, 0] *= 1.0 + mu * rho
    m[1, 1] *= 1.0 - rho
    return m / m.sum()


def predict(fitres: FitResult, home: str, away: str, div: str) -> dict | None:
    """Full match forecast. Returns None if either team is unseen."""
    t_idx, d_idx = fitres.team_index(), fitres.div_index()
    if home not in t_idx or away not in t_idx:
        return None
    i, j = t_idx[home], t_idx[away]
    # Fall back to the average home advantage for an unseen division.
    ha = fitres.home_adv[d_idx[div]] if div in d_idx else float(fitres.home_adv.mean())
    if fitres.home_team is not None:
        ha += float(fitres.home_team[i])

    lam = float(np.exp(np.clip(fitres.attack[i] + fitres.defence[j] + ha, -10, 4)))
    mu = float(np.exp(np.clip(fitres.attack[j] + fitres.defence[i], -10, 4)))
    m = score_matrix(lam, mu, fitres.rho)

    p_home = float(np.tril(m, -1).sum())
    p_draw = float(np.trace(m))
    p_away = float(np.triu(m, 1).sum())

    total = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
    p_over25 = float(m[total > 2.5].sum())
    p_btts = float(m[1:, 1:].sum())

    flat = m.flatten()
    top = flat.argsort()[::-1][:5]
    scorelines = [
        (int(k // (MAX_GOALS + 1)), int(k % (MAX_GOALS + 1)), float(flat[k])) for k in top
    ]

    eff_h = float(fitres.team_eff_n[i]) if fitres.team_eff_n is not None else float("nan")
    eff_a = float(fitres.team_eff_n[j]) if fitres.team_eff_n is not None else float("nan")

    return {
        "lambda_home": lam,
        "lambda_away": mu,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over25": p_over25,
        "p_btts": p_btts,
        "top_scorelines": scorelines,
        # Effective (time-weighted) matches behind each team's rating.
        "eff_n_home": eff_h,
        "eff_n_away": eff_a,
        "eff_n_min": min(eff_h, eff_a),
    }
