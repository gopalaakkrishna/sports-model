"""Margin/total model for NFL and WNBA — one fit, three markets.

Football and basketball scores are nothing like goals or runs: NFL games average
44 points and WNBA 163, with margins that are close to normally distributed
(NFL sd 14.6, WNBA sd 13.8). A Poisson goal model is the wrong shape entirely,
so this is a different family.

Each team carries an offensive and a defensive rating:

    points_home = off_home + def_away + home_adv
    points_away = off_away + def_home

Fitted by ridge-regularised weighted least squares with exponential time decay —
closed form, so a fit is milliseconds rather than an optimisation.

From one fit come three markets:

    moneyline   P(home win)      = Phi(margin / sigma_margin)
    spread      P(home covers s) = Phi((margin - s) / sigma_margin)
    total       P(over T)        = 1 - Phi((T - total) / sigma_total)

sigma comes from the residuals rather than being assumed, and both sigmas are
reported so the caller can see how much irreducible noise the sport carries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class MarginFit:
    teams: list[str]
    off: np.ndarray
    dfn: np.ndarray
    # League-average points per team per game. Without this term the ridge
    # shrinks off/def toward zero and the only unpenalised parameter left to
    # explain ~22 points a side is the home-advantage term, which then inflates
    # enormously and biases every margin toward the home team.
    intercept: float
    home_adv: float
    sigma_margin: float
    sigma_total: float
    n_games: int
    eff_n: float
    team_eff_n: np.ndarray
    mean_total: float
    # Optional starting-quarterback effect on his OWN team's points. Empty when
    # the fit was run without QB data. Heavily regularised: there are hundreds
    # of QBs, most with a handful of starts, and the same thin-data trap that
    # made a 3-game promoted club look mid-table applies here.
    qbs: list[str] = None
    qb: np.ndarray = None
    qb_eff_n: np.ndarray = None

    def idx(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}

    def qb_idx(self) -> dict[str, int]:
        return {q: i for i, q in enumerate(self.qbs or [])}


# Validated by ablation on 3,025 NFL games. reg_qb=4 is the interior optimum:
# 0.5 and 1 lose significance (CI straddles zero), 12 and 30 still help but
# less, and 4 is best at -0.00830 log loss with CI [-0.01264, -0.00387].
BEST_REG_QB = 4.0


def fit(games: pd.DataFrame, as_of: pd.Timestamp, xi: float = 0.0025,
        reg: float = 8.0, max_years: float = 4.0,
        reg_qb: float = BEST_REG_QB) -> MarginFit:
    """Fit on completed games strictly before `as_of`.

    reg is deliberately heavier than in the goal models: point ratings are on a
    much larger scale, and NFL teams play only ~17 games a season, so a light
    penalty would let a 3-0 start look like a dynasty.
    """
    h = games[(games["date"] < as_of) & games["played"]]
    h = h[h["date"] >= as_of - pd.Timedelta(days=365.25 * max_years)]
    if len(h) < 100:
        raise ValueError(f"only {len(h)} games before {as_of}, need >=100")

    teams = sorted(set(h["home_team"]) | set(h["away_team"]))
    ti = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = h["home_team"].map(ti).to_numpy(np.int64)
    ai = h["away_team"].map(ti).to_numpy(np.int64)
    hs = h["home_score"].to_numpy(float)
    as_ = h["away_score"].to_numpy(float)

    age = (as_of - h["date"]).dt.total_seconds().to_numpy() / 86400.0
    w = np.exp(-xi * age)

    # Optional quarterback block.
    use_qb = (reg_qb > 0 and "home_qb_name" in h.columns
              and h["home_qb_name"].notna().sum() > 100)
    if use_qb:
        hq_raw = h["home_qb_name"].fillna("__unknown__")
        aq_raw = h["away_qb_name"].fillna("__unknown__")
        qbs = sorted(set(hq_raw) | set(aq_raw))
        qi = {q: k for k, q in enumerate(qbs)}
        hq = hq_raw.map(qi).to_numpy(np.int64)
        aq = aq_raw.map(qi).to_numpy(np.int64)
        nq = len(qbs)
    else:
        qbs, hq, aq, nq = [], None, None, 0

    # Two observations per game: home points, then away points.
    # Parameters: [off (n), def (n), intercept, home_adv, qb (nq)].
    m = len(h)
    rows = 2 * m
    p = 2 * n + 2 + nq
    I_MU, I_HA, I_QB = 2 * n, 2 * n + 1, 2 * n + 2
    X = np.zeros((rows, p))
    y = np.zeros(rows)
    sw = np.concatenate([w, w])

    X[np.arange(m), hi] = 1.0            # off_home
    X[np.arange(m), n + ai] = 1.0        # def_away
    X[np.arange(m), I_MU] = 1.0          # league level
    X[np.arange(m), I_HA] = 1.0          # home advantage
    y[:m] = hs

    X[np.arange(m, rows), ai] = 1.0      # off_away
    X[np.arange(m, rows), n + hi] = 1.0  # def_home
    X[np.arange(m, rows), I_MU] = 1.0    # league level (both sides)
    y[m:] = as_

    if use_qb:
        # A quarterback lifts his OWN team's points.
        X[np.arange(m), I_QB + hq] = 1.0
        X[np.arange(m, rows), I_QB + aq] = 1.0

    # Weighted ridge, closed form. The level and home-advantage terms are not
    # penalised — only the team (and QB) deviations are.
    Xw = X * sw[:, None]
    A = X.T @ Xw
    b = Xw.T @ y
    pen = np.eye(p) * reg
    pen[I_MU, I_MU] = 0.0
    pen[I_HA, I_HA] = 0.0
    if use_qb:
        for k in range(nq):
            pen[I_QB + k, I_QB + k] = reg_qb
    beta = np.linalg.solve(A + pen, b)

    off, dfn = beta[:n], beta[n:2 * n]
    mu, ha = float(beta[I_MU]), float(beta[I_HA])
    qb_vec = beta[I_QB:] if use_qb else None

    if use_qb:
        # Re-centre the QB effects so their playing-time-weighted mean is zero,
        # moving the offset into the intercept. This leaves predictions for
        # KNOWN quarterbacks completely unchanged, but it makes the "unknown QB
        # -> 0.0" fallback mean league-average instead of one point below it.
        #
        # This matters more than it sounds: nflverse does not fill in starters
        # for FUTURE games, so every upcoming fixture takes the fallback. With
        # an uncentred vector (weighted mean +1.01) that quietly subtracted ~2
        # points from every predicted total, putting the model under the line in
        # 84% of games — a systematic bias that looks exactly like an edge.
        qb_w = (np.bincount(hq, weights=w, minlength=nq)
                + np.bincount(aq, weights=w, minlength=nq))
        if qb_w.sum() > 0:
            qb_mean = float(np.average(qb_vec, weights=qb_w))
            qb_vec = qb_vec - qb_mean
            mu = mu + qb_mean

    pred_h = mu + off[hi] + dfn[ai] + ha
    pred_a = mu + off[ai] + dfn[hi]
    if use_qb:
        pred_h = pred_h + qb_vec[hq]
        pred_a = pred_a + qb_vec[aq]
    resid_margin = (hs - as_) - (pred_h - pred_a)
    resid_total = (hs + as_) - (pred_h + pred_a)
    sm = float(np.sqrt(np.average(resid_margin ** 2, weights=w)))
    st = float(np.sqrt(np.average(resid_total ** 2, weights=w)))

    team_eff = (np.bincount(hi, weights=w, minlength=n)
                + np.bincount(ai, weights=w, minlength=n))
    qb_eff = (np.bincount(hq, weights=w, minlength=nq)
              + np.bincount(aq, weights=w, minlength=nq)) if use_qb else None
    return MarginFit(teams=teams, off=off, dfn=dfn, intercept=mu, home_adv=ha,
                     sigma_margin=sm, sigma_total=st, n_games=m,
                     eff_n=float(w.sum()), team_eff_n=team_eff,
                     mean_total=float(np.average(hs + as_, weights=w)),
                     qbs=qbs, qb=qb_vec, qb_eff_n=qb_eff)


def predict(f: MarginFit, home: str, away: str,
            spread: float | None = None,
            total_line: float | None = None,
            home_qb: str | None = None,
            away_qb: str | None = None) -> dict | None:
    """spread follows the nflverse convention: POSITIVE means the home team is
    favoured by that many points, so home covers when margin > spread.

    Verified against the data rather than assumed: with `margin - spread > 0`
    the home side covers 47.6% of the time (a fair line should split ~50/50),
    whereas `margin + spread > 0` gives 58%, which is the giveaway that the sign
    is inverted.
    """
    ti = f.idx()
    if home not in ti or away not in ti:
        return None
    i, j = ti[home], ti[away]
    ph = f.intercept + f.off[i] + f.dfn[j] + f.home_adv
    pa = f.intercept + f.off[j] + f.dfn[i]

    qb_eff = {"home": 0.0, "away": 0.0, "eff_n_home": 0.0, "eff_n_away": 0.0}
    if f.qb is not None and f.qbs:
        qi = f.qb_idx()
        # An unknown or debut starter falls back to league average (0.0).
        if home_qb and home_qb in qi:
            k = qi[home_qb]
            qb_eff["home"] = float(f.qb[k])
            qb_eff["eff_n_home"] = float(f.qb_eff_n[k])
            ph += qb_eff["home"]
        if away_qb and away_qb in qi:
            k = qi[away_qb]
            qb_eff["away"] = float(f.qb[k])
            qb_eff["eff_n_away"] = float(f.qb_eff_n[k])
            pa += qb_eff["away"]

    margin = ph - pa
    total = ph + pa

    p_home = float(norm.cdf(margin / f.sigma_margin))
    out = {
        "exp_home_points": float(ph), "exp_away_points": float(pa),
        "exp_margin": float(margin), "exp_total": float(total),
        "p_home": p_home, "p_away": 1.0 - p_home,
        "sigma_margin": f.sigma_margin, "sigma_total": f.sigma_total,
        "eff_n_home": float(f.team_eff_n[i]),
        "eff_n_away": float(f.team_eff_n[j]),
        "eff_n_min": float(min(f.team_eff_n[i], f.team_eff_n[j])),
        "qb_home_effect": qb_eff["home"], "qb_away_effect": qb_eff["away"],
        "qb_eff_n_min": min(qb_eff["eff_n_home"], qb_eff["eff_n_away"]),
    }
    if spread is not None:
        # Home covers when the margin exceeds the spread it is giving.
        out["p_home_covers"] = float(norm.cdf((margin - spread) / f.sigma_margin))
        out["spread"] = spread
    if total_line is not None:
        out["p_over"] = float(1.0 - norm.cdf((total_line - total) / f.sigma_total))
        out["total_line"] = total_line
    return out
