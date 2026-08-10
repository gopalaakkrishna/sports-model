"""Goals + shots-on-target hybrid.

Evidence for doing this at all, measured before building it. Predicting a team's
next-half goals per game from its first-half record:

    goals only        CV RMSE 0.3792
    goals + SOT       CV RMSE 0.3730   <- better
    SOT only          CV RMSE 0.3901   <- worse

So shots on target are a WEAKER standalone signal than goals, but carry
information goals do not: the SOT coefficient (+0.078) survives alongside goals
(+0.349). Both together beat either alone. This contradicts the common claim
that shots are simply "more stable than goals" — raw SOT ignores shot quality —
while still supporting a hybrid.

Method: fit the identical Dixon-Coles structure twice, once to goals and once to
shots on target. Convert the SOT rates to a goal scale using the league's
realised conversion rate, then blend the two expected-goal figures. The blend
weight is a hyperparameter to be validated, not assumed.

Shot data covers ~51% of matches (European divisions only, from 2005), so the
hybrid falls back to the goals-only model wherever shots are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import model as M


@dataclass
class HybridFit:
    goals: M.FitResult
    shots: M.FitResult | None
    conv_home: float          # goals per home shot on target
    conv_away: float
    weight: float             # weight on the shot-derived expected goals

    def teams(self) -> list[str]:
        return self.goals.teams


# Validated by ablation on 19,578 matches: 0.15 and 0.30 both beat goals-only
# with confidence intervals entirely below zero, 0.30 is best, and 0.50/0.75
# lose the effect and then reverse it. The optimum is interior, so the search
# range was wide enough — the same check that stopped the hyperparameter tuning
# from reaching a false conclusion.
BEST_SHOT_WEIGHT = 0.30


def fit(matches: pd.DataFrame, as_of: pd.Timestamp, xi: float = 0.0018,
        reg: float = 2.0, weight: float = BEST_SHOT_WEIGHT, reg_home: float = 0.0,
        max_years: float = 6.0) -> HybridFit:
    g = M.fit(matches, as_of, xi=xi, reg=reg, reg_home=reg_home,
              max_years=max_years, count_cols=("FTHG", "FTAG"))

    have_shots = matches.dropna(subset=["HST", "AST"])
    have_shots = have_shots[have_shots["Date"] < as_of]
    s = None
    conv_h = conv_a = 0.0
    if len(have_shots) >= 500:
        try:
            s = M.fit(matches, as_of, xi=xi, reg=reg, reg_home=reg_home,
                      max_years=max_years, count_cols=("HST", "AST"))
            recent = have_shots[have_shots["Date"] >=
                                as_of - pd.Timedelta(days=365.25 * max_years)]
            sh, sa = recent["HST"].sum(), recent["AST"].sum()
            conv_h = float(recent["FTHG"].sum() / sh) if sh > 0 else 0.0
            conv_a = float(recent["FTAG"].sum() / sa) if sa > 0 else 0.0
        except ValueError:
            s = None
    return HybridFit(goals=g, shots=s, conv_home=conv_h, conv_away=conv_a,
                     weight=weight)


def predict(hf: HybridFit, home: str, away: str, div: str) -> dict | None:
    base = M.predict(hf.goals, home, away, div)
    if base is None:
        return None
    if hf.shots is None or hf.weight <= 0:
        return base

    ti = hf.shots.team_index()
    di = hf.shots.div_index()
    if home not in ti or away not in ti:
        return base   # no shot ratings for these teams; fall back cleanly
    i, j = ti[home], ti[away]
    ha = (hf.shots.home_adv[di[div]] if div in di
          else float(hf.shots.home_adv.mean()))
    if hf.shots.home_team is not None:
        ha += float(hf.shots.home_team[i])

    lam_s = float(np.exp(np.clip(hf.shots.attack[i] + hf.shots.defence[j] + ha, -10, 4)))
    mu_s = float(np.exp(np.clip(hf.shots.attack[j] + hf.shots.defence[i], -10, 4)))
    lam_from_shots = lam_s * hf.conv_home
    mu_from_shots = mu_s * hf.conv_away
    if not (np.isfinite(lam_from_shots) and np.isfinite(mu_from_shots)):
        return base

    w = hf.weight
    lam = (1 - w) * base["lambda_home"] + w * lam_from_shots
    mu = (1 - w) * base["lambda_away"] + w * mu_from_shots

    m = M.score_matrix(lam, mu, hf.goals.rho)
    total = np.add.outer(np.arange(M.MAX_GOALS + 1), np.arange(M.MAX_GOALS + 1))
    flat = m.flatten()
    top = flat.argsort()[::-1][:5]
    return {
        "lambda_home": lam, "lambda_away": mu,
        "lambda_home_goals": base["lambda_home"],
        "lambda_home_shots": lam_from_shots,
        "p_home": float(np.tril(m, -1).sum()),
        "p_draw": float(np.trace(m)),
        "p_away": float(np.triu(m, 1).sum()),
        "p_over25": float(m[total > 2.5].sum()),
        "p_btts": float(m[1:, 1:].sum()),
        "top_scorelines": [(int(k // (M.MAX_GOALS + 1)), int(k % (M.MAX_GOALS + 1)),
                            float(flat[k])) for k in top],
        "eff_n_home": base["eff_n_home"], "eff_n_away": base["eff_n_away"],
        "eff_n_min": base["eff_n_min"],
    }
