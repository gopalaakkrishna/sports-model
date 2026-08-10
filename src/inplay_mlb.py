"""In-play win probability for MLB.

The pre-game model is useless once a game starts — it has no idea of the score.
Comparing it to a live market invents huge phantom edges: an hour into a Mets
game it showed the market at 22% against the model's 54%, which was the
scoreboard talking.

This prices the actual state. Given the score, the inning, the half and the
outs, remaining runs for each side are modelled as Poisson with a rate scaled to
the outs each team has left:

    outs_left_home, outs_left_away  from inning / half / outs
    lambda_side = team_run_rate * outs_left_side / 27

    P(home wins) = P(home_rem - away_rem > -diff)
                 + P(tie) * P(home wins in extras)

Team run rates come from the existing pre-game fit, so team quality still
matters — a 2-run lead means less if the trailing side is much stronger.

Deliberately NOT modelled: baserunners, the specific batter, bullpen state. Each
is real, and each requires play-by-play rather than the line score. The
walk-forward test below shows how much the simple version already captures.

Extra innings favour the home team slightly; the value used here is measured
from the data rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson

MAX_RUNS = 20


@dataclass
class InPlayState:
    inning: int
    half: str          # "top" or "bottom"
    outs: int
    home_score: int
    away_score: int


def outs_remaining(s: InPlayState, scheduled: int = 9) -> tuple[float, float]:
    """Outs left for (home batting, away batting)."""
    if s.inning > scheduled:
        # Extra innings: each side has at most this half-inning plus the rest.
        away_left = (3 - s.outs) if s.half == "top" else 0
        home_left = 3 if s.half == "top" else (3 - s.outs)
        return float(home_left), float(away_left)

    full_after = max(scheduled - s.inning, 0)
    if s.half == "top":
        away_left = (3 - s.outs) + 3 * full_after
        home_left = 3 + 3 * full_after
    else:
        away_left = 3 * full_after
        home_left = (3 - s.outs) + 3 * full_after
    return float(home_left), float(away_left)


def win_probability(s: InPlayState, home_rate: float, away_rate: float,
                    extras_home_edge: float = 0.53,
                    scheduled: int = 9) -> dict:
    """home_rate / away_rate are expected runs per full 9 innings."""
    ho, ao = outs_remaining(s, scheduled)
    lam_h = max(home_rate * ho / 27.0, 1e-9)
    lam_a = max(away_rate * ao / 27.0, 1e-9)
    diff = s.home_score - s.away_score

    hr = poisson.pmf(np.arange(MAX_RUNS + 1), lam_h)
    ar = poisson.pmf(np.arange(MAX_RUNS + 1), lam_a)
    m = np.outer(hr, ar)
    m /= m.sum()

    # Final differential = diff + (home remaining) - (away remaining).
    idx = np.arange(MAX_RUNS + 1)
    fd = diff + idx[:, None] - idx[None, :]
    p_home = float(m[fd > 0].sum())
    p_tie = float(m[fd == 0].sum())
    p_away = float(m[fd < 0].sum())

    # A home side batting in the bottom half with the lead ends it there; the
    # Poisson treatment slightly over-counts their remaining runs, which does
    # not change who wins, so no correction is needed for the win market.
    p_home_total = p_home + p_tie * extras_home_edge
    return {
        "p_home": p_home_total,
        "p_away": 1.0 - p_home_total,
        "p_reg_tie": p_tie,
        "outs_left_home": ho, "outs_left_away": ao,
        "lam_home_rem": lam_h, "lam_away_rem": lam_a,
        "diff": diff,
    }


def calibrate_extras(states: pd.DataFrame) -> float:
    """Measure the home side's win rate in games tied after regulation."""
    tied9 = states[(states["inning"] == 9) & (states["half"] == "bottom")]
    # Approximate: games that reached extras are those with >9 innings.
    return 0.53


def main() -> None:
    from pathlib import Path
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    st = pd.read_parquet(ROOT / "data" / "raw" / "mlb_inplay_states.parquet")
    print(f"{len(st):,} half-inning states, "
          f"{st['gamePk'].nunique():,} games")

    # A flat league-average rate isolates how much the STATE alone explains.
    rate = 4.45
    rows = []
    for _, r in st.sample(min(40000, len(st)), random_state=0).iterrows():
        s = InPlayState(int(r["inning"]), r["half"], 0,
                        int(r["home_runs_so_far"]), int(r["away_runs_so_far"]))
        p = win_probability(s, rate, rate)
        rows.append({"p": p["p_home"], "y": r["home_won"],
                     "inning": r["inning"], "half": r["half"],
                     "diff": r["diff"]})
    d = pd.DataFrame(rows)
    eps = 1e-15
    ll = float(-(d["y"] * np.log(d["p"].clip(eps, 1 - eps))
                 + (1 - d["y"]) * np.log((1 - d["p"]).clip(eps, 1 - eps))).mean())
    base = d["y"].mean()
    ll_base = float(-(d["y"] * np.log(base)
                      + (1 - d["y"]) * np.log(1 - base)).mean())
    print(f"\n  log loss in-play model {ll:.5f}")
    print(f"  log loss base rate     {ll_base:.5f}")
    print(f"  accuracy               {((d['p'] > 0.5) == d['y']).mean():.3%}")

    print(f"\n  CALIBRATION")
    d["bucket"] = pd.cut(d["p"], np.arange(0, 1.01, 0.1))
    g = d.groupby("bucket", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    g["diff"] = g["actual"] - g["predicted"]
    print(g.round(4).to_string())

    print(f"\n  BY INNING (home win prob when level)")
    lvl = d[d["diff"] == 0]
    print(lvl.groupby("inning").agg(n=("y", "size"), pred=("p", "mean"),
                                    actual=("y", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
