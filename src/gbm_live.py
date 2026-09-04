"""Live GBM predictions for unplayed fixtures, blended with Dixon-Coles.

WHAT THIS ADDS

gbm_model.py established the gain offline: blending a gradient booster into
Dixon-Coles is worth +0.00053 RPS, 95% CI [+0.00033, +0.00074], out of sample
and significant in four of five majors. That was a backtest. This is the part
that makes it reach the board.

THE AWKWARD BIT: FEATURES FOR A MATCH THAT HAS NOT HAPPENED

Every rolling feature is defined as "this team's last N matches", which for a
played match means shifting the series by one. For an UNPLAYED fixture there
is nothing to shift away from — the row has no result yet.

The trick is to append the fixtures to the history as rows with no result and
compute the rolling features over the combined, date-sorted frame. A fixture
then naturally picks up the team's genuine last N played matches, and because
its own result is NaN it can never contribute to anyone else's window either.
That keeps a single code path for backtest and live, which matters: two
implementations of the same feature is how a backtest and production quietly
stop measuring the same thing.

WHY THE MODEL IS RETRAINED EVERY RUN

Fitting on ~20k rows takes a few seconds, so there is no reason to persist a
booster and every reason not to: a stale pickle silently predicting from
last season's ratings is a failure mode with no symptom. Refit each run and
the model is always as current as the data.

THE BLEND WEIGHT IS FITTED, NOT CHOSEN

Held in data/processed/gbm_blend.json and refit by `--refit-weight` on the
walk-forward, never hand-set. Athena's published 50/50 is hers, not ours —
at 0.50 our blend measured no better than Dixon-Coles alone, because our
booster is weaker. Copying the ratio would have bought nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

import gbm_model as GM

BLEND = ROOT / "data" / "processed" / "gbm_blend.json"
# Used only if the fitted file is missing. Deliberately 0.0 — no file means no
# evidence, and no evidence means ship Dixon-Coles unchanged rather than guess.
FALLBACK_WEIGHT = 0.0


def load_weight() -> float:
    if not BLEND.exists():
        return FALLBACK_WEIGHT
    try:
        return float(json.loads(BLEND.read_text(encoding="utf-8"))["weight"])
    except (ValueError, KeyError, OSError):
        return FALLBACK_WEIGHT


def _elo_with_pending(df: pd.DataFrame, k: float = 20.0,
                      ha: float = 60.0) -> pd.DataFrame:
    """Elo that tolerates rows with no result.

    gbm_model.elo assumes every row settles. Here the tail of the frame is
    unplayed fixtures, which must receive a PRE-match rating and must not
    update anyone's rating afterwards.
    """
    r: dict[str, float] = {}
    eh, ea = np.empty(len(df)), np.empty(len(df))
    for i, (h, a, res) in enumerate(zip(df["HomeTeam"], df["AwayTeam"],
                                        df["FTR"])):
        rh, ra = r.get(h, 1500.0), r.get(a, 1500.0)
        eh[i], ea[i] = rh, ra
        if res not in ("H", "D", "A"):
            continue                      # unplayed: rate it, do not learn
        exp_h = 1.0 / (1.0 + 10 ** (-((rh + ha) - ra) / 400.0))
        sc = 1.0 if res == "H" else 0.5 if res == "D" else 0.0
        r[h] = rh + k * (sc - exp_h)
        r[a] = ra + k * ((1.0 - sc) - (1.0 - exp_h))
    return pd.DataFrame({"elo_h": eh, "elo_a": ea, "elo_d": eh - ea},
                        index=df.index)


def build_live(hist: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """History plus pending fixtures, with causal features on every row.

    `fixtures` needs Div / Date / HomeTeam / AwayTeam and the Dixon-Coles
    forecast (m_home, m_draw, m_away, lam_h, lam_a) so the booster sees the
    same stacked feature it was trained on.
    """
    hist = hist.copy()
    fixtures = fixtures.copy()

    # Strip any form/diff/elo columns already present. A caller may hand us a
    # frame that has been through gbm_model.build, and concatenating freshly
    # computed features onto stale ones of the same name yields DUPLICATE
    # columns — after which `df[hc] - df[ac]` is a DataFrame subtraction and
    # the whole feature set is quietly wrong rather than loudly broken.
    def _stale(c: str) -> bool:
        return (c.startswith(("h_", "a_", "d_")) and c.split("_")[-1].isdigit()) \
            or c in ("elo_h", "elo_a", "elo_d", "h_rest", "a_rest",
                     "h_played", "a_played", "h_mid", "a_mid", "_pending", "y")

    hist = hist.drop(columns=[c for c in hist.columns if _stale(c)])
    fixtures = fixtures.drop(columns=[c for c in fixtures.columns if _stale(c)])

    hist["_pending"] = False
    fixtures["_pending"] = True
    for c in ("FTHG", "FTAG", "FTR"):
        if c not in fixtures.columns:
            fixtures[c] = np.nan

    df = pd.concat([hist, fixtures], ignore_index=True, sort=False)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "_pending"]).reset_index(drop=True)

    df = pd.concat([df.drop(columns=[c for c in ("elo_h", "elo_a", "elo_d")
                                     if c in df.columns]),
                    _elo_with_pending(df)], axis=1)

    form = GM.team_form(df)
    fh = (form[form.is_home == 1].drop(columns=["is_home", "team"])
          .set_index("mid").add_prefix("h_").reindex(df.index))
    fa = (form[form.is_home == 0].drop(columns=["is_home", "team"])
          .set_index("mid").add_prefix("a_").reindex(df.index))
    df = pd.concat([df, fh, fa], axis=1)

    for w in GM.WINDOWS:
        for col in ("pts", "gf", "ga", "sf", "sa", "stf", "sta", "xgf", "xga"):
            hc, ac = f"h_{col}_{w}", f"a_{col}_{w}"
            if hc in df and ac in df:
                df[f"d_{col}_{w}"] = df[hc] - df[ac]
    df["y"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})
    return df


def predict(hist: pd.DataFrame, fixtures: pd.DataFrame,
            weight: float | None = None, verbose: bool = True) -> pd.DataFrame:
    """Blended 1X2 for each pending fixture.

    Returns the fixtures frame with p_home / p_draw / p_away (blended) plus
    the raw gbm_* columns, so a caller can see what each side contributed.
    """
    import lightgbm as lgb

    w = load_weight() if weight is None else weight
    df = build_live(hist, fixtures)
    feats = GM.feature_cols(df)
    train = df[~df["_pending"] & df["y"].notna()]
    pend = df[df["_pending"]]
    if pend.empty:
        return fixtures.iloc[0:0]
    if len(train) < GM.MIN_TRAIN or w <= 0:
        # Not enough history, or no fitted weight: hand back Dixon-Coles
        # untouched rather than a half-trained booster's opinion.
        if verbose:
            print(f"  gbm_live: falling back to Dixon-Coles "
                  f"(train={len(train)}, weight={w:.2f})")
        out = pend.copy()
        out["p_home"], out["p_draw"], out["p_away"] = (
            out["m_home"], out["m_draw"], out["m_away"])
        for c in ("g_home", "g_draw", "g_away"):
            out[c] = np.nan
        out["blend_weight"] = 0.0
        return out

    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=400,
        learning_rate=0.03, num_leaves=31, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, verbose=-1, n_jobs=-1)
    m.fit(train[feats], train["y"].astype(int))
    G = m.predict_proba(pend[feats])

    out = pend.copy()
    out["g_home"], out["g_draw"], out["g_away"] = G[:, 0], G[:, 1], G[:, 2]
    D = out[["m_home", "m_draw", "m_away"]].to_numpy(float)
    D = D / D.sum(axis=1, keepdims=True)
    P = (1 - w) * D + w * G
    P = P / P.sum(axis=1, keepdims=True)
    out["p_home"], out["p_draw"], out["p_away"] = P[:, 0], P[:, 1], P[:, 2]
    out["blend_weight"] = w
    if verbose:
        print(f"  gbm_live: trained on {len(train):,} matches, "
              f"blended {len(out)} fixtures at w={w:.2f}")
    return out


def refit_weight(verbose: bool = True) -> float:
    """Refit the blend weight on the walk-forward and persist it."""
    src = ROOT / "data" / "processed" / "gbm_preds.parquet"
    if not src.exists():
        print(f"no walk-forward at {src}; run gbm_model.py first")
        return load_weight()
    d = pd.read_parquet(src)
    d = d[d["g_home"].notna()]
    P_dc = np.array(d[["m_home", "m_draw", "m_away"]].to_numpy(float), copy=True)
    P_dc /= P_dc.sum(axis=1, keepdims=True)
    P_gb = np.array(d[["g_home", "g_draw", "g_away"]].to_numpy(float), copy=True)
    y = d["y"].to_numpy(int)
    grid = np.linspace(0, 1, 21)
    scores = [(w, GM.rps((1 - w) * P_dc + w * P_gb, y)) for w in grid]
    w, best = min(scores, key=lambda t: t[1])
    base = GM.rps(P_dc, y)
    BLEND.parent.mkdir(parents=True, exist_ok=True)
    BLEND.write_text(json.dumps({
        "weight": float(w),
        "n": int(len(d)),
        "rps_dc": round(base, 5),
        "rps_blend": round(best, 5),
        "rps_gain": round(base - best, 5),
        "note": ("Weight on the GBM in the blend, fitted on the walk-forward, "
                 "never hand-set. Athena's published 50/50 is hers: at 0.50 "
                 "our blend measured no better than Dixon-Coles alone."),
    }, indent=2), encoding="utf-8")
    if verbose:
        print(f"blend weight {w:.2f}  RPS {base:.5f} -> {best:.5f} "
              f"(gain {base - best:+.5f}) on n={len(d):,}")
        print(f"saved -> {BLEND}")
    return float(w)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-weight", action="store_true")
    a = ap.parse_args()
    if a.refit_weight:
        refit_weight()
    else:
        print(f"current blend weight: {load_weight():.2f}")
        if BLEND.exists():
            print(BLEND.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
