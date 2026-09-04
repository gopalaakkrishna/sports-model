"""Gradient-boosted 1X2 model, stacked on Dixon-Coles, blended 50/50.

WHY THIS EXISTS

Measured 2026-09-04: our Dixon-Coles sits at RPS 0.2008 (EPL) / 0.1962 (La
Liga) against the closing line's 0.1945 / 0.1913. Athena Huo publishes
0.1998 / 0.1960 on the same dataset and the same 4,180-match window — so we
are at parity, not behind.

Her two leagues differ in exactly one way, which prices the missing piece:

    La Liga  xG Dixon-Coles only        0.1960   (ours, goals-only: 0.1962)
    EPL      50/50 GBM + xG DC          0.1998   (ours, DC only:    0.2008)

xG is therefore worth ~0.0002 and the GBM blend ~0.0010. This file builds the
0.0010, which is the larger of the two and needs no new data source.

WHAT WOULD MAKE THIS FAKE, AND HOW EACH IS AVOIDED

A gradient booster will happily manufacture an edge out of leakage, and on
this problem the result would look plausible. Three specific hazards:

  1. Non-causal features. Every rolling statistic is computed per team then
     SHIFTED by one match, so a fixture never sees its own result. The shift
     happens before the pivot back to match level, so it cannot be undone by
     a later join.

  2. Training on the future. The booster is refit on a walk-forward cursor
     and only ever predicts matches strictly after its training cut. Refit
     every REFIT_DAYS rather than every match, because a full refit per
     fixture is ~50x the compute for a difference measured in the fourth
     decimal.

  3. Learning the market instead of the game. Closing odds are NOT features.
     They exist in the dataframe and are deliberately excluded — including
     them would produce a model that tracks the line, score beautifully
     against it, and tell us nothing about whether we can beat it.

The Dixon-Coles forecast IS used as a feature (stacking). That is legitimate:
those predictions come from the walk-forward cache, which was itself fitted
only on prior data.

SHOTS AS THE xG PROXY

We have no xG (fetch_understat.py was written but never run). We do have
shots and shots on target, which is most of what xG encodes at the team-form
level. Rolling shot rates stand in for it here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

PREDS = ROOT / "data" / "processed" / "backtest_preds.parquet"
RAW = ROOT / "data" / "raw" / "football_data_raw.parquet"

# Refit cadence for the walk-forward. 90 days keeps ~45 refits over 11 years.
REFIT_DAYS = 90
# Nothing is predicted until the booster has this much history to learn from.
MIN_TRAIN = 3000
# Rolling windows, in matches, for team form.
WINDOWS = (5, 10)

MAJORS = {"E0": "EPL", "SP1": "La Liga", "I1": "Serie A",
          "D1": "Bundesliga", "F1": "Ligue 1"}

# Columns that must NEVER become features, enforced by PREFIX rather than by
# exact name. The exact-name version of this let two leaks through on the
# first build, both of which would have manufactured a fake edge:
#
#   * The raw/DC merge suffixed the odds columns to PSH_x / PSH_y, which an
#     exact-match ban does not catch. A booster given closing odds learns the
#     market, scores beautifully against it, and proves nothing.
#   * HS / AS / HST / AST are THIS match's shots — known only after full time.
#     Rolling shot form (h_sf_5, d_stf_10, ...) is the causal version and is
#     built separately; the raw per-match columns are pure future information.
#
# Anything genuinely useful should be derived into an explicitly-named causal
# feature, so a broad prefix ban costs nothing and closes the whole class.
BANNED_PREFIXES = (
    "PSH", "PSD", "PSA",          # Pinnacle closing
    "Avg", "Max", "B365", "PC>", "PC<",   # other books, over/under
    "FTHG", "FTAG", "FTR",        # the answer
    "HS", "AS", "HST", "AST",     # this match's shots (post-hoc)
    # This match's xG. Understat publishes it AFTER the match, so it is
    # future information in exactly the way HS/AS were — and far more
    # dangerous, because xG predicts the result well enough that including it
    # would produce a spectacular, entirely fake improvement. Only the shifted
    # rolling forms (h_xgf_5, d_xga_10, ...) are causal.
    "xg_h", "xg_a",
    "y",
)


def is_banned(col: str) -> bool:
    return any(col.startswith(p) for p in BANNED_PREFIXES)


def rps(p: np.ndarray, idx: np.ndarray) -> float:
    """Ranked Probability Score for 1X2, normalised by (r-1)."""
    p = np.asarray(p, float)
    p = p / p.sum(axis=1, keepdims=True)
    cum_p = np.cumsum(p, axis=1)[:, :-1]
    obs = np.zeros_like(p)
    obs[np.arange(len(p)), idx] = 1
    cum_o = np.cumsum(obs, axis=1)[:, :-1]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1)) / (p.shape[1] - 1))


def team_form(df: pd.DataFrame) -> pd.DataFrame:
    """Per-team rolling form, strictly causal.

    Explodes each match into two team-rows, sorts by team and date, computes
    rolling means, then SHIFTS so a row never contains its own match. Only
    after that does it pivot back — doing the shift after the pivot is how
    this kind of feature usually leaks.
    """
    has_xg = "xg_h" in df.columns and "xg_a" in df.columns
    home = pd.DataFrame({
        "Date": df["Date"], "mid": df.index, "team": df["HomeTeam"],
        "opp": df["AwayTeam"], "is_home": 1,
        "gf": df["FTHG"], "ga": df["FTAG"],
        "sf": df.get("HS"), "sa": df.get("AS"),
        "stf": df.get("HST"), "sta": df.get("AST"),
        "xgf": df["xg_h"] if has_xg else np.nan,
        "xga": df["xg_a"] if has_xg else np.nan,
    })
    away = pd.DataFrame({
        "Date": df["Date"], "mid": df.index, "team": df["AwayTeam"],
        "opp": df["HomeTeam"], "is_home": 0,
        "gf": df["FTAG"], "ga": df["FTHG"],
        "sf": df.get("AS"), "sa": df.get("HS"),
        "stf": df.get("AST"), "sta": df.get("HST"),
        "xgf": df["xg_a"] if has_xg else np.nan,
        "xga": df["xg_h"] if has_xg else np.nan,
    })
    long = pd.concat([home, away], ignore_index=True)
    long["pts"] = np.where(long.gf > long.ga, 3, np.where(long.gf == long.ga, 1, 0))
    long = long.sort_values(["team", "Date"]).reset_index(drop=True)

    cols = ["pts", "gf", "ga", "sf", "sa", "stf", "sta"]
    if has_xg:
        cols += ["xgf", "xga"]
    g = long.groupby("team", sort=False)
    out = {"mid": long["mid"], "team": long["team"], "is_home": long["is_home"]}
    for w in WINDOWS:
        for col in cols:
            # shift(1) BEFORE rolling: the window ends at the previous match.
            out[f"{col}_{w}"] = (g[col].shift(1)
                                 .rolling(w, min_periods=max(2, w // 2)).mean()
                                 .reset_index(level=0, drop=True))
    out["rest"] = (g["Date"].diff().dt.days).clip(0, 30)
    out["played"] = g.cumcount()
    return pd.DataFrame(out)


def elo(df: pd.DataFrame, k: float = 20.0, ha: float = 60.0) -> pd.DataFrame:
    """Plain Elo, walked forward one match at a time. Causal by construction."""
    r: dict[str, float] = {}
    eh, ea = np.empty(len(df)), np.empty(len(df))
    for i, (h, a, res) in enumerate(zip(df["HomeTeam"], df["AwayTeam"], df["FTR"])):
        rh, ra = r.get(h, 1500.0), r.get(a, 1500.0)
        eh[i], ea[i] = rh, ra
        exp_h = 1.0 / (1.0 + 10 ** (-((rh + ha) - ra) / 400.0))
        sc = 1.0 if res == "H" else 0.5 if res == "D" else 0.0
        r[h] = rh + k * (sc - exp_h)
        r[a] = ra + k * ((1.0 - sc) - (1.0 - exp_h))
    return pd.DataFrame({"elo_h": eh, "elo_a": ea, "elo_d": eh - ea},
                        index=df.index)


def build(divs: list[str] | None = None, with_xg: bool = False) -> pd.DataFrame:
    """Assemble the feature table: walk-forward DC output + causal form.

    `with_xg` attaches Understat xG so team_form can build rolling xG-for and
    xG-against. The raw per-match xg_h/xg_a are prefix-banned from features —
    they are published after the whistle, and a booster given them would post
    a spectacular fake improvement.
    """
    dc = pd.read_parquet(PREDS)
    dc["Date"] = pd.to_datetime(dc["Date"])
    raw = pd.read_parquet(RAW)
    raw["Date"] = pd.to_datetime(raw["Date"])
    # Take ONLY the shot columns from raw. The DC cache already carries the
    # Pinnacle odds, and pulling them from both sides made pandas suffix them
    # to PSH_x / PSH_y — which then slipped past an exact-name leakage ban and
    # broke the market benchmark, which was looking for the unsuffixed name.
    keep = ["Div", "Date", "HomeTeam", "AwayTeam", "HS", "AS", "HST", "AST"]
    raw = raw[[c for c in keep if c in raw.columns]]

    df = dc.merge(raw, on=["Div", "Date", "HomeTeam", "AwayTeam"], how="left")
    if divs:
        df = df[df["Div"].isin(divs)]
    df = df[df["FTR"].notna()].sort_values("Date").reset_index(drop=True)

    if with_xg:
        import xg_join
        xg = xg_join.join(verbose=False)
        if len(xg):
            df = df.merge(xg[["Div", "Date", "HomeTeam", "AwayTeam",
                              "xg_h", "xg_a"]],
                          on=["Div", "Date", "HomeTeam", "AwayTeam"], how="left")
            cov = df["xg_h"].notna().mean()
            print(f"  xG attached to {df['xg_h'].notna().sum():,} of "
                  f"{len(df):,} matches ({cov:.1%})")

    df = pd.concat([df, elo(df)], axis=1)

    # Join form back on the match id by INDEX, not by a merge key. The merge
    # version (left_index=True, right_on="h_mid") silently reindexed the frame
    # and left half the difference columns null — the second merge compounded
    # the damage from the first. Setting mid as the index and reindexing to
    # df.index cannot misalign.
    form = team_form(df)
    fh = (form[form.is_home == 1].drop(columns=["is_home", "team"])
          .set_index("mid").add_prefix("h_").reindex(df.index))
    fa = (form[form.is_home == 0].drop(columns=["is_home", "team"])
          .set_index("mid").add_prefix("a_").reindex(df.index))
    df = pd.concat([df, fh, fa], axis=1)

    # Differences carry most of the signal; give the booster them directly.
    for w in WINDOWS:
        for col in ("pts", "gf", "ga", "sf", "sa", "stf", "sta", "xgf", "xga"):
            hcol, acol = f"h_{col}_{w}", f"a_{col}_{w}"
            if hcol in df and acol in df:
                df[f"d_{col}_{w}"] = df[hcol] - df[acol]
    df["y"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Numeric columns that are safe to learn from.

    Deliberately allow-by-exclusion with a loud prefix ban rather than an
    explicit allow-list, so a newly added causal feature works without edits
    while anything odds- or outcome-shaped stays out by construction.
    """
    skip = {"Date", "Div", "HomeTeam", "AwayTeam", "country", "season",
            "league", "h_mid", "a_mid", "mid"}
    return [c for c in df.columns
            if c not in skip and not is_banned(c) and df[c].dtype.kind in "fiub"]


def walk_forward(df: pd.DataFrame, refit_days: int = REFIT_DAYS,
                 verbose: bool = True) -> pd.DataFrame:
    """Refit periodically, predict only strictly-future matches."""
    import lightgbm as lgb

    feats = feature_cols(df)
    if verbose:
        print(f"{len(feats)} features, {len(df):,} matches")
        excl = sorted(c for c in df.columns if is_banned(c))
        print(f"  excluded as leakage ({len(excl)}): {excl[:8]} ...")

    df = df.sort_values("Date").reset_index(drop=True)
    preds = np.full((len(df), 3), np.nan)
    start = df["Date"].iloc[MIN_TRAIN] if len(df) > MIN_TRAIN else None
    if start is None:
        return df
    cursor, n_fits = start, 0
    end = df["Date"].max()

    while cursor <= end:
        nxt = cursor + pd.Timedelta(days=refit_days)
        tr = df[df["Date"] < cursor]
        te_mask = (df["Date"] >= cursor) & (df["Date"] < nxt)
        if len(tr) < MIN_TRAIN or not te_mask.any():
            cursor = nxt
            continue
        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=400,
            learning_rate=0.03, num_leaves=31, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, verbose=-1, n_jobs=-1)
        m.fit(tr[feats], tr["y"])
        preds[te_mask.to_numpy()] = m.predict_proba(df.loc[te_mask, feats])
        n_fits += 1
        if verbose and n_fits % 10 == 0:
            print(f"  {cursor.date()}  {n_fits} fits, "
                  f"{int(np.isfinite(preds[:, 0]).sum()):,} predicted")
        cursor = nxt

    df["g_home"], df["g_draw"], df["g_away"] = preds[:, 0], preds[:, 1], preds[:, 2]
    if verbose:
        print(f"  {n_fits} refits, {int(df['g_home'].notna().sum()):,} predicted")
    return df


def evaluate(df: pd.DataFrame, weights=(0.0, 0.25, 0.5, 0.75, 1.0)) -> None:
    """Score DC, GBM and their blends, per league and pooled, against market."""
    d = df[df["g_home"].notna()].copy()
    P_dc = d[["m_home", "m_draw", "m_away"]].to_numpy(float)
    P_dc = P_dc / P_dc.sum(axis=1, keepdims=True)
    P_gb = d[["g_home", "g_draw", "g_away"]].to_numpy(float)
    y = d["y"].to_numpy(int)

    mk = d.dropna(subset=["PSH", "PSD", "PSA"])
    if len(mk) > 200:
        inv = np.c_[1 / mk.PSH, 1 / mk.PSD, 1 / mk.PSA]
        Q = inv / inv.sum(axis=1, keepdims=True)
        market = rps(Q, mk["y"].to_numpy(int))
    else:
        market = float("nan")

    print(f"\n{'=' * 62}\nPOOLED  (n={len(d):,})")
    print(f"  market (Pinnacle close)   {market:.4f}   on n={len(mk):,}")
    print(f"  {'blend w':<10}{'RPS':>10}   (w = weight on GBM)")
    best = None
    for w in weights:
        P = (1 - w) * P_dc + w * P_gb
        s = rps(P, y)
        flag = ""
        if best is None or s < best[1]:
            best = (w, s)
        print(f"  {w:<10.2f}{s:>10.4f}{flag}")
    print(f"  -> best weight {best[0]:.2f} at RPS {best[1]:.4f}")

    print(f"\n{'league':<12}{'n':>6}{'DC':>9}{'GBM':>9}{'50/50':>9}"
          f"{'best':>9}{'market':>9}")
    for div, name in MAJORS.items():
        s = d[d["Div"] == div]
        if len(s) < 300:
            continue
        i = s.index.to_numpy()
        pos = d.index.get_indexer(i)
        a, b = P_dc[pos], P_gb[pos]
        yy = s["y"].to_numpy(int)
        r_dc, r_gb = rps(a, yy), rps(b, yy)
        r_50 = rps(0.5 * a + 0.5 * b, yy)
        ws = np.linspace(0, 1, 21)
        r_best = min(rps((1 - w) * a + w * b, yy) for w in ws)
        m = s.dropna(subset=["PSH", "PSD", "PSA"])
        if len(m) > 100:
            inv = np.c_[1 / m.PSH, 1 / m.PSD, 1 / m.PSA]
            rm = rps(inv / inv.sum(axis=1, keepdims=True), m["y"].to_numpy(int))
        else:
            rm = float("nan")
        print(f"{name:<12}{len(s):>6}{r_dc:>9.4f}{r_gb:>9.4f}{r_50:>9.4f}"
              f"{r_best:>9.4f}{rm:>9.4f}")

    print(f"\nAthena published: EPL 0.1998 (mkt 0.1943) · "
          f"La Liga 0.1960 (mkt 0.1910)")


def evaluate_oos(df: pd.DataFrame, refit_days: int = REFIT_DAYS) -> None:
    """Score the blend with the weight chosen OUT OF SAMPLE.

    The per-league "best" column is optimistic: it picks the weight that
    minimises RPS on the very rows being scored. That is the blend-weight
    equivalent of fitting on the test set, and on differences this small
    (fourth decimal) it is the difference between a real result and a
    flattering one.

    Here the weight is refitted on everything strictly BEFORE each cursor and
    applied forward, exactly as it would run live. If the blend only helps
    when the weight is chosen with hindsight, it does not help.
    """
    d = df[df["g_home"].notna()].sort_values("Date").reset_index(drop=True)
    P_dc = d[["m_home", "m_draw", "m_away"]].to_numpy(float)
    P_dc = P_dc / P_dc.sum(axis=1, keepdims=True)
    P_gb = d[["g_home", "g_draw", "g_away"]].to_numpy(float)
    y = d["y"].to_numpy(int)
    grid = np.linspace(0, 1, 21)

    chosen = np.full(len(d), np.nan)
    cursor = d["Date"].min() + pd.Timedelta(days=365)
    while cursor <= d["Date"].max():
        nxt = cursor + pd.Timedelta(days=refit_days)
        tr = (d["Date"] < cursor).to_numpy()
        te = ((d["Date"] >= cursor) & (d["Date"] < nxt)).to_numpy()
        if tr.sum() > 500 and te.any():
            w_best = min(grid, key=lambda w: rps(
                (1 - w) * P_dc[tr] + w * P_gb[tr], y[tr]))
            chosen[te] = w_best
        cursor = nxt

    ok = np.isfinite(chosen)
    if ok.sum() < 500:
        print("\nnot enough rows for an out-of-sample blend test")
        return
    P_oos = ((1 - chosen[ok, None]) * P_dc[ok] + chosen[ok, None] * P_gb[ok])
    print(f"\n{'=' * 62}\nOUT-OF-SAMPLE BLEND  (n={int(ok.sum()):,})")
    print(f"  DC alone                 {rps(P_dc[ok], y[ok]):.4f}")
    print(f"  GBM alone                {rps(P_gb[ok], y[ok]):.4f}")
    print(f"  fixed 50/50 (her weight) {rps(0.5 * P_dc[ok] + 0.5 * P_gb[ok], y[ok]):.4f}")
    print(f"  weight chosen OOS        {rps(P_oos, y[ok]):.4f}")
    print(f"    weight used: mean {np.nanmean(chosen[ok]):.2f}, "
          f"range {np.nanmin(chosen[ok]):.2f}-{np.nanmax(chosen[ok]):.2f}")
    delta = rps(P_dc[ok], y[ok]) - rps(P_oos, y[ok])
    print(f"  -> blend {'GAINS' if delta > 0 else 'LOSES'} "
          f"{abs(delta):.4f} RPS vs Dixon-Coles alone")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--majors-only", action="store_true", default=True)
    ap.add_argument("--all-divs", action="store_true")
    ap.add_argument("--refit-days", type=int, default=REFIT_DAYS)
    ap.add_argument("--out", default="data/processed/gbm_preds.parquet")
    args = ap.parse_args()

    divs = None if args.all_divs else list(MAJORS)
    print(f"building features ({'all divisions' if divs is None else 'majors'})")
    df = build(divs)
    df = walk_forward(df, refit_days=args.refit_days)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"saved -> {out}")
    evaluate(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
