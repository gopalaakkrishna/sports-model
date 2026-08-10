"""Single entry point for match history.

Combines the two football-data.co.uk file families - the European divisions
(E0, D1, SP1, ...) and the year-round leagues (ARG, USA/MLS, BRA, ...) - into
one frame with a common schema, and derives the country groupings the model
fits over.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import model as M

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# Divisions that are cross-border by design and must not be treated as
# contamination by the country-consistency check.
BRIDGE_PREFIXES = ("LC:",)


def drop_cross_country_contamination(df: pd.DataFrame, min_share: float = 0.05,
                                     verbose: bool = True) -> pd.DataFrame:
    """Remove matches filed under the wrong country.

    football-data.co.uk seeded the 2026/27 SP1/SP2 files with Scottish fixtures:
    10 rows putting 20 Scottish clubs into Spain's rating pool. Small in volume
    but harmful out of proportion — they carry the most recent dates, so the
    time decay weights them at the maximum, and they make Scottish names
    resolvable inside the Spanish team list.

    Rule: each team's country is the one holding the large majority of its
    matches. A row is dropped if either side appears in a country accounting for
    under `min_share` of that team's games. Legitimate cross-border clubs
    (Monaco in France, Cardiff in England) are unaffected because they play
    ~100% of their matches in that one country.
    """
    # Bridge competitions are cross-border BY DESIGN — Leagues Cup is MLS vs
    # Liga MX. They must be exempt from this check, which exists to catch teams
    # filed under the wrong country. Without the exemption this rule deletes
    # exactly the data that connects the leagues.
    is_bridge = df["Div"].astype(str).str.startswith(tuple(BRIDGE_PREFIXES))
    bridge_rows = df[is_bridge]
    d = df[~is_bridge].copy()

    d["_country"] = d["Div"].map(
        {lg: c for c, lgs in M.COUNTRY_GROUPS.items() for lg in lgs}
    )
    # Year-round leagues use "CODE:League" and are already country-scoped.
    mask_new = d["Div"].astype(str).str.contains(":", na=False)
    d.loc[mask_new, "_country"] = (
        d.loc[mask_new, "Div"].astype(str).str.split(":").str[0]
    )

    long = pd.concat([
        d[["_country", "HomeTeam"]].rename(columns={"HomeTeam": "team"}),
        d[["_country", "AwayTeam"]].rename(columns={"AwayTeam": "team"}),
    ]).dropna()
    counts = long.groupby(["team", "_country"]).size().rename("n").reset_index()
    totals = counts.groupby("team")["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    bad_pairs = set(
        map(tuple, counts.loc[counts["share"] < min_share, ["team", "_country"]].values)
    )
    if not bad_pairs:
        return df

    home_bad = [ (t, c) in bad_pairs for t, c in zip(d["HomeTeam"], d["_country"]) ]
    away_bad = [ (t, c) in bad_pairs for t, c in zip(d["AwayTeam"], d["_country"]) ]
    drop = pd.Series(home_bad, index=d.index) | pd.Series(away_bad, index=d.index)
    if verbose and drop.any():
        ex = d.loc[drop, ["Div", "Date", "HomeTeam", "AwayTeam"]].head(3)
        print(f"  data hygiene: dropped {int(drop.sum())} cross-country rows, e.g.")
        for _, r in ex.iterrows():
            print(f"    {r['Div']} {str(r['Date'])[:10]} {r['HomeTeam']} v {r['AwayTeam']}")
    kept = d.loc[~drop.values].drop(columns=["_country"])
    return pd.concat([kept, bridge_rows], ignore_index=True).sort_values(
        "Date").reset_index(drop=True)


def load_history(include_new: bool = True, clean: bool = True,
                 include_bridges: bool = True) -> pd.DataFrame:
    frames = []
    euro = RAW / "football_data_raw.parquet"
    if euro.exists():
        frames.append(pd.read_parquet(euro))
    if include_new:
        new = RAW / "new_leagues_raw.parquet"
        if new.exists():
            frames.append(pd.read_parquet(new))
    if include_bridges:
        # Cross-competition results that connect otherwise separate leagues.
        # Without these MLS and Liga MX have no shared opponents at all and
        # their ratings cannot be compared.
        import leagues_cup_data as LC
        frames.append(LC.load())
    if not frames:
        raise FileNotFoundError("no history parquet found - run fetch_data.py first")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("Date").reset_index(drop=True)
    if clean:
        df = drop_cross_country_contamination(df)
    return df


def country_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """European groups plus one group per year-round country present in df."""
    groups = {c: list(v) for c, v in M.COUNTRY_GROUPS.items()}
    present = set(df["Div"].dropna().unique())
    for div in sorted(present):
        if ":" not in div:
            continue
        code = div.split(":", 1)[0]
        name = M.NEW_LEAGUE_COUNTRIES.get(code, code)
        groups.setdefault(name, [])
        if div not in groups[name]:
            groups[name].append(div)
    # A combined North America group. Leagues Cup results link MLS and Liga MX,
    # so fitting the three together puts both leagues on ONE rating scale and
    # the league-strength offset is estimated rather than assumed. USA and
    # Mexico are also kept separately for domestic-only work.
    na = [d for d in ("USA:MLS", "MEX:Liga MX", "LC:Leagues Cup") if d in present]
    if "LC:Leagues Cup" in present and len(na) >= 3:
        groups["NorthAmerica"] = na

    # Drop groups with no data in this frame.
    return {c: d for c, d in groups.items() if present & set(d)}
