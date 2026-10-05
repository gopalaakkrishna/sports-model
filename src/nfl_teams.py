"""Shared, exact NFL identities for pricing and settlement (no fuzzy cities)."""

KALSHI_TO_NFL = {
    "Arizona": "ARI", "Atlanta": "ATL", "Baltimore": "BAL", "Buffalo": "BUF",
    "Carolina": "CAR", "Chicago": "CHI", "Cincinnati": "CIN", "Cleveland": "CLE",
    "Dallas": "DAL", "Denver": "DEN", "Detroit": "DET", "Green Bay": "GB",
    "Houston": "HOU", "Indianapolis": "IND", "Jacksonville": "JAX",
    "Kansas City": "KC", "Las Vegas": "LV", "Los Angeles C": "LAC",
    "Los Angeles R": "LA", "Miami": "MIA", "Minnesota": "MIN",
    "New England": "NE", "New Orleans": "NO", "New York G": "NYG",
    "New York J": "NYJ", "Philadelphia": "PHI", "Pittsburgh": "PIT",
    "San Francisco": "SF", "Seattle": "SEA", "Tampa Bay": "TB",
    "Tennessee": "TEN", "Washington": "WAS",
}


def nfl_abbr(value: str) -> str:
    return {"JAC": "JAX", "LAR": "LA"}.get(value, value)


def nfl_team(value: str) -> str | None:
    """Accept a known code or unambiguous Kalshi leg; reject generic LA/NY cities."""
    value = str(value).strip()
    legacy = {"NY Giants": "NYG", "NY Jets": "NYJ"}
    if value in legacy:
        return legacy[value]
    code = nfl_abbr(value.upper())
    if code in KALSHI_TO_NFL.values():
        return code
    # Older Kalshi legs used an explicit code plus mascot ('TEN Titans').
    # Match that code, never the shared city word in LA/NY team names.
    prefix = value.split(" ", 1)[0]
    if prefix.isupper() and nfl_abbr(prefix) in KALSHI_TO_NFL.values():
        return nfl_abbr(prefix)
    return {k.lower(): v for k, v in KALSHI_TO_NFL.items()}.get(value.lower())
