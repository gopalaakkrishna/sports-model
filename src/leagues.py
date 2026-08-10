"""One place that knows what a competition is called.

Division codes, Kalshi series tickers and ledger league strings all name the
same competitions differently — "SC0", "KXSCOTTISHPREMGAME" and "Scottish
Premiership" are one league. Anything that groups or reports by league needs a
single answer, or the same competition splits into three rows.

    pretty("SC0")                 -> "Scotland Premiership"
    pretty("KXLIGAMXGAME")        -> "Mexico Liga MX"
    sport_of("KXHUNDREDMATCH")    -> "cricket"
"""

from __future__ import annotations

# football-data.co.uk main divisions
DIVISIONS = {
    "E0": "England Premier League", "E1": "England Championship",
    "E2": "England League One", "E3": "England League Two",
    "EC": "England Conference",
    "SC0": "Scotland Premiership", "SC1": "Scotland Championship",
    "SC2": "Scotland League One", "SC3": "Scotland League Two",
    "D1": "Germany Bundesliga", "D2": "Germany 2. Bundesliga",
    "I1": "Italy Serie A", "I2": "Italy Serie B",
    "SP1": "Spain La Liga", "SP2": "Spain Segunda",
    "F1": "France Ligue 1", "F2": "France Ligue 2",
    "N1": "Netherlands Eredivisie", "B1": "Belgium Pro League",
    "P1": "Portugal Primeira Liga", "T1": "Turkey Super Lig",
    "G1": "Greece Super League",
    # year-round leagues (the /new files, keyed by country)
    "ARG": "Argentina Liga Profesional", "BRA": "Brazil Serie A",
    "MEX": "Mexico Liga MX", "USA": "USA MLS", "JPN": "Japan J1 League",
    "CHN": "China Super League", "NOR": "Norway Eliteserien",
    "SWE": "Sweden Allsvenskan", "FIN": "Finland Veikkausliiga",
    "IRL": "Ireland Premier Division", "DNK": "Denmark Superliga",
    "POL": "Poland Ekstraklasa", "ROU": "Romania Liga I",
    "RUS": "Russia Premier League", "AUT": "Austria Bundesliga",
    "SWZ": "Switzerland Super League",
    "LC": "Leagues Cup",
}

# Kalshi series ticker -> (sport, league)
SERIES = {
    "KXMLSGAME": ("soccer", "USA MLS"),
    "KXLIGAMXGAME": ("soccer", "Mexico Liga MX"),
    "KXLALIGAGAME": ("soccer", "Spain La Liga"),
    "KXBUNDESLIGA2GAME": ("soccer", "Germany 2. Bundesliga"),
    "KXALLSVENSKANGAME": ("soccer", "Sweden Allsvenskan"),
    "KXELITESERIENGAME": ("soccer", "Norway Eliteserien"),
    "KXSCOTTISHPREMGAME": ("soccer", "Scotland Premiership"),
    "KXJLEAGUEGAME": ("soccer", "Japan J1 League"),
    "KXLEAGUESCUPGAME": ("soccer", "Leagues Cup"),
    "KXMLBGAME": ("baseball", "MLB"),
    "KXWNBAGAME": ("basketball", "WNBA"),
    "KXNFLGAME": ("nfl", "NFL"),
    "KXHUNDREDMATCH": ("cricket", "The Hundred"),
}

# Free-text league strings that already appear in the ledger, normalised onto
# the same names so history does not fragment into near-duplicates.
ALIASES = {
    "mls": "USA MLS", "liga mx": "Mexico Liga MX",
    "liga profesional": "Argentina Liga Profesional",
    "bundesliga 2": "Germany 2. Bundesliga",
    "2. bundesliga": "Germany 2. Bundesliga",
    "eliteserien": "Norway Eliteserien",
    "allsvenskan": "Sweden Allsvenskan",
    "la liga": "Spain La Liga",
    "scottish premiership": "Scotland Premiership",
    "j1 league": "Japan J1 League", "j league": "Japan J1 League",
    "leagues cup": "Leagues Cup", "the hundred": "The Hundred",
    "hundred": "The Hundred",
    "mlb": "MLB", "wnba": "WNBA", "nfl": "NFL",
}


def pretty(code: str | None) -> str:
    if not code:
        return "Unclassified"
    c = str(code).strip()
    if c in DIVISIONS:
        return DIVISIONS[c]
    if c in SERIES:
        return SERIES[c][1]
    low = c.lower()
    if low in ALIASES:
        return ALIASES[low]
    # A Leagues Cup fixture is stored with an "LC:" prefix on the division.
    if c.startswith("LC:"):
        return "Leagues Cup"
    return c


def sport_of(code: str | None) -> str | None:
    if not code:
        return None
    c = str(code).strip()
    if c in SERIES:
        return SERIES[c][0]
    return None
