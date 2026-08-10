"""Resolve team names between the fixtures feed and the historical files.

football-data.co.uk uses different names in fixtures.csv than in the season
files ("Dundee Utd" vs "Dundee United", "Raith" vs "Raith Rvs"). Unresolved
names silently drop fixtures, so resolution is explicit and auditable.

Order of resolution:

  1. exact match
  2. normalised match (case, punctuation and spacing removed)
  3. curated alias table
  4. fuzzy match, only if clearly unambiguous - and always logged

Fuzzy matching is deliberately last and deliberately noisy. "Dundee Utd" is 0.80
similar to "Dundee", a completely different club, so anything the fuzzy step
resolves should be reviewed and promoted into ALIASES.
"""

from __future__ import annotations

import difflib
import re

# Curated fixtures-feed name -> historical-file name.
# Add to this whenever the audit reports an unresolved or fuzzy-matched name.
ALIASES = {
    # Scotland
    "Airdrieonians": "Airdrie Utd",
    "Annan": "Annan Athletic",
    "Dundee FC": "Dundee",
    "Dundee Utd": "Dundee United",
    "Elgin City": "Elgin",
    "Forfar Athletic": "Forfar",
    "Inverness": "Inverness C",
    "Partick Thistle": "Partick",
    "Queen of South": "Queen of Sth",
    "Raith": "Raith Rvs",
    "Stenhousemuir": "Stenhousemuir",
    "East Kilbride": "East Kilbride",
    "Spartans": "Spartans",
    "Kelty Hearts": "Kelty Hearts",
    "Edinburgh City": "Edinburgh City",
    "Cove Rangers": "Cove Rangers",

    # MLS — everyday short names differ a lot from the source file's full names.
    "LA Galaxy": "Los Angeles Galaxy",
    "LA Galaxy II": "Los Angeles Galaxy",
    "LAFC": "Los Angeles FC",
    # "Los Angeles" is deliberately absent: it could be either LA club.
    "NYCFC": "New York City",
    "New York City FC": "New York City",
    "NYRB": "New York Red Bulls",
    "RBNY": "New York Red Bulls",
    "Red Bulls": "New York Red Bulls",
    "SKC": "Sporting Kansas City",
    "Sporting KC": "Sporting Kansas City",
    "Kansas City": "Sporting Kansas City",
    "St Louis City": "St. Louis City",
    "St. Louis": "St. Louis City",
    "Atlanta United": "Atlanta Utd",
    "Atlanta": "Atlanta Utd",
    "Montreal": "CF Montreal",
    "Montreal Impact": "CF Montreal",
    "D.C. United": "DC United",
    "DC Utd": "DC United",
    "Vancouver": "Vancouver Whitecaps",
    "Seattle": "Seattle Sounders",
    "Portland": "Portland Timbers",
    "Columbus": "Columbus Crew",
    "Philadelphia": "Philadelphia Union",
    "Cincinnati": "FC Cincinnati",
    "Dallas": "FC Dallas",
    "Minnesota": "Minnesota United",
    "Charlotte FC": "Charlotte",
    "New England": "New England Revolution",
    "Colorado": "Colorado Rapids",
    "Houston": "Houston Dynamo",
    "Orlando": "Orlando City",
    "San Jose": "San Jose Earthquakes",
    "Nashville": "Nashville SC",
    "Miami": "Inter Miami",
    "Toronto": "Toronto FC",
    "Salt Lake": "Real Salt Lake",
    "Chicago": "Chicago Fire",
    "Austin": "Austin FC",
    "San Diego": "San Diego FC",
    # Kalshi's abbreviated MLS forms
    "Los Angeles G": "Los Angeles Galaxy",
    "Los Angeles F": "Los Angeles FC",
    "New York RB": "New York Red Bulls",
    "New York C": "New York City",
    "Saint Louis": "St. Louis City",
    "Saint Paul": "Minnesota United",

    # Spain — Kalshi long forms
    "Deportivo De La Coruna": "La Coruna",
    "Celta Vigo": "Celta",
    "Rayo Vallecano": "Vallecano",
    "Real Sociedad": "Sociedad",
    "Athletic Bilbao": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid",
    # "Atletico" alone is NOT mapped — Ath Madrid and Ath Bilbao both exist.

    # Germany
    "Nuremberg": "Nurnberg",
    "Kiel": "Holstein Kiel",
    "Cottbus": "Energie Cottbus",
    "Fuerth": "Greuther Furth",
    "Duesseldorf": "Fortuna Dusseldorf",

    # Liga MX
    "America": "Club America",
    "Club América": "Club America",
    "América": "Club America",
    "Tijuana": "Club Tijuana",
    "Xolos": "Club Tijuana",
    "Leon": "Club Leon",
    "León": "Club Leon",
    "Chivas": "Guadalajara Chivas",
    "Guadalajara": "Guadalajara Chivas",
    "Tigres": "Tigres UANL",
    "Pumas": "UNAM Pumas",
    "Pumas UNAM": "UNAM Pumas",
    "San Luis": "Atl. San Luis",
    "Atletico San Luis": "Atl. San Luis",
    "Mazatlan": "Mazatlan FC",
    "FC Juarez": "Juarez",

    # Sweden — Allsvenskan
    "BK Hacken": "Hacken",
    "Malmo": "Malmo FF",
    "Vasteraas": "Vasteras SK",
    "IFK Goteborg": "Goteborg",
    "IFK Norrkoping": "Norrkoping",
    "IF Elfsborg": "Elfsborg",

    # Scotland
    "Heart of Midlothian": "Hearts",
    "Saint Mirren": "St Mirren",
    "Saint Johnstone": "St Johnstone",

    # Norway — Eliteserien
    "Bodoe/Glimt": "Bodo/Glimt",
    "Lillestroem": "Lillestrom",
    "Sarpsborg": "Sarpsborg 08",
    "Valerenga": "Valerenga",
    "Tromsoe": "Tromso",

    # Japan — J1. Kalshi uses short forms; the history uses full club names.
    "Avispa": "Avispa Fukuoka",
    "Cerezo": "Cerezo Osaka",
    "Frontale": "Kawasaki Frontale",
    "Gamba": "Gamba Osaka",
    "Hiroshima": "Sanfrecce Hiroshima",
    "Kashima": "Kashima Antlers",
    "Kashiwa": "Kashiwa Reysol",
    "Kobe": "Vissel Kobe",
    "Kyoto Sanga": "Kyoto",
    "Machida Z": "Machida",
    "Marinos": "Yokohama F. Marinos",
    "Nagoya": "Nagoya Grampus",
    "Shimizu": "Shimizu S-Pulse",
    "Urawa": "Urawa Reds",
    "Fagiano O": "Okayama",
    "Tokyo": "FC Tokyo",
    # Deliberately NOT mapped — different clubs from anything in the J1 history,
    # or ambiguous against a club that IS present:
    #   "Tokyo V"      Tokyo Verdy, not FC Tokyo
    #   "United Chiba" JEF United Chiba
    #   "V-Varen"      V-Varen Nagasaki
    #   "Mito H"       Mito HollyHock
    # Leaving them unresolved drops the fixture, which is correct. Mapping any
    # of them to a near-miss would silently corrupt that club's predictions.
}

# Names that must never be fuzzy-matched, because a close string is a different
# club. Resolution for these has to come from ALIASES or an exact match.
FUZZY_BLOCKLIST = {
    "Dundee", "Dundee FC", "Dundee Utd", "Dundee United",
    "Man City", "Man United",
    "Sheffield United", "Sheffield Weds",
    "Nott'm Forest", "Notts County",
    "Bristol City", "Bristol Rvs",
    "Inter", "Milan",
    # Two LA clubs and two NY clubs — never let fuzzy pick between them.
    "Los Angeles", "Los Angeles FC", "Los Angeles Galaxy",
    "New York", "New York City", "New York Red Bulls",
    # Tokyo Verdy is not FC Tokyo; the two are ~0.8 similar as strings.
    "Tokyo V", "Tokyo Verdy", "FC Tokyo",
    # Likewise these Japanese clubs sit close to J1 sides they are not.
    "United Chiba", "V-Varen", "Mito H",
    # "Atletico" is ambiguous between Ath Madrid and Ath Bilbao.
    "Atletico",
}

_norm_re = re.compile(r"[^a-z0-9]")


def normalise(name: str) -> str:
    return _norm_re.sub("", str(name).lower())


class TeamResolver:
    """Maps feed names onto the historical names for one pool of teams."""

    def __init__(self, known: list[str], fuzzy_cutoff: float = 0.88):
        self.known = list(known)
        self.by_norm = {}
        for k in self.known:
            self.by_norm.setdefault(normalise(k), k)
        self.fuzzy_cutoff = fuzzy_cutoff
        self.fuzzy_log: list[tuple[str, str, float]] = []
        self.unresolved: set[str] = set()

    def resolve(self, name: str) -> str | None:
        if name in self.by_norm.values() and name in self.known:
            return name
        n = normalise(name)
        if n in self.by_norm:
            return self.by_norm[n]

        alias = ALIASES.get(name)
        if alias:
            if alias in self.known:
                return alias
            an = normalise(alias)
            if an in self.by_norm:
                return self.by_norm[an]

        if name in FUZZY_BLOCKLIST:
            self.unresolved.add(name)
            return None

        # Fuzzy, and only when the best candidate is clearly ahead of the next.
        matches = difflib.get_close_matches(n, list(self.by_norm), n=2, cutoff=self.fuzzy_cutoff)
        if matches:
            best = matches[0]
            score = difflib.SequenceMatcher(None, n, best).ratio()
            if len(matches) > 1:
                second = difflib.SequenceMatcher(None, n, matches[1]).ratio()
                if score - second < 0.05:
                    self.unresolved.add(name)
                    return None
            resolved = self.by_norm[best]
            self.fuzzy_log.append((name, resolved, score))
            return resolved

        self.unresolved.add(name)
        return None

    def report(self) -> None:
        if self.fuzzy_log:
            print("  fuzzy-matched names (review and add to ALIASES):")
            for src, dst, sc in self.fuzzy_log:
                print(f"    {src!r} -> {dst!r}  (similarity {sc:.2f})")
        if self.unresolved:
            print("  UNRESOLVED names (these fixtures are dropped):")
            for u in sorted(self.unresolved):
                near = difflib.get_close_matches(normalise(u), list(self.by_norm), n=3, cutoff=0.6)
                cands = [self.by_norm[m] for m in near]
                print(f"    {u!r}  nearest: {cands}")
