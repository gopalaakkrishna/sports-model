"""Loud failure for the one bug this pipeline keeps hitting.

Three times now a Kalshi response has arrived FULL of markets and been parsed
into nothing, because a title format, a field name or a ticker shape changed
underneath us:

  * kalshi_edge._event_titles() and settle._kalshi_soccer_results() split a
    per-market title on " vs ". Kalshi moved market titles to the per-outcome
    form ("Cincinnati wins", "Tie is the result"), so the split matched zero
    of 82 markets and both functions returned {}.
  * mlb_predict.kalshi_mlb_markets() had the same split. It returned {} for
    every game, so MLB picks published with NO bid/ask for about four weeks.

Every one failed the same way: the parse returned empty, the caller read that
as "no games today", and the pipeline carried on publishing a plausible board.
EV went UNCOMPUTABLE (NaN) rather than wrong, so no threshold tripped, nothing
raised, and a NaN column reads as "no market" rather than "the parser is dead".

THE ASYMMETRY THIS ENCODES

An empty UPSTREAM is normal and must stay silent: no fixtures on a Tuesday, a
season over, a league dark for the summer. A non-empty upstream that parses to
NOTHING is always a bug. That distinction is cheap to test and is precisely
the one no existing check was making.

WHERE TO PUT IT

Guard the PARSE step only, the step that turns a response into rows. Never
guard downstream of the business filters: date windows, league scope, spread
and depth screens and conviction floors are all entitled to return zero, and
wrapping those would produce false alarms on quiet days.

WHY RAISING IS SAFE HERE

auto_update.run() treats a non-zero exit as non-fatal by design (see its
docstring: "One dead API must not stop the rest of the chain"). It logs
FAILED plus the last three lines of stderr and carries on, so raising kills
exactly one predictor, surfaces the traceback in the run log, and still lets
finished games settle and the board republish. The missing sport then also
shows up as a stale source in the published payload, because export_tara ages
each report by the date in its filename.
"""

from __future__ import annotations


class ParseYieldedNothing(RuntimeError):
    """A non-empty upstream response parsed into zero usable rows."""


def _n(x) -> int:
    return x if isinstance(x, int) else len(x)


def parsed_or_die(raw, parsed, *, what: str, series: str | None = None,
                  samples=None, min_ratio: float | None = None,
                  n_samples: int = 3):
    """Assert that a non-empty upstream produced usable rows.

    raw / parsed may be ints or anything with a length.

    Raises ParseYieldedNothing when the upstream carried rows and the parse
    produced none, or, if min_ratio is given, when it produced a smaller
    fraction than that. min_ratio is opt-in per call site: a partial drop is
    normal in some places (fixtures we deliberately do not price) and a red
    flag in others, and only the call site knows which.

    `samples` should be a few RAW items, ideally the exact strings the parse
    tried to match. They are put in the exception message so whoever reads
    the failure sees the new upstream format immediately, instead of having
    to re-query the API to find out what changed.
    """
    raw_n, parsed_n = _n(raw), _n(parsed)
    if raw_n == 0:
        return parsed          # empty upstream is legitimate; stay silent
    if parsed_n and (min_ratio is None or parsed_n >= raw_n * min_ratio):
        return parsed

    where = f"{what}" + (f" [{series}]" if series else "")
    if parsed_n == 0:
        why = (f"upstream returned {raw_n} row(s) and NONE parsed. The "
               f"upstream format almost certainly changed.")
    else:
        why = (f"only {parsed_n} of {raw_n} row(s) parsed "
               f"({parsed_n / raw_n:.0%}, below the {min_ratio:.0%} floor).")

    msg = [f"{where}: {why}"]
    if samples:
        shown = [repr(s) for s in list(samples)[:n_samples]]
        msg.append("  raw sample: " + "; ".join(shown))
    msg.append("  This is the silent-empty guard in src/guards.py. It fires "
               "only when the response was NON-empty, so it is not a quiet "
               "day. Compare the sample above against what the parser "
               "expects.")
    raise ParseYieldedNothing("\n".join(msg))
