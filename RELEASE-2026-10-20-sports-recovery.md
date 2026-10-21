# Sports full-refresh date and recovery repair

Scope: existing personal `tara-sports-refresh` Worker and sports-model pipeline.
No Tara trading settings, entry/exit rules, AutoTrade, credentials, notification
subscriptions, ledger history, or company Cloudflare resources are changed.

## Root cause

`mlb_season_wins.py` requested today's schedule through a fixed October5 date.
On October20 the inverted interval returned StatsAPI HTTP400. The full chain
correctly remained degraded, but recovery dispatched another expensive attempt
roughly hourly. Some full jobs also hit their existing30minute job timeout.
Live/light successes did not repair the historical full-refresh failure.

## Changes and checks

- Official MLB calendar, inclusive regular-season boundaries, no postseason
  simulation or reversed date requests. Legitimate offseason is a documented
  no-op; failed/invalid upstream evidence is still an error.
- Heavy recovery waits2h then6h from completion and is blocked after3 recent
  unsuccessful attempts in rolling24h. Old failures/cancellations count;
  restarts and light refreshes cannot erase the budget. The independent last
  success query avoids dropping older successful evidence from the first page.
- Health explicitly separates live-board freshness from full-refresh health,
  attempts, retry-after and blocked state. No failure-email filtering/muting.
- Existing five-minute Cloudflare tick, GitHub schedules, concurrency group,
  fixed workflow/repository allowlist and encrypted secret are preserved.
- Follow-up verification found `refresh_all.py` printed failed source steps
  without returning nonzero. It now propagates failure; the parent logs the
  seven public-source receipts even on success. Unknown/empty source selection
  is not a successful refresh. No errors are suppressed to get a green run.
- Full runs restore the existing historical-download cache before running ALL
  source fetchers. Current-season downloads and incremental updates still run;
  this avoids downloading decades of unchanged CSVs on each cold runner. The
  existing30minute full-job limit is NOT raised.
- Python regression coverage: preseason/offseason, exact boundaries, extended
  season calendar, bad/ambiguous/upstream-failed calendars, reversed interval,
  regular-only schedule, and no offseason model/log/price/archive effects.
- Scheduler regression coverage: backoff from completion, cancelled attempts,
  budget exhaustion, legacy retry storm, history truncation, old success lookup,
  continued light refreshes, and existing authentication/no-public-dispatch guards.
- Additional regressions cover each failed source's nonzero result, all-seven
  success, parent receipt visibility, honest degraded full-pipeline publication,
  read-only inventory, and cache restore without skipping the full chain.

## Verification and rollback

Before shipping: complete Python/Node suites; exact diff excludes canonical
records and unrelated app files. Deploy only account
`5bd757dd1285beb06bf95ef9e2962c6b`, existing Worker, preserving encrypted
`GITHUB_TOKEN` and cron. Verify delivered health version and exhausted-budget
state before explicitly starting one repair-verification full workflow.

Accept only a real successful full job with passing regenerated-report check,
saved raw-data cache and a new canonical/public board whose `refresh.source`
is `gha-full`, `status` is `ok` and `failed_steps` is empty. A later light board
must not be substituted as evidence of full success. Record exact run/commit,
timestamps, public hash and personal Worker deployment in local release evidence.

Rollback trigger: changed personal target/secrets/cron, regression failure,
publication failure or new incorrect full-recovery behavior. Restore the prior
Worker deployment and revert only the reviewed repair code; never rewrite
historical ledger records, suppress failures, or enable real trading.
