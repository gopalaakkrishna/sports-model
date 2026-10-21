# Independent Sports refresh clock

Deploy only to Gopalaakkrishna's personal Cloudflare account
`5bd757dd1285beb06bf95ef9e2962c6b`, subdomain
`gopalaakkrishna.workers.dev`. The company Cloudflare account is out of scope.
`wrangler.toml` pins the personal account to prevent accidental deployment.

The `tara-sports-refresh` Worker runs every five minutes. It checks the real
published forecast timestamp and existing GitHub jobs. If the board is at
least twelve minutes old, it dispatches the existing `light.yml` on `main`.
The daily full refresh is recovered when its last success is over26hours old.
Recovery backs off from completion for two hours after the first unsuccessful
attempt and six hours after the second. Cloudflare dispatches no further heavy
recovery while three unsuccessful full attempts remain in the rolling24hour
window. Scheduled/manual/cancelled runs also consume this conservative budget;
deploying the Worker or completing a light refresh does not reset it. GitHub's
once-daily fallback schedule remains unchanged. This is a freshness target, not a
guarantee of exchange/API or GitHub-runner availability. GitHub cron remains a
best-effort fallback, not the sole clock.

Light and full jobs share one concurrency group and check out the latest main
before changing records. The Worker skips queued/running jobs, never cancels
them, and reports jobs queued/stuck for45minutes as errors. A dispatch accepted
by GitHub is not reported as a completed refresh. Verify a new `generated`
timestamp and `refresh.status` in the public snapshot after a completed job.

The only secret is `GITHUB_TOKEN`: a fine-grained token limited to sports-model,
Actions read/write and the required Metadata read. No Contents, account,
trading, Supabase or Cloudflare API credentials are required by the Worker.
Store it as an encrypted Worker secret, never in source, logs or a browser
bundle. The initially approved token expires January13,2027; rotate it before
then. Expired/missing access returns a visible503 health failure and cron error.

`GET /health` is read-only and reports forecast age and pipeline health.
It separately reports `fullRefreshHealthy` and the full recovery status, last
success/attempt, recent attempts, retry limit and retry-after time. A fresh live
board does NOT mean the historical data/full pipeline is healthy. Exhausted or
incomplete recovery history stays visibly blocked; failure emails are not
muted and failed jobs are not reclassified as successes. A repair verification
may explicitly dispatch one full workflow; it remains serialized with live
refreshes and is recorded in the same history.
There is no public endpoint that starts jobs. Production writes happen only
inside the existing serialized Sports workflow, which remains paper record
bookkeeping and never places exchange orders.

Tests: `node --test tests/scheduler.test.mjs` and
`python -m unittest discover -s tests -v`. Both workflows run these tests.
MLB season-win measurement now reads the exact season's official regular-season
start/end calendar and skips simulation outside it. Missing/invalid/API-failed
calendar evidence still fails visibly; no forecasts/results/log rows are
fabricated to make an offseason refresh pass.
The full source wrapper also returns nonzero if any configured source fails;
its seven source receipts are retained in CI logs. Full workflows restore the
historical-download cache but still execute all source refreshes, with the
same30minute job timeout and the current season explicitly re-fetched.
Rollback: remove/disable this Worker's cron, revert the reviewed Sports commit,
and leave GitHub's original schedules available. Do not alter trading settings.
