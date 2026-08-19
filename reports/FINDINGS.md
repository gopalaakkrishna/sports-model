# Findings

Walk-forward backtest, 2015-08-01 to 2026-08-03, 68,304 matches with Pinnacle
closing odds across 22 divisions in 11 countries. Every prediction uses only
data available before that match.

## Headline

| | Log loss | Brier | Accuracy |
|---|---|---|---|
| Base rate (always predict H/D/A frequencies) | 1.07464 | — | — |
| **This model** | **1.01489** | 0.60798 | 49.08% |
| Pinnacle closing odds (de-vigged) | 0.99811 | 0.59654 | 50.52% |

The model is clearly better than nothing — it beats the base rate by 0.060 in
log loss. It loses to the closing line by 0.017.

**Optimal blend weight on the model, against the market: 0.00.** Fitted on
2015-2022 and validated on 2022-2026, out of sample. The market already contains
everything the model knows, and then some.

## The important table

Bucketing predictions by how far the model disagrees with the market:

| Disagreement | n | Model | Market | Gap |
|---|---|---|---|---|
| 0.0-2.8% | 17,076 | 0.9938 | 0.9923 | +0.0015 |
| 2.8-4.9% | 17,076 | 1.0138 | 1.0099 | +0.0039 |
| 4.9-8.0% | 17,076 | 1.0208 | 1.0065 | +0.0143 |
| 8.0-11.6% | 10,245 | 1.0270 | 0.9968 | +0.0302 |
| 11.6-43% | 6,831 | 1.0373 | **0.9641** | **+0.0731** |

When the model agrees with the market, both are accurate. **The further the
model strays from the market, the worse it gets — and the market gets better.**

This is the direct test of the "value" claim in the reel. "My model says 76%,
the market says 69%" is a ~7% disagreement. In that bucket the market is
substantially more accurate. The gap is the model's error, not an edge.

## Flat-stake simulation

Betting every outcome where `model probability x best available odds > 1`, at
the best price across all books — the most generous assumption available:

| Threshold | Bets | Profit | ROI |
|---|---|---|---|
| EV > 0% | 58,218 | −2,687 | −4.62% |
| EV > 2% | 50,973 | −2,579 | −5.06% |
| EV > 5% | 41,704 | −2,249 | −5.39% |
| EV > 10% | 29,833 | −2,115 | −7.09% |
| EV > 20% | 15,891 | −1,178 | **−7.42%** |

Losses get **worse** as the filter gets stricter. Being more selective about
"value" loses more money, because bigger disagreement means bigger model error.
Real prices would be worse still — best odds get limited and move.

## Calibration

The model is well calibrated, which is worth something on its own:

| Predicted | n | Actual | Error |
|---|---|---|---|
| 7.0% | 3,869 | 6.7% | −0.3% |
| 16.1% | 18,046 | 15.3% | −0.8% |
| 25.9% | 83,647 | 26.0% | +0.1% |
| 34.3% | 46,489 | 34.0% | −0.3% |
| 44.6% | 28,565 | 44.4% | −0.2% |
| 54.3% | 14,105 | 54.6% | +0.4% |
| 64.3% | 6,015 | 66.7% | +2.3% |
| 74.4% | 2,894 | 77.2% | +2.8% |
| 83.9% | 1,173 | 83.8% | −0.1% |

Errors are within ~1% across most of the range. Fitted temperature was 1.025 —
essentially no correction needed. When this model says 74%, it happens 77% of
the time. That is a genuinely useful forecast; it just is not a *better*
forecast than the closing line.

## By division

Closest to the market: German 2. Bundesliga (+0.0119), Greek Super League
(+0.0138), La Liga (+0.0139). Worst: Turkish Super Lig (+0.0246), Segunda
(+0.0205), Ligue 2 (+0.0205).

Two divisions produced a non-zero blend weight — D2 (0.06) and G1 (0.08) — on
~3,000 and ~2,200 matches respectively. Those are small samples and small
weights; treat as noise unless they survive on more data.

No division had the model beating the market. The lower-division "soft market"
hypothesis did not hold here: England League Two (+0.0165) was no better than
the Premier League (+0.0165).

## Hyperparameter tuning did not replicate

A 12-point grid over time-decay and regularisation was run on 2016-2022 only,
with 2022-2026 held out. Best on the tuning window: `xi=0.0025` (277-day
half-life), `reg=1.0`, log loss 1.01682 — a 0.0006 improvement on the default.
The optimum was interior, not at a grid edge (`xi=0.0040` was worse at all three
regularisation levels), so the grid was wide enough.

On the held-out window, evaluated on the identical 22,446 matches:

| Parameters | Log loss | vs market |
|---|---|---|
| Default `xi=0.0018, reg=2.0` | 1.01066 | +0.01772 |
| Tuned `xi=0.0025, reg=1.0` | 1.01087 | +0.01793 |

Tuning effect **+0.00021 (worse)**, paired bootstrap 95% CI
[−0.00042, +0.00086] — straddles zero, not significant.

The in-sample gain was noise. Hyperparameters are immaterial in this range, so
the defaults are left unchanged. This is exactly why the holdout was reserved
before tuning rather than after.

## The benchmark is robust to which line you use

football-data.co.uk **stopped publishing Pinnacle odds during 2025/26** —
coverage decays from Nov 2025 and is zero from Feb 2026. Market-average and
Bet365 odds have full coverage. So the Pinnacle benchmark cannot cover recent
matches, and will not exist for future fixtures.

Re-running against the market average (a weaker line — it blends sharp and soft
books, 6.5% overround vs Pinnacle's 4.1%), which should flatter the model:

| Benchmark | Period | n | Model | Market | Gap | Blend w |
|---|---|---|---|---|---|---|
| Pinnacle | to 2026-01-15 | 24,189 | 1.01399 | 0.99560 | +0.01840 | 0.00 |
| Market avg | to 2026-08-03 | 28,587 | 1.01498 | 0.99727 | +0.01770 | 0.00 |
| Market avg (matched) | to 2026-01-15 | 24,189 | 1.01399 | 0.99614 | +0.01785 | 0.00 |
| Market avg | **after 2026-01-15** | 3,421 | 1.01819 | 0.99976 | +0.01844 | 0.00 |

Blend weight is 0.00 against every benchmark, over every period, including the
most recent 3,421 matches against the softest available line. The conclusion
does not depend on the choice of benchmark.

## Year-round leagues (including MLS)

16 further countries were added — MLS, Argentina, Brazil, Mexico, Japan, China,
the Nordics and others — that run through the European off-season. 62,430
matches back to 2012. Backtest 2018-2026, 34,965 matches scored:

| | Log loss | Accuracy |
|---|---|---|
| Base rate | 1.07238 | — |
| Model | 1.02542 | 48.82% |
| Market closing | 1.00242 | 50.35% |

Gap **+0.02301**, blend weight **0.00**. Worse than Europe, not better.

The "these markets are softer, so the model should do better" hypothesis fails a
second time. Per division:

| Division | n | Model | Market | Gap | Blend w |
|---|---|---|---|---|---|
| ARG Copa de la Liga | 916 | 1.0765 | 1.0643 | +0.0122 | 0.16 |
| Russia Premier | 1,938 | 0.9976 | 0.9840 | +0.0136 | 0.14 |
| Japan J1 | 2,674 | 1.0522 | 1.0378 | +0.0144 | 0.05 |
| Mexico Liga MX | 2,840 | 1.0220 | 1.0062 | +0.0158 | 0.00 |
| **USA MLS** | **3,955** | **1.0380** | **1.0149** | **+0.0231** | **0.00** |
| Brazil Serie A | 3,235 | 1.0176 | 0.9948 | +0.0228 | 0.00 |
| Sweden Allsvenskan | 2,033 | 1.0115 | 0.9806 | +0.0309 | 0.00 |
| China Super League | 1,983 | 0.9718 | 0.9181 | +0.0537 | 0.00 |

MLS sits mid-table and gives the model zero weight. The only non-zero weights
are Argentina's Copa (0.16, n=916), Russia (0.14, n=1,938) and Japan (0.05).
Those are small samples and are the most likely candidates for noise — but they
are also the only places in 33 divisions worth re-testing on more data.

China is the standout failure: the market beats the model by 0.054, three times
the typical gap. Worth understanding before ever trusting the model there.

### MLS specifics

Home advantage in MLS is **+0.283 (x1.33 on expected home goals)**, clearly
above the European range of +0.17 to +0.25. Travel distance is real and the
model picks it up without being told.

Current top of the MLS ratings (as of 2026-08-02): Los Angeles FC, Vancouver
Whitecaps, Inter Miami. Miami has the league's best attack (+0.536) alongside a
poor defence (+0.198) — the model recovers the well-known shape of that team
from results alone.

## Structural limitation: disconnected leagues

The model cannot predict matches between two leagues that never meet in the
training data. MLS and Liga MX have **zero** matches between them across all
213,000 rows, so their rating graphs are disconnected and the relative strength
of the two leagues is not identifiable. Fitting them jointly does not help: with
no connecting edges the ridge pulls both to a common mean, silently imposing
"the leagues are equally strong".

Worked example, LAFC v Guadalajara (Leagues Cup, 2026-08-05, LAFC at home):

| | LAFC | Draw | Chivas |
|---|---|---|---|
| Model, equal-leagues assumption | 52.5% | 25.6% | 21.9% |
| Market (two independent quotes agree) | 40.3% | 28.1% | 31.6% |

A 12.2-point disagreement. Solving for the offset that reconciles it gives
delta = +0.380.

**But a second match contradicts that.** Checking Inter Miami v Atl. San Luis
the same day:

| Match | Market H/D/A | Model at delta=0 | Implied delta | Overround |
|---|---|---|---|---|
| LAFC v Chivas | 40/28/32 | 52/26/22 | **+0.380** | 10.8% |
| Inter Miami v Atl. San Luis | 68/17/15 | 69/16/15 | **+0.012** | 0.1% |

The two implied offsets are nowhere near each other, so a single league-strength
constant does **not** explain the disagreement. And the Miami read is the more
reliable of the two: its odds carry a 0.1% overround (essentially vig-free best
prices) against 10.8% for the LAFC quote.

Revised conclusion: **the MLS/Liga MX league offset is approximately zero**, and
the model's cross-league predictions are usable under the equal-leagues
assumption — validated by Miami matching the market to within one point. The
LAFC v Chivas gap is *team-level* rating error on one or both of those clubs,
which no global constant can repair.

This is worth recording as a methodological point: one match looked like clean
evidence for a large league offset, and it was wrong. A single observation with
soft prices reconciled to a confident-looking parameter estimate. The second
match, with better prices, killed it.

Connecting the graphs properly still needs Leagues Cup / CONCACAF Champions Cup
results (roughly 300-500 historical MLS v Liga MX matches).
`predict_cross_league.py` makes the offset an explicit input and sweeps it,
reporting whether a call is stable across the range.

## Venue liquidity: Kalshi vs Polymarket

Both expose public read APIs with no credentials. Tradeability screen: two-sided
quote, spread <= 0.06, liquidity >= $500, volume/depth >= $250.

| Venue / sport | Markets | Pass screen | Typical spread | Typical depth |
|---|---|---|---|---|
| Kalshi MLB game winner | 82 | **80** | 2-4c | $900-$4,900 |
| Kalshi soccer game winner | 244 | **125** | 1-5c | $860-$8,100 |
| Polymarket soccer | 35 | **3** | 1c on the 3 | $15k-$26k on the 3 |

**A field-name trap cost me the right answer twice.** Kalshi migrated to
dollar-denominated fields (`yes_bid_dollars`, `volume_fp`, `open_interest_fp`).
The legacy `yes_bid` / `volume` keys are still present in responses but always
empty, so reading them makes every market on the exchange look dead. On that
basis I first concluded "Kalshi soccer has no liquidity" — the opposite of the
truth. Anything screening a venue on volume alone should also check resting
orderbook depth: a newly opened market can show zero volume while already
holding thousands of contracts in the book.

Polymarket is the genuinely thin venue here, and thin in a specific way: 16 of
35 soccer markets sit at ~0.50 with $4 of liquidity and spreads of 0.91-0.99.
Those show enormous apparent edge and cannot be traded at all. Its three liquid
markets are, by contrast, tighter than any bookmaker.

Practical consequence: **Kalshi is the venue with usable depth**, across both
MLB and soccer, and its soccer markets are three-way (Home/Tie/Away) which maps
directly onto the 1X2 model already built.

## MLB

28,072 games, 2015-2026, from MLB StatsAPI (free, no key), including probable
starting pitchers for 27,219 of them. Model: Poisson on runs with team
offence/defence, a **starting-pitcher** term (the away starter suppresses home
runs), a **park** term lifting both sides, exponential time decay and ridge.
Analytic gradient verified to 7.5e-7 relative.

Sanity checks pass on their own: best-rated starters come out Sale, Skubal,
Wheeler, Yamamoto; worst run prevention is Colorado.

### Raw, the model is worse than doing nothing

Walk-forward, 17,025 games:

| | Log loss |
|---|---|
| Raw model | 0.69999 |
| Base rate (always the home-win rate) | 0.69117 |
| **Platt-calibrated** | **0.68228** |

Uncalibrated it *loses* to the base rate. Fitted shrinkage **a = 0.414** — the
raw Poisson model is over-confident by about 2.4x. Baseball has far more
game-to-game variance than a Poisson run model implies: it was claiming 73%
where reality was 63%, and 27% where reality was 43%. Calibrated, it beats the
base rate by 0.0089.

For scale, the soccer model beats its base rate by 0.060. Baseball is far harder.

### Park factors matter for totals, not for winners

Adding a venue term cut the Coors Field error on total runs from **+1.78 to
+0.33**, and the worst venue error across the league from 1.78 to 0.63. Win
probability barely moved — park effects lift both teams symmetrically and cancel
in the winner market. They matter for over/under, not for who wins.

### Against Kalshi, the model is strictly less informative

Kalshi MLB books are genuinely deep — tens of thousands of contracts per level,
1c spreads, two-way ask sums around 1.02 (tighter than Pinnacle's 1.041). Note
`liquidity_dollars` reports 0.0000 while the book is full, so liquidity has to
be read from the orderbook, not that field.

Comparing model to Kalshi across the slate:

| Market prob | n | Mean model | Model − market |
|---|---|---|---|
| 0.00-0.35 | 1 | 33.6% | **+7.2%** |
| 0.35-0.45 | 9 | 45.1% | +4.6% |
| 0.45-0.55 | 20 | 50.0% | −0.0% |
| 0.55-0.65 | 9 | 54.9% | −4.6% |
| 0.65-1.00 | 1 | 66.4% | **−7.2%** |

Correlation between market price and model-minus-market: **−0.644**, monotone
across every bucket. **79% of positive-EV legs sit on the underdog.**

The cause is the calibration itself. Model probabilities span 33.6-66.4%; the
market spans 26.5-73.5%. Shrinking the model was necessary for its own accuracy,
but it leaves the model strictly *less* discriminating than the market. Against
a sharper counterparty that mechanically manufactures apparent edge on every
longshot.

This is the cleanest negative result in the project: the "edge" is not
information the model has, it is confidence the model lacks. A genuine edge
would not line up monotonically with the market price.

## MLB vs Kalshi's closing line

907 games (2026-05-24 to 2026-08-05) with a genuine **pre-first-pitch** closing
price. Kalshi contracts trade through the game, so `last_price` on a settled
market knows the score; these come from the last candlestick strictly before
first pitch, with kickoff parsed from the ticker.

| | Log loss | vs market |
|---|---|---|
| Base rate | 0.69222 | +0.01057 |
| Model (raw) | 0.69690 | +0.01525 |
| **Model (calibrated)** | **0.68364** | **+0.00199** |
| Kalshi pre-game close | 0.68165 | — |

Mean book sum 1.0117 — a **1.17% spread**, far tighter than Pinnacle's 4.1%.
Median closing-hour volume 7,679 contracts.

**The model is +0.00199 behind, 95% CI [-0.00503, +0.00906] — statistically
indistinguishable from the closing line.** That is a different state from
soccer, where the market won decisively and the blend weight was 0.00 in every
single test. It is *not* evidence of edge; it is failure to detect a difference.

### The blend weight looked real and was not

In-sample, the optimal blend put **0.33** on the model — the first non-zero
weight anywhere in this project. Validated properly:

* Fitted on the first half (w = 0.15), applied to the second: improvement
  **+0.00085, 95% CI [-0.00082, +0.00248]**. Straddles zero.
* The second half's own best weight was 0.48, versus 0.15 on the first.
* Month by month: **w = 0.00 (May), 0.40 (June), 0.68 (July)**.

A weight swinging from 0 to 0.68 across three months is being fitted to noise.
The in-sample 0.33 does not survive.

One observation worth testing rather than believing: the model-minus-market gap
improves monotonically through the season — +0.02254 in May, +0.00110 in June,
−0.00238 in July. A plausible mechanism exists (early-season ratings lean on
last year; by July there is half a season of current data). It is equally
consistent with noise across three months. It is a hypothesis for next season,
not a finding.

### Method note

The first attempt at this fetch returned 24 games from 926 and exited cleanly.
Cause: HTTP 429 rate limiting, with non-200 responses treated as "no data". The
run reported success and produced a plausible-looking small sample. Retry with
backoff recovered 919 games — **99.6% versus 2.6%**. Any fetch that silently
drops data on a non-200 will eventually produce a confident wrong answer; the
fetcher now counts throttling explicitly and warns when retention falls below
60%.

## Feature ablations — what was tested and rejected

Every proposed improvement is run through the same test: the identical
walk-forward backtest with the feature on and off, scored on exactly the same
matches, with a bootstrap CI on the paired per-match difference. Plausibility is
not evidence.

### Per-team home advantage — REJECTED

Motivation: one home-advantage number per division cannot represent a ground
like Toluca's at 2,660m. Implemented as a ridge-shrunk per-team deviation.

19,578 matches, five countries, 2022-2026:

| Variant | Log loss | vs baseline | 95% CI | Verdict |
|---|---|---|---|---|
| off | 1.02891 | — | — | — |
| reg_home=4 | 1.03100 | +0.00208 | [+0.00112, +0.00306] | **HURTS** |
| reg_home=8 | 1.03036 | +0.00145 | [+0.00065, +0.00227] | **HURTS** |

Both confidence intervals sit entirely above zero. The extra parameters fit
noise. Not shipped.

### The altitude story was wrong

The feature above was built on a specific claim: that altitude explains why the
model rated Toluca at 45% where the market said 71%. Correlation between fitted
per-team home advantage and stadium elevation across Liga MX:

**−0.595. The wrong sign.**

Liga MX's strongest home grounds are Mazatlan (10m) and Tijuana (30m); among the
weakest are Pumas, Cruz Azul and Pachuca, all above 2,200m. The likely reason is
that nearly every Liga MX visitor is itself altitude-adapted, so the effect
barely registers in domestic results. Altitude may well punish a sea-level MLS
side, but that is not measurable from data in which every away team is
acclimatised — and it should not be asserted as though it were.

### Shots on target — evidence gathered before building

Predicting a team's next-half goals per game from its first-half record
(4,866 team-seasons, 5-fold CV):

| Features | CV RMSE |
|---|---|
| goals only | 0.3792 |
| **goals + SOT** | **0.3730** |
| SOT only | 0.3901 |

Raw SOT is a *worse* standalone predictor than goals, so the common "shots are
more stable than goals" claim does not survive contact with this data — raw SOT
ignores shot quality entirely. But it carries information goals do not: its
coefficient (+0.078) holds up alongside goals (+0.349), and the pair beats
either alone. That justified building the hybrid; the ablation decides whether
it ships.

### Shots hybrid — ACCEPTED

Same 19,578 matches, five countries, 2022-2026:

| Variant | Log loss | vs baseline | 95% CI | Verdict |
|---|---|---|---|---|
| off | 1.02891 | — | — | — |
| shot_weight=0.15 | 1.02826 | −0.00066 | [−0.00097, −0.00036] | **HELPS** |
| **shot_weight=0.30** | **1.02806** | **−0.00086** | [−0.00147, −0.00025] | **HELPS — best** |
| shot_weight=0.50 | 1.02849 | −0.00042 | [−0.00146, +0.00053] | no effect |
| shot_weight=0.75 | 1.03018 | +0.00127 | [−0.00025, +0.00277] | no effect |

The two smaller weights have confidence intervals entirely below zero. Beyond
0.30 the effect fades and then reverses, so **the optimum is interior** and the
search range was wide enough — the same edge-of-grid check that stopped the
hyperparameter tuning from producing a false conclusion. Shipped at 0.30. **This is the first feature
in the project to pass a proper ablation.** Everything previously adopted on
plausibility — tuned hyperparameters, altitude, per-team home advantage — failed
when tested.

Keep it in proportion: the gap to the closing line is +0.017 and this closes
roughly 5% of it. Real, statistically significant, and nowhere near enough to
beat the market. It also only applies to the ~51% of matches carrying shot data
(European divisions); everything else falls back to goals alone.

Worth noting what the result is NOT. Raw shots on target are a *worse* predictor
than goals. The gain comes from combining two noisy views of the same underlying
strength, not from shots being some purer signal. The popular framing —
"shots are more stable than goals, so model shots" — is contradicted by the data
here; only the hybrid helps.

## Live test: 2026-08-05 Leagues Cup, six fixtures

The first genuine out-of-sample test of the cross-league model, predictions
locked before kickoff.

| Fixture | Model | Market | Result |
|---|---|---|---|
| Inter Miami | 70% | 68% | Miami won 4-2 ✓ |
| FC Dallas | **58%** | 41% | Dallas won 2-0 ✓ **model beat market** |
| Orlando (at Orlando) | 42% | 40% | Orlando won 2-1 ✓ |
| Nashville | **61%** | 44% | Leon won 1-0 ✗ market right |
| Toluca | **45%** | **72%** | Toluca won 3-0 ✗ market right |
| LAFC | 53% | 33% | 1-1 draw — both had the draw at 26% |

**4 of 6 picks correct (66.7%), but log loss 0.711 against the market's 0.489.**
Winning two thirds of the picks while losing decisively on log loss is exactly
why this ledger scores probabilities rather than win rate: the Toluca pick
"won" at 45%, but the market's 72% was far better calibrated, and being right
for the wrong reason is not edge.

On the four material disagreements the score was **1-2 with one dead heat** —
Dallas to the model, Nashville and Toluca to the market, LAFC identical (both
26% on the draw, and neither favourite won).

Two process notes worth keeping:

* **Deferring to the market beat the model, twice.** Taking the market's Toluca
  over the model's 45%, and passing on LAFC, were both correct.
* **A settlement was entered from bad data.** An ESPN extraction labelled an
  in-progress first-half score as "Final" and two predictions were settled on
  it. Caught, reversed, and the ledger gained an `unsettle` command that
  requires a reason and preserves the original in `correction_history`. A
  scorecard that can be quietly rewritten is worthless.

### Refit with the new results

Adding the six 2026 matches to the bridge:

| Bridge | League gap |
|---|---|
| 2025 only | +0.196 |
| 2025 + 2026 | **+0.165** |

A shift of −0.030, worth 1-2 points of win probability. **That does not explain
disagreements of 15-18 points.** The gap was never the real problem — the error
is team-level and still unidentified, which matches the earlier diagnostic where
Orlando sat within 2.4 points of the market while three other fixtures were far
outside it.

MLS went 3W 1D 2L on the night (50%, versus 43% across 2025), but on a 9-8 goal
difference and with both defeats coming against the stronger Liga MX sides.

## NFL and WNBA — margin models

Football and basketball scores are nowhere near Poisson (NFL averages 44 points,
WNBA 163), so these use a different family: ridge-regularised weighted least
squares on points, with team offence/defence, a league level, and home
advantage. Margins are close to normal (NFL sd 14.6, WNBA sd 13.8), which yields
three markets from one fit — moneyline, spread and total.

NFL data comes from nflverse and ships with **closing spreads, totals and
moneylines**, so the market benchmark needs no separate collection.

### NFL, 3,025 games

| Market | Model | Benchmark | Verdict |
|---|---|---|---|
| Moneyline | 0.6406 | base rate 0.6886 | much better than nothing |
| Moneyline | 0.6407 | **closing line 0.6125** | market better by +0.028 |
| Spread | 50.15% | 52.4% needed to beat juice | **no edge** |
| Total | 50.45% | 52.4% | **no edge** |
| Margin MAE | 10.23 | market spread 9.81 | market better |

### WNBA, 2,304 games

Log loss 0.6286 against a base rate of 0.6873; accuracy 65.2%; margin MAE 10.19.
No closing-line benchmark is available (Basketball Reference carries no odds),
so the model can be shown to work but not to beat anything.

### Two bugs that manufactured a fake edge

The first NFL run reported **69.5% spread accuracy** and a model beating the
closing spread on margin MAE. That would be a world-class edge. It was two
faults compounding:

* **Spread sign inverted.** nflverse uses POSITIVE `spread_line` for a home
  favourite, so home covers when `margin - spread > 0`. The code used
  `margin + spread` in both the prediction and the scoring — consistently
  wrong, so a mis-signed prediction was graded against a mis-signed target.
  Checked against the data rather than assumed: the correct rule gives 47.6%
  home covers, the inverted one 58%.
* **No global intercept.** Ridge shrank team ratings toward zero, leaving
  home advantage as the only unpenalised parameter able to explain ~22 points a
  side. It absorbed the league scoring level, producing a 72% mean home win
  probability against an actual 54.8%, and totals of 39.0 against an actual
  45.7.

Corrected, both spread and total sit at ~50%, which is what an efficient closing
line should look like to a model with no edge.

### Starting quarterback — ACCEPTED

The NFL analogue of MLB's starting pitcher, added as a per-QB effect on his own
team's points, heavily regularised.

| reg_qb | ML log loss | vs off | 95% CI | Verdict |
|---|---|---|---|---|
| off | 0.64063 | — | — | — |
| 0.5 | 0.63720 | −0.00343 | [−0.01120, +0.00465] | no effect |
| 1 | 0.63467 | −0.00595 | [−0.01271, +0.00109] | no effect |
| 2 | 0.63283 | −0.00780 | [−0.01358, −0.00227] | HELPS |
| **4** | **0.63232** | **−0.00830** | [−0.01264, −0.00387] | **HELPS — best** |
| 12 | 0.63430 | −0.00632 | [−0.00877, −0.00390] | HELPS |
| 30 | 0.63689 | −0.00373 | [−0.00497, −0.00254] | HELPS |

The optimum is interior — too little regularisation loses significance, too much
degrades. The fitted QB differential spans −9.4 to +9.0 points, an 18-point
swing between the best and worst starters, which is consistent with how the sport
is usually described.

This is the **second feature in the project to survive a proper ablation** (after
the goals+shots hybrid). It closes 29% of the gap to the closing moneyline,
taking it from +0.028 to roughly +0.020.

Notably it does **not** improve spread picking (50.15% to 49.64%). It sharpens
who wins, not by how much.

### The QB fallback bug

nflverse does not record starting quarterbacks for FUTURE games, so every
upcoming fixture hit the "unknown QB" fallback, which mapped to 0.0 on the
assumption that zero meant league average. It did not: the fitted QB effects had
a playing-time-weighted mean of **+1.011**, so the fallback silently subtracted
about a point per team — two points per game.

The result was a model sitting below the closing total in **84% of upcoming
games**, with a mean gap of −1.75. In isolation that looks like a systematic
edge on unders. It was a centring error.

In-sample bias was exactly 0.00 throughout, because quarterbacks ARE known
there. The fault only existed on the games one would actually bet.

Fixed by re-centring the QB vector to a weighted mean of zero and moving the
offset into the intercept — an exact reparameterisation, so predictions for
known quarterbacks are unchanged and the ablation result stands. Upcoming totals
now sit +0.27 from the line rather than −1.75.

## NFL wind and the totals market

The clearest candidate edge this project has produced, and still not verified.

Wind is recorded for the 5,206 outdoor games with a closing total. Betting the
UNDER at a threshold fixed in advance:

| Period | n | Under % | 95% CI | ROI at −110 |
|---|---|---|---|---|
| Full 1999-2026 | 1,280 | 56.09% | [53.4%, 58.8%] | +7.09% |
| First half (1999-2011) | 684 | 54.82% | [51.0%, 58.5%] | inconclusive |
| Second half (2011-2026) | 596 | 57.55% | [53.5%, 61.4%] | +9.87% |
| Last 10 seasons (2016+) | 415 | 58.31% | [53.5%, 62.9%] | +11.33% |

Break-even at −110 juice is 52.38%. The interval excludes it over the full
sample, the second half and the last decade, and the effect is stronger in
recent data rather than decaying.

Reasons it is not being treated as an edge:

* **The threshold came from the data.** 12 mph was chosen after inspecting the
  wind buckets. Fixing it before this particular test is a weaker guarantee than
  genuine pre-registration.
* **15 mph does not hold up.** It clears the bar on the full sample but goes
  inconclusive on both the second half and the last decade. A real effect should
  not be that fragile to a threshold shift.
* **The era pattern is unstable.** Mean residual (total − line) at wind ≥12 runs
  +0.08, −0.49, −0.97, **−2.43**, −0.35 across five eras. One window does most
  of the work and the most recent collapses toward zero.
* **Wind is recorded, not forecast.** nflverse stores what actually happened; a
  bettor has only a forecast. The backtest is therefore optimistic about what is
  knowable beforehand.
* **Structural prior.** This is a simple, visible, decade-old pattern in the most
  heavily modelled betting market in existence.

The only test that settles it is forward results. `margin_predict.py` flags
qualifying games so they can be logged and scored; roughly 200 live games would
be needed, which is about one NFL season.

## By era

| Period | n | Model | Market | Gap |
|---|---|---|---|---|
| 2015-2017 | 16,658 | 1.0154 | 0.9989 | +0.0165 |
| 2018-2020 | 18,437 | 1.0170 | 1.0026 | +0.0143 |
| 2021-2023 | 20,668 | 1.0140 | 0.9953 | +0.0187 |
| 2024-2026 | 12,541 | 1.0126 | 0.9950 | +0.0176 |

Stable. The market has not got dramatically sharper, and neither has the gap
closed.

## Side findings

**The Dixon-Coles low-score correction has decayed out of modern football.**
Fitted rho is ≈ −0.03 on 2005-2010 data, matching the original paper's
estimates, but ≈ 0 on post-2020 data. Verified against a numerical gradient and
across four eras. The correction is now doing nothing.

**Over/under 2.5 goals** log loss 0.68552 vs base rate 0.69313 — a thin margin.
Totals are harder than 1X2 for this model. No closing totals odds were stored in
the backtest, so there is no market comparison for it yet.

## What would actually be needed to beat the market

Not more history. The model already sees 150,000 matches; the reel's "every game
since 1872" is not the constraint, and pre-war international results carry
almost no signal about a 2026 fixture.

The market prices things this model cannot see:

- team news, injuries and suspensions, confirmed lineups
- rotation ahead of European fixtures and cup ties
- motivation (already relegated, already champions, dead rubbers)
- managerial changes
- shot-quality data (xG) rather than realised goals

The plausible paths to a real edge are (a) faster reaction than the market to
information rather than better modelling of the same information, or (b) markets
that are genuinely thin — obscure divisions and in-play, where the closing line
is much weaker than it is in these 22 leagues.

## Is Kalshi a softer market than the bookmakers? (2026-08-11)

Everything above benchmarks against BOOKMAKER odds — Pinnacle, market average.
We do not trade those. We trade Kalshi, and only MLB closes had ever been
collected, so the question had never been asked anywhere else. Kalshi's median
bid-ask spread ranges from 1c (MLB, WNBA) to 29c (Liga MX), a 29x difference,
which made "the thin books are soft" a reasonable hypothesis.

`fetch_kalshi_closes.py` now collects pre-game prices for every series. NOTE:
these are prices ~8h before market close, NOT closing lines. Kalshi exposes no
reliable start time (`occurrence_datetime` is the expected EXPIRATION, and
anchoring to it produced pure post-game lookahead — every winning leg at 0.99).
Anchoring to `close_time` minus 8h clears the longest plausible event. That is
also closer to when this pipeline could actually trade, since it commits picks
up to 36h ahead.

2,899 legs across 9 series, 2026-05-25 .. 2026-08-11.

### Market informativeness

| market | events | ways | overround | Kalshi LL | base LL | info gain |
|---|---|---|---|---|---|---|
| Bundesliga 2 | 9 | 3 | 1.004 | 0.6782 | 0.6365 | −0.0417 |
| Liga MX | 28 | 3 | 1.020 | 0.6354 | 0.6365 | +0.0011 |
| MLB | 992 | 2 | 1.012 | 0.6835 | 0.6931 | +0.0096 |
| Scottish Prem | 12 | 3 | 1.010 | 0.6087 | 0.6365 | +0.0278 |
| Allsvenskan | 52 | 3 | 1.009 | 0.5877 | 0.6365 | +0.0488 |
| The Hundred | 23 | 2 | 1.014 | 0.6372 | 0.6931 | +0.0559 |
| Eliteserien | 51 | 3 | 1.013 | 0.5342 | 0.6365 | +0.1023 |
| WNBA | 201 | 2 | 1.010 | 0.5769 | 0.6931 | +0.1163 |

Liga MX's near-zero info gain looks like the soft market we were hunting. It is
not evidence of one. Info gain conflates "the market knows nothing" with "the
outcomes are genuinely unpredictable" — a perfectly priced league of coin flips
scores zero too. n=28 settles nothing either way.

### Calibration — the actual test for softness

A soft market is MIS-priced in a direction you can bet against. Pooled across
every market, no price band is outside 2 standard errors:

| band | n | said | actual | error |
|---|---|---|---|---|
| 20-30% | 231 | 25.1% | 22.5% | −2.6% |
| 40-50% | 912 | 45.4% | 45.0% | −0.5% |
| 50-60% | 929 | 54.4% | 54.9% | +0.5% |
| 60-70% | 270 | 63.7% | 60.7% | −3.0% |

Favourite-longshot bias, the classic soft-market signature, is absent
everywhere it can be measured. MLB favourites (n=510) were priced 59.8% and won
59.4%.

**Conclusion: Kalshi is well calibrated in every market with a usable sample.
The soft-market hypothesis is not supported.** The thin books remain untested
on sample size, and their spreads (29c on Liga MX) would consume any edge that
did exist — a mispriced market you cannot cross into cheaply is not an
opportunity.

The data now accumulates automatically, so the thin markets become testable
with a few more months of fixtures.

---

## Totals markets: soccer works, baseball does not (2026-08-19)

We priced one market per fixture while Kalshi listed roughly seven. Dixon-Coles
and the MLB model both produce a full score matrix, and both already computed
totals probabilities that nothing consumed. Kalshi lists ~450 open total
markets at 1c spreads on leagues already on the board, so this was the largest
available increase in pick volume with no new data source.

Both were put through the same gate. They did not get the same answer.

### Soccer — over 2.5 goals: SHIPPED (paper lane)

Walk-forward, 24,898 matches (`totals_backtest.py`).

| test | result |
|---|---|
| vs constant base rate | **+0.0067 log loss — beats it** |
| vs closing line | 0.68110 vs 0.66899, **blend weight 0.00** |
| 95% CI on that gap | [+0.00748, +0.01662] — market significantly sharper |
| \|gap\|>10% subset | model 0.7418 vs market 0.6856 — disagreement is model error |

Same verdict as the winner market: real information about goals, no edge over
the price. Platt scaling on major divisions moved the over side from stated
66.2%/actual 63.8% to **stated 64.7%/actual 64.1%** on a held-out time split.

**Excluded:** the under side stays broken after calibration (stated 62.9%,
actual 51.0%), and BTTS has **no skill at all** (−0.0002 against a constant).

### MLB — over 8.5 / 9.5 runs: REJECTED

Walk-forward, 6,572 games.

| line | vs base rate | ≥60% band |
|---|---|---|
| over 8.5 | **−0.0247 — worse than a constant** | stated 69.7%, actual 54.4% |
| over 9.5 | **−0.0245 — worse than a constant** | stated 68.5%, actual 48.1% |

At a stated 80–90%, the actual rate is 56%. This is not a tuning problem. Runs
are modelled as two independent Poissons, and baseball scoring is heavily
overdispersed — one big inning breaks the assumption. The winner market
survives because the error largely cancels when taking the difference of two
similarly-misspecified distributions; a total is exposed to it directly.
Calibration only rescales, so it cannot repair a variance misspecification of
this size.

**The lesson worth keeping:** "totals worked for soccer" does not transfer.
The two models fail in different places, and each market has to earn its own
evidence. A warning sits on the returned fields in `mlb_model.predict` so the
numbers cannot be picked up without the measurement attached.

### Data gap closed

`fetch_data.KEEP` had never captured closing over/under odds, which is why the
totals market could not be compared against the line retrospectively — we had
the goals but not the price. Now keeping `PC/AvgC/MaxC >2.5`, rebuilt from the
454 cached CSVs with no re-download: **49,858 matches with closing O/U prices
across 21 divisions, 2019–2026.**
