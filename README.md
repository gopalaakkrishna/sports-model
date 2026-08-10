# Football match prediction model

A Dixon-Coles bivariate Poisson goal model for European club football, with
walk-forward backtesting against Pinnacle closing odds.

It predicts **goals**, and everything else falls out of the goal distribution:
win/draw/loss, exact scoreline, over/under 2.5, both teams to score.

## What makes this different from the version in the Instagram reel

The reel's model was almost certainly the same core idea (team ratings ->
expected goals -> Poisson -> outcome probabilities). Three things are added here:

1. **Walk-forward backtesting.** Every prediction is made using only data
   available before that match. No lookahead.
2. **A real benchmark.** Performance is measured against de-vigged Pinnacle
   closing odds, not against a coin flip. Closing odds at a sharp book are the
   strongest public forecast that exists. "My model says 76%, the market says
   69%" is only value if you have demonstrated your model beats the closing
   line over thousands of matches. Usually it does not.
3. **Calibration.** Raw Poisson models are over-confident. The calibration step
   measures and corrects this, and reports the honest answer to "does this model
   know anything the market doesn't?"

## Layout

```
src/fetch_data.py        European divisions: results + closing odds
src/fetch_new_leagues.py year-round leagues (MLS, Argentina, Brazil, ...)
src/data.py              combined loader + country groupings
src/model.py             Dixon-Coles fit and prediction
src/team_names.py        name resolution between the fixtures and season feeds
src/test_model.py        gradient check + smoke test
src/backtest.py          walk-forward evaluation vs the market
src/tune.py              hyperparameter grid search (early window only)
src/calibrate.py         temperature scaling + market blend
src/analyze_edge.py      where (if anywhere) the model competes
src/eval_vs_avg.py       robustness to the choice of benchmark
src/predict_upcoming.py  forecasts for scheduled fixtures
src/predict_cross_league.py  matches between leagues that never meet (Leagues Cup)
src/ratings.py           current team ratings + on-demand matchup predictions

  baseball
src/fetch_mlb.py         MLB results + probable starters (MLB StatsAPI)
src/mlb_model.py         Poisson runs: team, starting pitcher, park
src/test_mlb.py          gradient check + smoke test
src/mlb_backtest.py      walk-forward + Platt calibration
src/mlb_predict.py       today's slate priced against Kalshi
src/fetch_kalshi_mlb.py  pre-game closing lines from Kalshi candlesticks
src/mlb_vs_market.py     model vs the closing line

  markets and tracking
src/prediction_markets.py  Kalshi/Polymarket quotes with a tradeability screen
src/kalshi_edge.py         soccer model vs Kalshi three-way markets
src/ledger.py              locked predictions, settlement, scorecard

data/raw/                cached CSVs and the combined parquets
reports/                 backtest logs, tuning results, prediction CSVs
```

## Tracking predictions

```bash
python src/ledger.py lock --sport soccer --event "A v B" --market 1X2 \
    --pick HOME --model-prob 0.52 --market-prob 0.44 --odds 2.2 --venue kalshi
python src/ledger.py settle --id 3 --outcome HOME
python src/ledger.py report
```

Locks record the market price at that moment — without it you cannot later tell
whether you beat the market or merely agreed with it. Scoring is by log loss
against the market, not win rate: backing favourites yields a fine win rate and
can still lose money.

## Daily use

```bash
python src/refresh_all.py            # every source, in parallel
python src/today.py                  # everything starting from now, ET, with confidence
python src/daily_slate.py --n 5 --log # ranked picks, written to the ledger
python src/settle.py                 # resolve finished predictions
python src/ledger.py report          # scorecard
```

In-play (MLB only):

```bash
python src/inplay_live.py --watch 40 --interval 150   # log model vs live market
python src/inplay_score.py                            # score once games settle
```

## Sports covered

| Sport | Model | vs base rate | vs market |
|---|---|---|---|
| Soccer (33 divisions) | Dixon-Coles + shots hybrid | much better | −0.017 |
| MLB pre-game | Poisson runs + pitcher + park | +0.009 | at parity |
| MLB in-play | Poisson remaining outs + team rates | +0.22 | untested |
| NFL | Margin/total + starting QB | +0.056 | −0.020 |
| WNBA | Margin/total | +0.059 | less discriminating |

Every model beats doing nothing. None has been shown to beat its market.

## Coverage

33 divisions. European leagues (England 4 tiers, Scotland 4, Germany 2, Italy 2,
Spain 2, France 2, Netherlands, Belgium, Portugal, Turkey, Greece) from 2005,
plus year-round leagues (MLS, Argentina, Brazil, Mexico, Japan, China, Norway,
Sweden, Finland, Denmark, Ireland, Poland, Romania, Russia, Austria,
Switzerland) from 2012. ~213,000 matches total.

The year-round leagues matter because they run through the European off-season —
in August, MLS and the South American leagues are live while Europe is not.

## The model

For a match between home team `i` and away team `j`:

```
lambda = exp(attack_i + defence_j + home_adv_div)    expected home goals
mu     = exp(attack_j + defence_i)                   expected away goals
```

Goals are Poisson around those rates, with the Dixon-Coles `tau` correction for
low scores. Fitting is weighted maximum likelihood with an exponential time
decay, so recent matches count more, plus an L2 ridge on the ratings.

Two deliberate choices:

- **Fits are per country, pooling all divisions.** Promotion and relegation link
  the divisions into one connected graph, so a promoted side arrives with a real
  rating rather than a cold start. Home advantage is fitted per division.
- **Ridge regularisation** both resolves the additive identifiability of
  attack/defence and shrinks thin-sample teams toward average.

## Usage

```bash
python src/fetch_data.py          # European divisions (cached)
python src/fetch_new_leagues.py   # MLS, Argentina, Brazil, ... (year-round)
python src/test_model.py          # verify the fit
python src/backtest.py            # walk-forward evaluation
python src/calibrate.py           # fit calibration, report edge vs market
python src/predict_upcoming.py    # forecasts for scheduled fixtures
```

Backtest a subset of countries:

```bash
python src/backtest.py --countries "USA,Argentina,Brazil" --out mls.parquet
```

Ratings and one-off matchups, useful when the fixtures feed has not yet
published the next round:

```bash
python src/ratings.py --country USA --match "Inter Miami" "LA Galaxy"
```

## Data

football-data.co.uk, 2005-present, 22 divisions across 11 countries: ~151,000
matches, ~95,000 with Pinnacle closing odds.

Two traps worth recording:

**UTF-8 BOM.** The recent season CSVs carry one. Read as latin-1 the first
column name becomes `ï»¿Div` rather than `Div`, which silently nulls the
division and drops every match from 2024 onward. `fetch_data.py` strips the BOM
and runs integrity checks that would catch a repeat.

**Pinnacle odds stopped.** Coverage decays from Nov 2025 and is zero from
Feb 2026, so the sharpest benchmark is unavailable for recent and future
matches. `AvgH/AvgD/AvgA` (average across books) has full coverage and is what
`predict_upcoming.py` compares against. `eval_vs_avg.py` confirms the backtest
conclusion holds against both lines.

**Team names differ between feeds.** fixtures.csv and the season files disagree
("Dundee Utd" vs "Dundee United", "Raith" vs "Raith Rvs"). `team_names.py`
resolves them exact -> normalised -> curated alias -> fuzzy, and logs every
fuzzy match for review. Note "Dundee Utd" is 0.80 string-similar to "Dundee", a
different club, so a blocklist prevents fuzzy matching on ambiguous names. When
the big leagues restart, run `predict_upcoming.py` and add anything it reports
as unresolved to `ALIASES`.
