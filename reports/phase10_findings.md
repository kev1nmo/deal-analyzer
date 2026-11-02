# Phase 10 findings: monitoring and retraining

## Setup

- Every API prediction is logged to Postgres (`predictions`, with its model feature vector and model version). Real rents arrive later through `POST /actuals` into `rent_actuals`, linked by prediction id.
- No real post-2019 rents exist for these listings, so the drill replays the newest 20% of listings (the Phase 3 test set, Sep and Dec 2019) as live traffic through the production analyzer, then records their real rents as actuals. Replay rows are tagged by `source`, so they never mix with API traffic.
- Drift rule (SQL, per month and model version, months with at least 30 actuals): MAE above 1.25 x the champion's reference MAE ($155, limit $194), or 80%-range coverage below 70%.
- Retraining: the newest 30% of labeled rows (by date) are the recent holdout. The challenger is refit with the champion's hyperparameters on the original training data plus the older 70%. It is registered in the MLflow Model Registry with alias `challenger`, and promoted to `champion` only if its holdout MAE is lower **and** the 95% paired-bootstrap CI of the gain is above zero.

## Scenario `replay`: late-2019 listings replayed as live traffic, real rents

| month   |   n |     mae |   bias_pct |   coverage |   mae_drift |   coverage_drift | drift   |
|:--------|----:|--------:|-----------:|-----------:|------------:|-----------------:|:--------|
| 2019-09 | 631 | 147.219 |     -0.014 |      0.789 |           0 |                0 | False   |
| 2019-11 |   3 | 385.767 |      0.097 |      0.667 |           0 |                0 | False   |
| 2019-12 | 124 | 188.846 |     -0.017 |      0.790 |           0 |                0 | False   |

Drift flagged: **no**. Retrain reason: forced (no drift flagged). 758 labeled rows; 530 added to training, 228 held out.

| | MAE | 80% coverage |
|---|---:|---:|
| champion (v1) | $185 | 78.1% |
| challenger (v2) | $187 | 74.1% |

Gain -0.8% (95% CI -3.8% to 2.1%). **Decision: keep champion.**

## Scenario `replay-shock`: same listings, every actual rent raised 15% (simulated market jump)

| month   |   n |     mae |   bias_pct |   coverage |   mae_drift |   coverage_drift | drift   |
|:--------|----:|--------:|-----------:|-----------:|------------:|-----------------:|:--------|
| 2019-09 | 631 | 235.071 |      0.118 |      0.634 |           1 |                1 | True    |
| 2019-11 |   3 | 607.201 |      0.215 |      0.667 |           0 |                0 | False   |
| 2019-12 | 124 | 277.458 |      0.116 |      0.726 |           1 |                0 | True    |

Drift flagged: **yes**. Retrain reason: drift. 758 labeled rows; 530 added to training, 228 held out.

| | MAE | 80% coverage |
|---|---:|---:|
| champion (v1) | $278 | 68.4% |
| challenger (v2) | $243 | 71.9% |

Gain 12.5% (95% CI 9.8% to 15.5%). **Decision: PROMOTE.**
