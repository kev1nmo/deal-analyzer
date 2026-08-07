# Phase 3 findings: baseline and validation design

## 1. Train / test split (time holdout)

The newest 20% of listings by date are the test set. Models never see them until final evaluation.

| set   |   listings | from       | to         |   median_rent |   share_1br |   share_3br_plus |   share_coarse_location |   map_points |   median_km_to_downtown |
|:------|-----------:|:-----------|:-----------|--------------:|------------:|-----------------:|------------------------:|-------------:|------------------------:|
| train |       3034 | 2018-12-07 | 2019-09-18 |      1340.000 |       0.405 |            0.135 |                   0.063 |          131 |                  15.494 |
| test  |        758 | 2019-09-18 | 2019-12-26 |      1195.000 |       0.364 |            0.234 |                   0.057 |           49 |                  28.271 |

## 2. Spatial cross-validation

The 131 map points are grouped into 40 location clusters with KMeans (UTM meters, weighted by listings per point). The clusters are dealt into 5 folds, so every listing in a cluster lands in the same fold. Nearby listings are never split between train and validation.

|   fold |   listings |   clusters |   median_rent |
|-------:|-----------:|-----------:|--------------:|
|      0 |        603 |          7 |          1422 |
|      1 |        611 |          8 |          1405 |
|      2 |        599 |          7 |          1300 |
|      3 |        611 |          9 |          1285 |
|      4 |        610 |          9 |          1340 |

## 3. Baselines

- `global_median`: one number for every listing.
- `beds_median`: median rent for the bedroom count.
- `zip_beds_median` (the spec's baseline): median rent for the ZIP x bedroom combination, falling back to the bedroom median when that combination was not in training.

### Cross-validation on the training set (mean over 5 folds, $)

| baseline        | cv      |   mae |   mae_fold_sd |   mape |
|:----------------|:--------|------:|--------------:|-------:|
| global_median   | spatial | 317.8 |          49.4 |   22.2 |
| global_median   | random  | 316.0 |          16.2 |   22.1 |
| beds_median     | spatial | 307.2 |          56.2 |   21.4 |
| beds_median     | random  | 305.2 |          14.7 |   21.3 |
| zip_beds_median | spatial | 313.6 |          63.7 |   22.0 |
| zip_beds_median | random  | 226.7 |           7.8 |   15.5 |

Random folds make `zip_beds_median` look better than it is: random CV MAE $227 vs spatial CV $314. Under random folds, listings from the same map point land in both train and validation, so the ZIP median has already seen near-copies. Spatial CV is the honest estimate for a neighborhood the model has not seen. It is the CV score used for all tuning from Phase 4 on.

### What spatial CV and the test set each measure

- Under spatial CV, `zip_beds_median` found its ZIP x beds cell for only 6% of validation listings. Holding out a whole cluster holds out its ZIPs too, so the baseline falls back to the bedroom median and scores about the same as `beds_median`. ZIP identity does not carry over to new areas. Phase 4 models need continuous location features (distances, tract and ZIP statistics) to beat it.
- The test set is a different question: 88% of test listings sit on a map point seen in training and 95% in a seen ZIP. It measures *later listings in mostly familiar places*. Spatial CV measures *new places*. Report both.
- The test period skews suburban (see the median km to downtown above) and its median rent is lower, even with more 3+ bedroom units. Late-2019 listings are a shifted sample, not a random one.

### Test set (fit on all training data, $)

| baseline        |   mae |   mape |   median_ae |   mae_ci_low |   mae_ci_high |
|:----------------|------:|-------:|------------:|-------------:|--------------:|
| global_median   | 263.1 |   23.5 |       225.0 |        249.4 |         277.6 |
| beds_median     | 245.2 |   22.0 |       218.5 |        232.5 |         258.7 |
| zip_beds_median | 221.7 |   18.2 |       178.2 |        207.9 |         237.7 |

`mae_ci_low` / `mae_ci_high`: 95% bootstrap interval for test MAE (2,000 resamples).

**Baseline to beat: `zip_beds_median` test MAE $222** (95% CI $208-$238), MAPE 18.2%. Phase 4 target (20% lower): **MAE <= $177**.

How test listings were predicted by the main baseline:

| prediction_source    |   share |
|:---------------------|--------:|
| ZIP x beds           |   0.831 |
| beds only (fallback) |   0.169 |

Test MAE by bedrooms (main baseline):

|   bedrooms |   listings |   median_rent |   mae |
|-----------:|-----------:|--------------:|------:|
|          1 |        276 |        1034.0 | 184.3 |
|          2 |        305 |        1184.0 | 220.2 |
|          3 |        145 |        1430.0 | 278.5 |
|          4 |         32 |        1585.0 | 300.5 |
