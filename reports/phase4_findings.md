# Phase 4 findings: rent model

## 1. Setup

- 46 features in 6 groups: size, location, neighborhood (Census and HUD), market (ZORI with metro fallback, mortgage rate), listing (amenity flags, pets, photo) and nearby rent (rent per sq ft at the nearest training map points).
- Nearby rent is leakage-safe. During training each listing sees only *other* map points, never its own. At prediction time it sees every training point. It is computed inside the model pipeline, so in CV it only ever sees the training folds.
- Both models predict log(rent) and convert back to dollars.
- Tuning used spatial CV on the training set only (Phase 3 folds). Ridge: 40 grid points. XGBoost: 60 Optuna trials. The winner was picked on CV MAE before the test set was opened.

## 2. Spatial cross-validation (training set, $)

| model                      |   cv_mae |   cv_mape |   cv_median_ae |   cv_mae_fold_sd |   gain_vs_baseline_pct |
|:---------------------------|---------:|----------:|---------------:|-----------------:|-----------------------:|
| zip_beds_median (baseline) |    313.6 |      22.0 |          232.0 |             63.7 |                    0.0 |
| ridge                      |    243.0 |      16.5 |          171.7 |             42.7 |                   22.5 |
| xgboost                    |    233.3 |      15.8 |          160.0 |             44.4 |                   25.6 |

Best Ridge: alpha = 1000.0, k = 12 nearby points. Best XGBoost: k = 8, objective = reg:squarederror, n_estimators = 400, learning_rate = 0.02981, max_depth = 5, min_child_weight = 6.81, subsample = 0.9862, colsample_bytree = 0.4454, reg_lambda = 1.147, reg_alpha = 0.3006.

## 3. Test set (newest 20%, opened once)

| model                      |   mae |   mape |   median_ae |   mae_ci_low |   mae_ci_high |   gain_vs_baseline |   gain_ci_low |   gain_ci_high |
|:---------------------------|------:|-------:|------------:|-------------:|--------------:|-------------------:|--------------:|---------------:|
| zip_beds_median (baseline) | 221.7 |   18.2 |       178.2 |        207.9 |         237.7 |              nan   |         nan   |          nan   |
| ridge                      | 170.9 |   14.1 |       121.2 |        159.3 |         183.6 |               22.9 |          17.6 |           28.0 |
| xgboost                    | 155.0 |   12.4 |       114.8 |        143.9 |         167.1 |               30.1 |          25.3 |           34.7 |

`gain_vs_baseline`: % reduction in MAE vs the ZIP x beds median, with a 95% paired bootstrap CI (the same resampled listings for both models).

**Winner (chosen on CV): xgboost.** Test MAE $155 vs baseline $222: 30.1% lower (95% CI 25.3% to 34.7%). Target was 20% lower (MAE <= $177): **met**.

## 4. By region (county) - test MAE / MAPE

| county   |   listings |   baseline_mae |   baseline_mape |   ridge_mae |   ridge_mape |   xgboost_mae |   xgboost_mape |
|:---------|-----------:|---------------:|----------------:|------------:|-------------:|--------------:|---------------:|
| Cobb     |        412 |          206.1 |            16.9 |       153.9 |         12.6 |         139.4 |           11.3 |
| Fulton   |        188 |          266.6 |            20.1 |       240.6 |         18.8 |         219.2 |           16.5 |
| Gwinnett |         73 |          137.0 |            12.2 |       131.1 |         12.8 |          90.0 |            8.5 |
| Other    |         48 |          231.3 |            19.9 |       106.7 |          8.6 |         102.3 |            7.9 |
| DeKalb   |         34 |          336.5 |            34.8 |       165.5 |         16.2 |         203.4 |           18.2 |
| Clayton  |          3 |          155.3 |            11.3 |       183.0 |         16.0 |         141.7 |            9.9 |

## 5. By price band (actual rent) - test MAE / MAPE

| price_band   |   listings |   baseline_mae |   baseline_mape |   ridge_mae |   ridge_mape |   xgboost_mae |   xgboost_mape |
|:-------------|-----------:|---------------:|----------------:|------------:|-------------:|--------------:|---------------:|
| < $1,000     |        164 |          240.3 |            28.4 |       188.8 |         22.3 |         148.9 |           17.6 |
| $1,000-1,299 |        308 |          141.5 |            12.4 |       119.1 |         10.5 |         102.4 |            9.0 |
| $1,300-1,599 |        210 |          251.6 |            17.5 |       163.7 |         11.3 |         167.6 |           11.6 |
| $1,600+      |         76 |          424.0 |            21.7 |       362.0 |         18.4 |         346.5 |           17.5 |

Bands use the *actual* rent, so every model overpredicts the cheapest band and underpredicts the priciest (regression to the middle). Compare models within a band, not bands with each other.

## 6. Which feature groups matter? (XGBoost, spatial CV, one group dropped at a time)

| dropped             |   n_features |   cv_mae |   mae_change |
|:--------------------|-------------:|---------:|-------------:|
| none (all features) |           46 |    233.3 |          0.0 |
| size                |           41 |    264.5 |         31.2 |
| location            |           35 |    239.5 |          6.2 |
| neighborhood        |           37 |    240.7 |          7.5 |
| market              |           42 |    238.3 |          5.0 |
| listing             |           31 |    247.2 |         14.0 |
| nearby_rent         |           44 |    244.5 |         11.2 |

`mae_change`: how much CV MAE rises when the group is removed. Bigger means the group carries more information the others cannot replace.
