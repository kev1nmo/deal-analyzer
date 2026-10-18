# Decisions

## Phase 1: Data

- **Listings source:** UCI "Apartments for Rent Classified" (100K, CC BY 4.0), filtered to the Atlanta area. That leaves 4,299 listings (Dec 2018 to Dec 2019), below the ~20K target. No licensed API offered free Atlanta listings with history, and scraping is out of scope.
- **Database:** a single DuckDB file (`data/deals.duckdb`), built from Parquet files in `data/raw/`.

## Phase 2: EDA and statistics

- **Cleaning rules** (`src/clean.py`): drop exact reposts (keep the latest), > 4 bedrooms, sizes outside 250-3,500 sq ft, units under 220 sq ft per bedroom, and rent/sq ft outliers (|robust z| > 3.5 within each bedroom group). 4,299 → 3,792 rows.
- **Location repair:** listing coordinates are rounded geocodes (157 unique points). A listing whose city differs from its point's majority city is moved to its own city's main point (110 moved, 11 dropped). 233 listings on the Midtown fallback point are flagged `coarse_location`.
- **Spatial features:** distances are computed in UTM 16N (EPSG:32616, meters). ZIP comes from ZCTA polygons, with nearest-polygon fallback. Tract ACS stats come from 2024 cartographic tract boundaries for all of Georgia. Transit means MARTA heavy rail (GTFS route_type 1, 38 stations).
- **Hypothesis data:** the listings have no building-size field, so the 3-4 unit test uses Census ACS 1-year PUMS 2022-2024 (BLD = units in structure). It covers renter households in Fulton, DeKalb, Cobb, Gwinnett and Clayton PUMAs.
- **Hypothesis model:** WLS of log(rent) on building size (ref 5+ units), bedrooms, rooms, decade built, PUMA and year, using household weights and HC1 standard errors. Robustness checks: unweighted, PUMA-clustered SE, and gross rent. Result: 3-4 unit buildings rent at -15.3% (95% CI -21.2% to -9.0%).
- **Timing mismatch:** HUD SAFMR and ACS are newer than the listings. They are used only as relative neighborhood signals. ZORI and mortgage rates are matched to the listing month.

## Phase 3: Baseline and validation

- **Test set:** the newest 20% of listings by `listed_at` (758 listings, 2019-09-18 to 2019-12-26). The cutoff is the date boundary closest to 20%, so listings with the same timestamp never straddle it. Train and test assignments are saved in `data/processed/splits.parquet`, and every later phase reads that file instead of re-splitting.
- **Spatial CV:** KMeans on the unique map points, projected to UTM meters and weighted by listings per point, with 40 clusters. GroupKFold deals the clusters into 5 folds. KMeans uses only coordinates, never rent, so it is fit on all points and test listings get a cluster label for per-region reporting later.
- **Why 40 clusters, not 20:** with 20 clusters, one downtown cluster held 19% of training listings and became a fold by itself. Fold MAE then ranged from $236 to $503, and the CV mean was really "can you predict downtown from the suburbs?". With 40 clusters the folds are equal-sized (20% each) and each spans city and suburbs. The fold SD halved. Clusters are about 3 km in radius, which is larger than the ~1 km precision of the listing coordinates. The choice was made on balance (largest-cluster share, fold sizes), not on which setting gave the lowest MAE. Shuffled fold assignment was also tried: it produced unequal folds (15-28%) and no better balance, so it was rejected.
- **Main baseline:** median rent by ZIP x bedrooms, falling back to the bedroom median and then the global median. Two simpler baselines (global median, bedroom median) give context.
- **Evaluation:** MAE is the headline metric (in dollars, readable), with MAPE and median absolute error alongside. Test MAE gets a 95% bootstrap CI (2,000 resamples) so the Phase 4 "20% below baseline" claim can be checked against noise.
- **Which CV score counts:** spatial CV is the score for all tuning and model selection from Phase 4 on. Random CV is reported only to show the leak ($227 vs $314 for the ZIP baseline).
- **MLflow:** tracking points at the local `mlflow.db` (SQLite), so runs log without a server. `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001` reads the same file. Experiment: `rent-model`.

## Phase 4: Rent model

- **Target:** both models predict log(rent) and convert back with exp. Rent is right-skewed and errors scale with price, so log makes the problem closer to symmetric. exp of a log prediction is a median-type estimate, which suits MAE.
- **Features (46, in 6 groups):** size, location (coordinates, distances, county one-hots, `coarse_location`), neighborhood (tract and ZIP ACS, HUD SAFMR), market (ZORI with metro fallback plus a missing flag, 30-year mortgage rate), listing (12 amenity flags, amenity count, pets, photo level) and nearby rent.
- **Nearby rent instead of raw rent per sq ft:** rent per sq ft is computed from the target, so it can't be a feature directly. `NeighborRentPerSqft` takes a listing-weighted average of median log rent per sq ft at the k nearest *training* map points. During training it skips the listing's own point, so a listing never sees its own rent or same-point near-copies. At prediction time it uses all training points, including the same point when one exists. It sits inside the sklearn pipeline, so CV refits it on each training fold. k is tuned (Ridge best 12, XGBoost best 8).
- **Missing values:** Ridge uses median imputation plus missing-indicator columns, fitted inside the pipeline. XGBoost handles NaN natively.
- **Search:** Ridge grid of alpha (1 to 30,000) x k. The first grid stopped at 1,000, which was also the best value, so the grid was widened until the optimum sat inside it. XGBoost: 60 Optuna TPE trials (seed 42) over objective (absolute or squared error), trees, learning rate, depth, child weight, subsampling and L1/L2 penalties. Every trial is a nested MLflow run.
- **Selection rule:** lowest spatial CV MAE. The test set was evaluated once, after the choice. Both models' test scores are reported, but only the CV winner is registered.
- **Significance of the gain:** a paired bootstrap resamples the same test listings for model and baseline, giving a CI on the % MAE reduction. XGBoost: 30.1% (25.3% to 34.7%).
- **Regions:** counties (Fulton, DeKalb, Cobb, Gwinnett, Clayton, Other). They are readable for a report, unlike the 40 CV clusters. Price bands: < $1,000, $1,000-1,299, $1,300-1,599, $1,600+, based on actual rent.
- **Model registry:** the winner is registered as `rent-model` with skops serialization, the MLflow 3 default. Only the five classes the pipeline needs are listed as trusted, instead of falling back to pickle.
- **Environment:** XGBoost on macOS needs OpenMP, which is installed into the conda env (`conda install -c conda-forge llvm-openmp`) rather than system-wide.

## Phase 5: Text feature (PyTorch)

- **Text instead of photos:** the photo links are from 2019, and labeling about 1K photos by hand is costly. Fine-tuning a pretrained text model on listing descriptions keeps the deep learning part (PyTorch fine-tuning) without manual labels.
- **Number masking:** every number in the title and description is replaced with `[num]`. 15% of descriptions contain the exact rent.
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (22M params) with mean pooling and a linear head, predicting log(rent per sq ft). The settings were fixed up front, not tuned: 3 epochs, encoder lr 3e-5, head lr 1e-3, Huber loss, 160 tokens. About 85 s per fit on Apple MPS.
- **Stacking without leakage:** each training listing's score comes from a model fit on the other 4 spatial folds. Test scores come from a model fit on all training data. Scores are cached in `data/processed/text_scores.parquet`, and `--retrain` refits them.
- **Process isolation:** PyTorch and XGBoost each bundle their own OpenMP runtime, and loading both in one process segfaults on macOS. `src/text_scores.py` (PyTorch) runs as a subprocess, and `src/phase5.py` (XGBoost) never imports torch.
- **Outcome:** the text model was not adopted. CV MAE improved $233 → $207, but test MAE did not ($155 → $157, CI -4.8% to +2.2%). The likely reason is that text stands in for location knowledge, which spatial CV removes but the test set mostly has. The CV figure also carries a small second-order stacking leak. Full reasoning: `reports/phase5_findings.md` §5.

## Phase 6: Ranges, drivers, comps

- **Conformal method:** MAPIE `CrossConformalRegressor` (CV+) on log(rent), so ranges are multiplicative and widen with price. Point predictions come from the model refit on all training data.
- **Location-aware calibration:** calibrating only on spatial folds over-covered the test set (87%, width $621), because it prepares for new neighborhoods while 88% of test listings are at known points. `analyze()` uses random-fold calibration for known map points and spatial-fold calibration for new ones. The rule follows from the deployment setting, not from tuning on test. Test coverage: 78.9% (target 80%), median width $475.
- **Per-county (Mondrian) calibration was tried and rejected:** it moved Fulton from 65% to 72% but dropped Gwinnett to 70% and overall to 76%. Fulton's shortfall comes from the late-2019 in-town price shift (a break in exchangeability), not from interval width. This is a documented limitation for Phase 10 drift monitoring. The code stays in `RentAnalyzer` (`mode="mondrian"`) for comparison.
- **SHAP:** TreeExplainer on the XGBoost step, applied to the pipeline-transformed features. Values are on the log scale, so they are reported as % effects (exp(phi) - 1). A test checks that the SHAP values add up to the prediction.
- **Comps:** 5 nearest training listings in a weighted space: km east/north (weight 1), log sq ft (4), bedrooms (1.5), bathrooms (1). Location dominates, and size breaks ties.
- **Artifact:** the fitted `RentAnalyzer` is saved to `models/rent_analyzer.joblib` (git-ignored) for the Phase 9 API.

## Phase 7: Rent-growth forecast

- **Target:** 12-month log growth of ZORI per ZIP, forecast quarterly from Dec 2021 to Dec 2023 (forecast windows cover 2022-2024). Only ZIPs with 36+ months of *unbroken* history are scored. Each forecast sees data up to its origin only, and CPI rent is lagged a month for its release delay.
- **SARIMA:** one (1,1,1)x(0,1,1,12) model per ZIP on log rent, fixed up front. Stationarity and invertibility are enforced: without them, 13 of 856 fits exploded.
- **XGBoost:** one global model on lag features. It trains only on rows whose 12-month target was already observed at the origin. A test checks this by poisoning later targets.
- **Shipped = XGBoost, after comparing three options on two tests** (2022-24 MAE / share of ZIPs inside the 80% band in the 2019 backtest): no-change 3.09 pp / 2%, XGBoost 5.41 / 44%, auto-select 6.98 / 44%. No-change wins only in flat markets. Auto-select (`method="auto"`: best of no-change, persistence and XGBoost on the prior 2 years) lagged the 2022 turn. XGBoost is the only option that holds up in both regimes, and auto-select picks it in Dec 2018 and today anyway.
- **Later years:** an AR(1) fit to annual Atlanta CPI rent (complete years since 1998, up to `as_of`), so the long-run mean is not a ZORI boom-era number. Year-1 uncertainty comes from the forecaster's own errors in the 2 years before `as_of`.

## Phase 8: Monte Carlo deal simulation

- **Annual steps, vectorized over paths:** all drivers are drawn as (paths x years) arrays first, then one cash-flow function runs on them. The same function is reused with actual 2019-2024 drivers for the backtest.
- **Rent level:** a lognormal fitted so its 10th/90th percentiles equal the conformal 80% range. The 2019 rent prediction is rolled to the deal date with the ZIP's ZORI.
- **Vacancy and inflation:** AR(1) on 30 years of annual US rental vacancy and US CPI (added `CPIAUCSL` to the FRED pull). FRED has no Atlanta vacancy series.
- **Rates:** the loan is fixed-rate, so the rate path (a random walk with annual sd from FRED) only affects the exit: exit cap = entry cap + 0.5 x rate change.
- **FHA:** 3.5% down, 1.75% upfront MIP financed, 0.55% annual MIP on the balance for the life of the loan, the owner occupies one unit in year 1, and the self-sufficiency test (75% of all units' rent >= PITI, 3-4 units). **Conventional:** 25% down, +0.75 pp investor rate.
- **Prices are assumptions** ($165k/unit today, $120k/unit in 2019): there is no licensed sales dataset.
- **IRR:** vectorized bisection on NPV (NaN if no sign change), tested against SciPy `brentq`.

