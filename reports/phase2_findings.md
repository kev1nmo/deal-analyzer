# Phase 2 findings

## 1. Cleaning

| step                                                      |   rows |
|:----------------------------------------------------------|-------:|
| raw listings                                              |   4299 |
| after removing exact reposts                              |   3860 |
| after removing > 4 bedrooms                               |   3848 |
| after keeping 250-3500 sq ft                              |   3845 |
| after removing units under 220 sq ft per bedroom          |   3813 |
| after removing rent/sq ft outliers (|robust z| > 3.5)     |   3803 |
| after dropping listings whose location could not be fixed |   3792 |

- Locations are approximate: 157 unique map points for 3,792 listings (neighborhood-level precision, not street addresses).
- 110 listings had a map point that belonged to a different city; they were moved to their own city's main point.
- 233 listings sit on a city-wide fallback point (`coarse_location`); their spatial features are city-level only.

## 2. Rent by bedrooms

|   bedrooms |   listings |   median_rent |   median_sqft |   median_rent_per_sqft |
|-----------:|-----------:|--------------:|--------------:|-----------------------:|
|          0 |          6 |       1199.50 |        487.50 |                   1.74 |
|          1 |       1506 |       1204.00 |        785.00 |                   1.54 |
|          2 |       1692 |       1350.00 |       1146.50 |                   1.19 |
|          3 |        517 |       1449.00 |       1429.00 |                   1.02 |
|          4 |         71 |       1664.00 |       2200.00 |                   0.77 |

## 3. Feature coverage (share of listings with a value)

| feature             |   share_filled |   missing |
|:--------------------|---------------:|----------:|
| address             |          0.032 |      3671 |
| pets_allowed        |          0.461 |      2042 |
| zori_yoy_at_listing |          0.645 |      1345 |
| zori_at_listing     |          0.687 |      1188 |
| amenities           |          0.833 |       633 |
| zip_median_income   |          0.908 |       348 |
| tract_median_income |          0.939 |       231 |
| tract_median_rent   |          0.950 |       191 |
| zip_vacancy_rate    |          0.970 |       113 |
| zip_renter_share    |          0.970 |       113 |
| zip_median_rent     |          0.970 |       113 |
| tract_vacancy_rate  |          0.985 |        56 |
| tract_renter_share  |          0.985 |        56 |

Every column not listed is 100% filled.

Timing caveat: listings run from Dec 2018 to Dec 2019, while HUD fair rents are current fiscal-year values and Census figures cover 2020-2024. Treat those as relative neighborhood signals, not rent levels at listing time. ZORI and mortgage rates are matched to the listing month.

## 4. Hypothesis test: do 3-4 unit buildings rent at a per-unit discount?

The listings data has no building size (all are tagged 'apartment'), so the test uses Census ACS PUMS microdata: 11,700 renter households in Fulton, DeKalb, Cobb, Gwinnett and Clayton PUMAs, survey years 2022-2024.

Model: log(monthly rent) ~ building size + bedrooms + rooms + decade built + PUMA + survey year, weighted by household weight, heteroskedasticity-robust (HC1) standard errors. Reference group: 5+ unit buildings.

**Result:** Holding bedrooms, rooms, building age, location (PUMA) and survey year fixed, units in 3-4 unit buildings rent for 15.3% less than units in 5+ unit buildings (difference -15.3%, 95% CI -21.2% to -9.0%, p = 6.71e-06); this is statistically significant at the 5% level.

| building   |   households_sampled |   weighted_median_rent |   median_bedrooms |
|:-----------|---------------------:|-----------------------:|------------------:|
| 1 unit     |                 3712 |                   1600 |                 3 |
| 2 units    |                  318 |                   1200 |                 2 |
| 3-4 units  |                  803 |                   1300 |                 2 |
| 5+ units   |                 6867 |                   1500 |                 2 |

Effects for every building size and robustness check (percent difference vs 5+ unit buildings):

| spec                           | building   |   pct_vs_5plus |   ci_low_pct |   ci_high_pct |   p_value |     n |
|:-------------------------------|:-----------|---------------:|-------------:|--------------:|----------:|------:|
| Main: weighted, robust SE      | 1 unit     |          1.316 |       -2.570 |         5.357 |     0.512 | 11700 |
| Main: weighted, robust SE      | 2 units    |        -17.515 |      -25.594 |        -8.560 |     0.000 | 11700 |
| Main: weighted, robust SE      | 3-4 units  |        -15.332 |      -21.248 |        -8.970 |     0.000 | 11700 |
| Unweighted, robust SE          | 1 unit     |          2.168 |       -1.254 |         5.709 |     0.217 | 11700 |
| Unweighted, robust SE          | 2 units    |        -13.992 |      -20.367 |        -7.107 |     0.000 | 11700 |
| Unweighted, robust SE          | 3-4 units  |        -12.209 |      -16.316 |        -7.900 |     0.000 | 11700 |
| Weighted, SE clustered by PUMA | 1 unit     |          1.316 |       -4.363 |         7.332 |     0.657 | 11700 |
| Weighted, SE clustered by PUMA | 2 units    |        -17.515 |      -26.947 |        -6.865 |     0.002 | 11700 |
| Weighted, SE clustered by PUMA | 3-4 units  |        -15.332 |      -21.675 |        -8.474 |     0.000 | 11700 |
| Gross rent (incl. utilities)   | 1 unit     |          5.616 |        2.316 |         9.022 |     0.001 | 11700 |
| Gross rent (incl. utilities)   | 2 units    |        -11.137 |      -17.135 |        -4.704 |     0.001 | 11700 |
| Gross rent (incl. utilities)   | 3-4 units  |        -10.265 |      -14.627 |        -5.680 |     0.000 | 11700 |

Caveats: PUMS has no square footage (rooms and bedrooms stand in for size), PUMAs are large areas (about 100K+ people), and rents are what current tenants pay, not new asking rents.
