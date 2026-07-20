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

