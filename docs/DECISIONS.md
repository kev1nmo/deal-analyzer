# Decisions

## Phase 1: Data

- **Listings source:** UCI "Apartments for Rent Classified" (100K, CC BY 4.0), filtered to the Atlanta area. That leaves 4,299 listings (Dec 2018 to Dec 2019), below the ~20K target. No licensed API offered free Atlanta listings with history, and scraping is out of scope.
- **Database:** a single DuckDB file (`data/deals.duckdb`), built from Parquet files in `data/raw/`.

