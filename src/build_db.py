from pathlib import Path

import duckdb

from src.config import DATA_RAW

TABLES = ["listings", "hud_safmr", "acs_zcta", "acs_tract", "zillow_zori", "fred", "zcta_centroids"]

HAVERSINE = """
CREATE OR REPLACE MACRO haversine_km(lat1, lon1, lat2, lon2) AS
    2 * 6371 * asin(sqrt(
        pow(sin(radians(lat2 - lat1) / 2), 2)
        + cos(radians(lat1)) * cos(radians(lat2)) * pow(sin(radians(lon2 - lon1) / 2), 2)
    ))
"""

LISTING_ZIP = """
CREATE OR REPLACE TABLE listing_zip AS
SELECT
    l.id,
    arg_min(z.zcta, haversine_km(l.latitude, l.longitude, z.lat, z.lon)) AS zip,
    min(haversine_km(l.latitude, l.longitude, z.lat, z.lon)) AS km_to_zip_centroid
FROM listings l
CROSS JOIN zcta_centroids z
GROUP BY l.id
"""

MORTGAGE = """
CREATE OR REPLACE TABLE mortgage_weekly AS
SELECT date, value AS mortgage_30y
FROM fred
WHERE series_id = 'MORTGAGE30US'
ORDER BY date
"""

ENRICHED = """
CREATE OR REPLACE VIEW listings_enriched AS
SELECT
    l.*,
    lz.zip,
    lz.km_to_zip_centroid,
    a.median_hh_income AS zip_median_income,
    a.median_gross_rent AS zip_median_rent,
    a.renter_share AS zip_renter_share,
    a.vacancy_rate AS zip_vacancy_rate,
    h.safmr AS hud_safmr,
    zr.zori AS zori_at_listing,
    zr.zori_yoy AS zori_yoy_at_listing,
    zr.county,
    m.mortgage_30y AS mortgage_30y_at_listing
FROM listings l
LEFT JOIN listing_zip lz ON lz.id = l.id
LEFT JOIN acs_zcta a ON a.zcta = lz.zip
LEFT JOIN hud_safmr h ON h.zip = lz.zip AND h.bedrooms = least(CAST(l.bedrooms AS INTEGER), 4)
LEFT JOIN zillow_zori zr ON zr.zip = lz.zip AND zr.month = date_trunc('month', l.listed_at)
ASOF LEFT JOIN mortgage_weekly m ON l.listed_at >= m.date
"""

COVERAGE = """
SELECT
    count(*) AS listings,
    min(listed_at)::DATE AS first_listing,
    max(listed_at)::DATE AS last_listing,
    round(avg((zip IS NOT NULL)::INT), 3) AS has_zip,
    round(avg((zip_median_income IS NOT NULL)::INT), 3) AS has_acs,
    round(avg((hud_safmr IS NOT NULL)::INT), 3) AS has_safmr,
    round(avg((zori_at_listing IS NOT NULL)::INT), 3) AS has_zori,
    round(avg((mortgage_30y_at_listing IS NOT NULL)::INT), 3) AS has_mortgage,
    round(avg((square_feet IS NOT NULL)::INT), 3) AS has_sqft,
    round(avg((has_photo <> 'No')::INT), 3) AS has_photo
FROM listings_enriched
"""


def build(con: duckdb.DuckDBPyConnection, raw_dir: Path = DATA_RAW) -> None:
    missing = [t for t in TABLES if not (raw_dir / f"{t}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"Missing Parquet files in {raw_dir}: {missing}")
    for t in TABLES:
        con.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM read_parquet('{(raw_dir / f'{t}.parquet').as_posix()}')")
    for sql in [HAVERSINE, LISTING_ZIP, MORTGAGE, ENRICHED]:
        con.execute(sql)


def coverage(con: duckdb.DuckDBPyConnection):
    return con.sql(COVERAGE).df()
