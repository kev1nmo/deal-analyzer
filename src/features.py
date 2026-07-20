import duckdb
import pandas as pd

FEATURES_SQL = """
CREATE OR REPLACE TABLE listings_features AS
WITH metro_zori AS (
    SELECT month, median(zori) AS zori_metro, median(zori_yoy) AS zori_yoy_metro
    FROM zillow_zori
    GROUP BY month
)
SELECT
    l.*,
    a.median_hh_income AS zip_median_income,
    a.median_gross_rent AS zip_median_rent,
    a.renter_share AS zip_renter_share,
    a.vacancy_rate AS zip_vacancy_rate,
    h.safmr AS hud_safmr,
    zr.zori AS zori_at_listing,
    zr.zori_yoy AS zori_yoy_at_listing,
    mz.zori_metro AS zori_metro_at_listing,
    mz.zori_yoy_metro AS zori_yoy_metro_at_listing,
    m.mortgage_30y AS mortgage_30y_at_listing
FROM listings_clean l
LEFT JOIN acs_zcta a ON a.zcta = l.zip
LEFT JOIN hud_safmr h ON h.zip = l.zip AND h.bedrooms = least(l.bedrooms, 4)
LEFT JOIN zillow_zori zr ON zr.zip = l.zip AND zr.month = date_trunc('month', l.listed_at)
LEFT JOIN metro_zori mz ON mz.month = date_trunc('month', l.listed_at)
ASOF LEFT JOIN mortgage_weekly m ON l.listed_at >= m.date
ORDER BY l.listed_at, l.id
"""


def build_features(con: duckdb.DuckDBPyConnection, listings: pd.DataFrame) -> pd.DataFrame:
    con.register("listings_clean_df", listings)
    con.execute("CREATE OR REPLACE TABLE listings_clean AS SELECT * FROM listings_clean_df")
    con.unregister("listings_clean_df")
    con.execute(FEATURES_SQL)
    df = con.sql("SELECT * FROM listings_features").df()
    if len(df) != len(listings) or not df["id"].is_unique:
        raise ValueError(f"Join changed the row count: {len(listings)} listings in, {len(df)} rows out")
    return df


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    share = df.notna().mean().rename("share_filled").to_frame()
    share["missing"] = df.isna().sum()
    return share.sort_values("share_filled")
