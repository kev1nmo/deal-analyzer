import numpy as np
import pandas as pd

DUPLICATE_KEYS = ["latitude", "longitude", "bedrooms", "bathrooms", "square_feet", "rent"]
MAX_BEDROOMS = 4
SQFT_RANGE = (250, 3500)
MIN_SQFT_PER_BEDROOM = 220
ROBUST_Z_LIMIT = 3.5
COARSE_POINT_MIN_CITIES = 3


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("listed_at").drop_duplicates(DUPLICATE_KEYS, keep="last")


def robust_z(values: pd.Series, groups: pd.Series) -> pd.Series:
    med = values.groupby(groups).transform("median")
    mad = (values - med).abs().groupby(groups).transform("median")
    return 0.6745 * (values - med) / mad.replace(0, np.nan)


def fix_locations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    point = list(zip(df["latitude"], df["longitude"]))
    df["_point"] = point
    stats = df.groupby("_point")["cityname"].agg(
        point_city=lambda s: s.value_counts().index[0],
        point_cities="nunique",
    )
    df = df.join(stats, on="_point")
    df["listings_at_point"] = df.groupby("_point")["id"].transform("size")
    home = (
        df[df["cityname"] == df["point_city"]]
        .groupby(["cityname", "_point"]).size().rename("n").reset_index()
        .sort_values("n", ascending=False).drop_duplicates("cityname")
        .set_index("cityname")["_point"]
    )
    misplaced = df["cityname"] != df["point_city"]
    movable = misplaced & df["cityname"].isin(home.index)
    cities = df.loc[movable, "cityname"]
    df.loc[movable, "latitude"] = cities.map(home.map(lambda p: p[0])).astype(float)
    df.loc[movable, "longitude"] = cities.map(home.map(lambda p: p[1])).astype(float)
    df["location_moved"] = movable
    df["location_unfixable"] = misplaced & ~movable
    df["coarse_location"] = (df["point_cities"] >= COARSE_POINT_MIN_CITIES) & ~movable
    return df.drop(columns=["_point", "point_city", "point_cities"])


def clean_listings(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = [("raw listings", len(raw))]
    df = drop_duplicates(raw)
    log.append(("after removing exact reposts", len(df)))
    df = df[df["bedrooms"] <= MAX_BEDROOMS]
    log.append((f"after removing > {MAX_BEDROOMS} bedrooms", len(df)))
    df = df[df["square_feet"].between(*SQFT_RANGE)]
    log.append((f"after keeping {SQFT_RANGE[0]}-{SQFT_RANGE[1]} sq ft", len(df)))
    df = df[df["square_feet"] >= MIN_SQFT_PER_BEDROOM * df["bedrooms"]]
    log.append((f"after removing units under {MIN_SQFT_PER_BEDROOM} sq ft per bedroom", len(df)))
    df = df.assign(rent_per_sqft=df["rent"] / df["square_feet"])
    z = robust_z(np.log(df["rent_per_sqft"]), df["bedrooms"])
    df = df[z.abs() <= ROBUST_Z_LIMIT]
    log.append((f"after removing rent/sq ft outliers (|robust z| > {ROBUST_Z_LIMIT})", len(df)))
    df = fix_locations(df)
    df = df[~df["location_unfixable"]].drop(columns="location_unfixable")
    log.append(("after dropping listings whose location could not be fixed", len(df)))
    df["bedrooms"] = df["bedrooms"].astype(int)
    df = df.sort_values("listed_at").reset_index(drop=True)
    return df, pd.DataFrame(log, columns=["step", "rows"])
