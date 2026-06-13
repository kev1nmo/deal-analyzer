import zipfile
from pathlib import Path

import pandas as pd

from src.config import DOWNTOWN_LATLON, GAZETTEER_YEARS, MARKET_RADIUS_KM
from src.sources.common import download_first, haversine_km, save

BASE = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer"


def candidates() -> list[tuple[str, str]]:
    return [
        (f"{BASE}/{y}_Gazetteer/{y}_Gaz_zcta_national.zip", f"{y}_Gaz_zcta_national.zip")
        for y in GAZETTEER_YEARS
    ]


def read_gazetteer(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith(".txt"))
        with z.open(name) as f:
            return pd.read_csv(f, sep="\t", dtype={"GEOID": str})


def centroids(raw: pd.DataFrame, center=DOWNTOWN_LATLON, radius_km=MARKET_RADIUS_KM + 15) -> pd.DataFrame:
    df = raw.copy()
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"GEOID": "zcta", "INTPTLAT": "lat", "INTPTLONG": "lon", "ALAND_SQMI": "land_sqmi"})
    df["zcta"] = df["zcta"].str.zfill(5)
    df["km_to_downtown"] = haversine_km(center[0], center[1], df["lat"], df["lon"])
    df = df[df["km_to_downtown"] <= radius_km]
    return df[["zcta", "lat", "lon", "land_sqmi", "km_to_downtown"]].reset_index(drop=True)


def run() -> pd.DataFrame:
    path, url = download_first(candidates())
    df = centroids(read_gazetteer(path))
    save(df, "zcta_centroids")
    print(f"  source: {url}")
    return df
