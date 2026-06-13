import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.config import DATA_RAW, MARKET_BBOX, STATE_FIPS
from src.sources.common import download, download_first, save

GEO_DIR = DATA_RAW / "geo"
TIGER = "https://www2.census.gov/geo/tiger"
ZCTA_URL = f"{TIGER}/GENZ2020/shp/cb_2020_us_zcta520_500k.zip"
TRACT_URL = f"{TIGER}/GENZ2024/shp/cb_2024_{STATE_FIPS}_tract_500k.zip"
MARTA_URLS = [
    ("https://www.itsmarta.com/google_transit_feed/google_transit.zip", "marta_gtfs.zip"),
    (
        "https://files.mobilitydatabase.org/mdb-368/mdb-368-202604190110/mdb-368-202604190110.zip",
        "marta_gtfs_mdb.zip",
    ),
]
RAIL_ROUTE_TYPE = 1


def read_shapes(zip_path: Path, bbox=MARKET_BBOX) -> gpd.GeoDataFrame:
    return gpd.read_file(f"zip://{zip_path}", bbox=bbox).to_crs(4326)


def zcta_shapes(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = raw.rename(columns={"ZCTA5CE20": "zcta"})[["zcta", "geometry"]]
    return gdf.sort_values("zcta").reset_index(drop=True)


def tract_shapes(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = raw.rename(columns={"GEOID": "geoid"})[["geoid", "geometry"]]
    return gdf.sort_values("geoid").reset_index(drop=True)


def read_gtfs(zip_path: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    with zipfile.ZipFile(zip_path) as z:
        names = {Path(n).name: n for n in z.namelist()}
        for table, cols in {
            "routes": ["route_id", "route_type"],
            "trips": ["route_id", "trip_id"],
            "stop_times": ["trip_id", "stop_id"],
            "stops": None,
        }.items():
            with z.open(names[f"{table}.txt"]) as f:
                tables[table] = pd.read_csv(f, usecols=cols, dtype=str)
    return tables


def rail_stations(gtfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    routes = gtfs["routes"]
    rail_routes = routes.loc[pd.to_numeric(routes["route_type"]) == RAIL_ROUTE_TYPE, "route_id"]
    rail_trips = gtfs["trips"].loc[gtfs["trips"]["route_id"].isin(rail_routes), "trip_id"]
    rail_stop_ids = gtfs["stop_times"].loc[gtfs["stop_times"]["trip_id"].isin(rail_trips), "stop_id"].unique()
    stops = gtfs["stops"].copy()
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"])
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"])
    rail = stops[stops["stop_id"].isin(rail_stop_ids)].copy()
    if "parent_station" in rail and rail["parent_station"].notna().any():
        parents = stops.set_index("stop_id")
        has_parent = rail["parent_station"].isin(parents.index)
        rail.loc[has_parent, "stop_name"] = parents.loc[rail.loc[has_parent, "parent_station"], "stop_name"].values
    rail["station"] = (
        rail["stop_name"].str.upper()
        .str.replace(r"\s+(NB|SB|EB|WB)\b.*$", "", regex=True)
        .str.replace(r"\s+RAIL\b", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    out = rail.groupby("station", as_index=False).agg(lat=("stop_lat", "mean"), lon=("stop_lon", "mean"))
    return out.sort_values("station").reset_index(drop=True)


def run_shapes() -> pd.DataFrame:
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    zcta = zcta_shapes(read_shapes(download(ZCTA_URL, "cb_2020_us_zcta520_500k.zip")))
    zcta.to_parquet(GEO_DIR / "zcta_shapes.parquet")
    tracts = tract_shapes(read_shapes(download(TRACT_URL, f"cb_2024_{STATE_FIPS}_tract_500k.zip")))
    tracts.to_parquet(GEO_DIR / "tract_shapes.parquet")
    print(f"  {len(zcta)} ZCTA shapes, {len(tracts)} tract shapes")
    return tracts


def run_marta() -> pd.DataFrame:
    path, url = download_first(MARTA_URLS)
    df = rail_stations(read_gtfs(path))
    save(df, "marta_stations")
    print(f"  source: {url}")
    print(f"  {len(df)} rail stations")
    return df
