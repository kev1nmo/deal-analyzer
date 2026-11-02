import threading
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

from src.build_db import MORTGAGE
from src.config import DATA_RAW
from src.features import build_features
from src.model_features import FEATURES, NEIGHBOR, add_base_features
from src.spatial import add_spatial_features

MODEL_INPUT = [f for f in FEATURES if f not in NEIGHBOR]
LOOKUP_TABLES = ["acs_zcta", "hud_safmr", "zillow_zori", "fred"]
MODEL_DATE = pd.Timestamp("2019-12-01")


class FeatureBuilder:
    def __init__(self, raw_dir: Path = DATA_RAW):
        self.zcta = gpd.read_parquet(raw_dir / "geo" / "zcta_shapes.parquet")
        self.tracts = gpd.read_parquet(raw_dir / "geo" / "tract_shapes.parquet")
        self.stations = pd.read_parquet(raw_dir / "marta_stations.parquet")
        self.acs_tract = pd.read_parquet(raw_dir / "acs_tract.parquet")
        self.con = duckdb.connect()
        for table in LOOKUP_TABLES:
            path = (raw_dir / f"{table}.parquet").as_posix()
            self.con.execute(f"CREATE TABLE {table} AS SELECT * FROM read_parquet('{path}')")
        self.con.execute(MORTGAGE)
        zori = pd.read_parquet(raw_dir / "zillow_zori.parquet")
        self.zori = zori.pivot_table(index="month", columns="zip", values="zori")
        self.zori_metro = zori.groupby("month")["zori"].median()
        self.latest_month = pd.Timestamp(self.zori.index.max())
        self.lock = threading.Lock()

    def build(self, listings: pd.DataFrame) -> pd.DataFrame:
        spatial = add_spatial_features(listings, self.zcta, self.tracts, self.stations, self.acs_tract)
        with self.lock:
            joined = build_features(self.con, spatial)
        joined = joined.set_index("id").loc[listings["id"].to_numpy()].reset_index()
        return add_base_features(joined)

    def rent_ratio(self, zip_code: str, start: pd.Timestamp, end: pd.Timestamp | None = None) -> float:
        start = start.to_period("M").to_timestamp()
        end = self.latest_month if end is None else end.to_period("M").to_timestamp()
        if zip_code in self.zori.columns:
            a, b = self.zori.at[start, zip_code], self.zori.at[end, zip_code]
            if np.isfinite(a) and np.isfinite(b):
                return float(b / a)
        return float(self.zori_metro.at[end] / self.zori_metro.at[start])
