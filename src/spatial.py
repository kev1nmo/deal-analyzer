import geopandas as gpd
import pandas as pd

from src.config import DOWNTOWN_LATLON, PROJECTED_CRS

TRACT_COLUMNS = {
    "median_hh_income": "tract_median_income",
    "median_gross_rent": "tract_median_rent",
    "renter_share": "tract_renter_share",
    "vacancy_rate": "tract_vacancy_rate",
}


def to_points(lat: pd.Series, lon: pd.Series, index=None) -> gpd.GeoDataFrame:
    geometry = gpd.points_from_xy(lon, lat)
    return gpd.GeoDataFrame(index=index, geometry=geometry, crs=4326).to_crs(PROJECTED_CRS)


def assign_polygon(points: gpd.GeoDataFrame, shapes: gpd.GeoDataFrame, key: str) -> pd.Series:
    shapes = shapes[[key, "geometry"]].to_crs(PROJECTED_CRS)
    inside = gpd.sjoin(points, shapes, how="left", predicate="within")
    result = inside[~inside.index.duplicated()][key].reindex(points.index)
    missing = result.isna()
    if missing.any():
        near = gpd.sjoin_nearest(points[missing], shapes, how="left")
        result[missing] = near[~near.index.duplicated()][key].reindex(result[missing].index)
    return result


def nearest(points: gpd.GeoDataFrame, targets: gpd.GeoDataFrame, key: str) -> pd.DataFrame:
    near = gpd.sjoin_nearest(points, targets[[key, "geometry"]], how="left", distance_col="meters")
    near = near[~near.index.duplicated()].reindex(points.index)
    return pd.DataFrame({key: near[key], "km": near["meters"] / 1000}, index=points.index)


def add_spatial_features(
    listings: pd.DataFrame,
    zcta_shapes: gpd.GeoDataFrame,
    tract_shapes: gpd.GeoDataFrame,
    stations: pd.DataFrame,
    acs_tract: pd.DataFrame,
) -> pd.DataFrame:
    out = listings.reset_index(drop=True).copy()
    points = to_points(out["latitude"], out["longitude"], out.index)
    out["zip"] = assign_polygon(points, zcta_shapes, "zcta")
    out["tract"] = assign_polygon(points, tract_shapes, "geoid")
    out["county_fips"] = out["tract"].str[2:5]

    downtown = to_points(pd.Series([DOWNTOWN_LATLON[0]]), pd.Series([DOWNTOWN_LATLON[1]])).geometry.iloc[0]
    out["km_to_downtown"] = points.distance(downtown) / 1000

    station_points = to_points(stations["lat"], stations["lon"]).assign(station=stations["station"].values)
    near = nearest(points, station_points, "station")
    out["nearest_marta_station"] = near["station"]
    out["km_to_marta"] = near["km"]

    tract_stats = acs_tract[["geoid", *TRACT_COLUMNS]].rename(columns={"geoid": "tract", **TRACT_COLUMNS})
    return out.merge(tract_stats, on="tract", how="left")
