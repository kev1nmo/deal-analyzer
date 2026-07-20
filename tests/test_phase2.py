import io
import zipfile

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src import hypothesis
from src.clean import clean_listings, fix_locations
from src.features import build_features
from src.sources import geo, pums
from src.spatial import add_spatial_features


def listing(id, lat=33.80, lon=-84.39, city="Atlanta", beds=2, sqft=1000, rent=1400, day=1):
    return {
        "id": id, "listed_at": pd.Timestamp("2019-02-01") + pd.to_timedelta(day, unit="D"), "rent": float(rent),
        "bedrooms": float(beds), "bathrooms": 1.0, "square_feet": sqft, "cityname": city,
        "latitude": lat, "longitude": lon,
    }


def test_clean_listings_rules():
    rows = [listing(i, rent=1400 + i) for i in range(20)]
    rows += [
        listing(100, rent=1400, day=0),
        listing(101, rent=1400, day=5),
        listing(102, beds=5, sqft=2500),
        listing(103, sqft=200),
        listing(104, beds=3, sqft=450),
        listing(105, rent=9000),
    ]
    df, log = clean_listings(pd.DataFrame(rows))
    assert 100 not in df["id"].values
    assert 101 in df["id"].values
    assert not set(df["id"]) & {102, 103, 104, 105}
    assert log["rows"].is_monotonic_decreasing
    assert log["rows"].iloc[0] == 26
    assert df["listed_at"].is_monotonic_increasing


def test_fix_locations_moves_and_flags():
    rows = [listing(i, lat=33.78, lon=-84.39, city="Atlanta") for i in range(10)]
    rows += [listing(10 + i, lat=33.44, lon=-84.15, city="Mcdonough") for i in range(5)]
    rows += [listing(20, lat=33.78, lon=-84.39, city="Mcdonough"), listing(21, lat=33.78, lon=-84.39, city="Smyrna")]
    rows += [listing(22, lat=33.78, lon=-84.39, city="Duluth")]
    df = fix_locations(pd.DataFrame(rows)).set_index("id")
    assert df.loc[20, ["latitude", "longitude"]].tolist() == [33.44, -84.15]
    assert df.loc[20, "location_moved"]
    assert df.loc[21, "location_unfixable"]
    assert df.loc[0, "coarse_location"]
    assert not df.loc[10, "coarse_location"]


def gtfs_zip(tmp_path):
    files = {
        "routes.txt": "route_id,route_type\nR1,1\nB1,3\n",
        "trips.txt": "route_id,trip_id\nR1,t1\nB1,t2\n",
        "stop_times.txt": "trip_id,stop_id,arrival_time\nt1,s1,08:00:00\nt1,s2,08:05:00\nt1,s3,08:09:00\nt2,s9,08:00:00\n",
        "stops.txt": (
            "stop_id,stop_name,stop_lat,stop_lon,parent_station\n"
            "p1,FIVE POINTS STATION,33.7539,-84.3916,\n"
            "s1,FIVE POINTS STATION NB,33.7540,-84.3917,p1\n"
            "s2,FIVE POINTS STATION SB,33.7538,-84.3915,p1\n"
            "s3,NORTH AVE STATION,33.7716,-84.3869,\n"
            "s9,PEACHTREE ST @ 10TH ST,33.781,-84.385,\n"
        ),
    }
    path = tmp_path / "gtfs.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, text in files.items():
            z.writestr(f"feed/{name}", text)
    return path


def test_rail_stations_keeps_rail_only(tmp_path):
    stations = geo.rail_stations(geo.read_gtfs(gtfs_zip(tmp_path)))
    assert stations["station"].tolist() == ["FIVE POINTS STATION", "NORTH AVE STATION"]
    assert stations.loc[0, "lat"] == pytest.approx(33.7539)


def test_read_shapes_filters_to_bbox(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"ZCTA5CE20": ["30303", "31401"]},
        geometry=[box(-84.40, 33.74, -84.38, 33.76), box(-81.10, 32.07, -81.08, 32.09)],
        crs=4326,
    )
    folder = tmp_path / "shp"
    folder.mkdir()
    gdf.to_file(folder / "zcta.shp")
    zip_path = tmp_path / "zcta.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in folder.iterdir():
            z.write(f, f.name)
    shapes = geo.zcta_shapes(geo.read_shapes(zip_path))
    assert shapes["zcta"].tolist() == ["30303"]


def test_spatial_features():
    zcta = gpd.GeoDataFrame({"zcta": ["30303", "30309"]},
                            geometry=[box(-84.42, 33.73, -84.38, 33.77), box(-84.40, 33.77, -84.36, 33.81)], crs=4326)
    tracts = gpd.GeoDataFrame({"geoid": ["13121000100", "13089000200"]},
                              geometry=[box(-84.42, 33.73, -84.38, 33.77), box(-84.40, 33.77, -84.36, 33.81)], crs=4326)
    stations = pd.DataFrame({"station": ["FIVE POINTS", "ARTS CENTER"], "lat": [33.7539, 33.7893], "lon": [-84.3916, -84.3871]})
    acs = pd.DataFrame({"geoid": ["13121000100", "13089000200"], "median_hh_income": [40000, 90000],
                        "median_gross_rent": [1000, 1600], "renter_share": [0.7, 0.6], "vacancy_rate": [0.1, 0.05]})
    listings = pd.DataFrame([listing(1, lat=33.7539, lon=-84.3916), listing(2, lat=33.79, lon=-84.387),
                             listing(3, lat=33.83, lon=-84.38)])
    out = add_spatial_features(listings, zcta, tracts, stations, acs).set_index("id")
    assert out.loc[1, "zip"] == "30303"
    assert out.loc[2, "zip"] == "30309"
    assert out.loc[3, "zip"] == "30309"
    assert out.loc[1, "km_to_marta"] == pytest.approx(0, abs=0.01)
    assert out.loc[2, "nearest_marta_station"] == "ARTS CENTER"
    assert out.loc[1, "km_to_downtown"] == pytest.approx(0.02, abs=0.05)
    assert out.loc[2, "tract_median_income"] == 90000
    assert out.loc[1, "county_fips"] == "121"
    assert len(out) == 3


def test_build_features_joins_context():
    con = duckdb.connect()
    con.execute("CREATE TABLE acs_zcta AS SELECT '30303' AS zcta, 50000 AS median_hh_income, 1200 AS median_gross_rent, 0.7 AS renter_share, 0.1 AS vacancy_rate")
    con.execute("CREATE TABLE hud_safmr AS SELECT '30303' AS zip, 2 AS bedrooms, 1600.0 AS safmr")
    con.execute("CREATE TABLE zillow_zori AS SELECT * FROM (VALUES ('30303', DATE '2019-02-01', 1500.0, 0.04), ('30309', DATE '2019-02-01', 1700.0, 0.06)) t(zip, month, zori, zori_yoy)")
    con.execute("CREATE TABLE mortgage_weekly AS SELECT DATE '2019-01-31' AS date, 4.46 AS mortgage_30y")
    listings = pd.DataFrame([{**listing(1), "zip": "30303"}, {**listing(2), "zip": "30318"}])
    listings["bedrooms"] = listings["bedrooms"].astype(int)
    df = build_features(con, listings).set_index("id")
    assert df.loc[1, "hud_safmr"] == 1600.0
    assert df.loc[1, "zori_at_listing"] == 1500.0
    assert pd.isna(df.loc[2, "zori_at_listing"])
    assert df.loc[2, "zori_metro_at_listing"] == 1600.0
    assert df.loc[2, "mortgage_30y_at_listing"] == 4.46
    assert len(df) == 2


def test_pums_tidy_and_puma_filter():
    names = pd.DataFrame({
        "NAME": ["Atlanta Regional Commission--Fulton County (Central)--Atlanta City (Southwest) PUMA, Georgia",
                 "Coastal Regional Commission--Chatham County PUMA, Georgia"],
        "public use microdata area": ["01406", "04000"],
        "state": ["13", "13"],
    })
    pumas = pums.atlanta_pumas(names)
    assert pumas["puma"].tolist() == ["01406"]
    raw = pd.DataFrame({
        "SERIALNO": ["a", "a", "b", "c", "d"], "SPORDER": ["1", "2", "1", "1", "1"], "TEN": ["3", "3", "3", "1", "3"],
        "RNTP": ["1200", "1200", "900", "0", "1000"], "GRNTP": ["1350", "1350", "1000", "0", "1100"],
        "BLD": ["05", "05", "09", "02", "05"], "BDSP": ["2"] * 5, "RMSP": ["4"] * 5, "YRBLT": ["1980"] * 5,
        "WGTP": ["20"] * 5, "public use microdata area": ["01406", "01406", "01406", "01406", "04000"], "state": ["13"] * 5,
    })
    df = pums.tidy(raw, pumas, 2024)
    assert df["serialno"].tolist() == ["a", "b"]
    assert df["bld"].tolist() == [5, 9]
    assert (df["year"] == 2024).all()


def simulated_pums(n=6000, effect=-0.08, seed=0):
    rng = np.random.default_rng(seed)
    bld = rng.choice([2, 4, 5, 6, 9], size=n, p=[0.25, 0.1, 0.15, 0.2, 0.3])
    beds = rng.integers(0, 5, size=n)
    rooms = beds + rng.integers(1, 4, size=n)
    puma = rng.choice(["01401", "01402", "01403", "01404"], size=n)
    year_built = rng.choice([1939, 1965, 1985, 2005, 2015, 2021], size=n)
    year = rng.choice([2022, 2023, 2024], size=n)
    log_rent = (
        7.0 + 0.12 * beds + 0.03 * rooms + 0.05 * (puma == "01402") + 0.03 * (year - 2022)
        + effect * (bld == 5) - 0.02 * (bld == 4) + 0.05 * (bld == 2) + rng.normal(0, 0.15, size=n)
    )
    rent = np.round(np.exp(log_rent))
    return pd.DataFrame({
        "serialno": np.arange(n).astype(str), "year": year, "puma": puma, "puma_name": "x",
        "rent": rent, "gross_rent": rent + 150, "bld": bld, "bedrooms": beds, "rooms": rooms,
        "year_built": year_built, "weight": rng.integers(5, 40, size=n),
    })


def test_hypothesis_recovers_known_effect():
    df = hypothesis.prepare(simulated_pums())
    effects = hypothesis.run_specs(df)
    main = effects[(effects["spec"] == next(iter(hypothesis.SPECS))) & (effects["building"] == "3-4 units")].iloc[0]
    assert main["ci_low_pct"] < np.expm1(-0.08) * 100 < main["ci_high_pct"]
    assert main["p_value"] < 0.001
    assert set(effects["building"]) == {"1 unit", "2 units", "3-4 units"}
    assert len(effects) == 3 * len(hypothesis.SPECS)
    assert "less than units in 5+ unit buildings" in hypothesis.headline(effects)


def test_hypothesis_no_effect_is_not_significant():
    df = hypothesis.prepare(simulated_pums(effect=0.0, seed=3))
    effects = hypothesis.run_specs(df)
    main = effects[(effects["spec"] == next(iter(hypothesis.SPECS))) & (effects["building"] == "3-4 units")].iloc[0]
    assert main["ci_low_pct"] < 0 < main["ci_high_pct"]


def test_prepare_drops_out_of_scope_buildings():
    raw = simulated_pums(n=50)
    raw.loc[0, "bld"] = 1
    raw.loc[1, "bld"] = 10
    raw.loc[2, "year_built"] = 1938
    df = hypothesis.prepare(raw)
    assert len(df) == 47
    assert set(hypothesis.decade_built(pd.Series([1939, 1945, 2010, 2021]))) == {"pre-1940", "1940", "2010", "2020+"}


def test_weighted_median():
    assert hypothesis.weighted_median(pd.Series([1, 2, 3]), pd.Series([1, 1, 10])) == 3
    assert hypothesis.weighted_median(pd.Series([5, 1, 3]), pd.Series([1, 1, 1])) == 3
