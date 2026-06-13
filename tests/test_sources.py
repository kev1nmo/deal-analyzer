import pandas as pd
import pytest

from src import build_db
from src.sources import census, fred, hud, listings, zcta, zillow
from src.sources.common import haversine_km


def raw_listing(**overrides):
    row = {
        "id": 1, "category": "housing/rent/apartment", "title": "Nice 2BR", "body": "Great unit",
        "amenities": "Pool", "bathrooms": "1", "bedrooms": "2", "currency": "USD", "fee": "No",
        "has_photo": "Yes", "pets_allowed": "Cats", "price": "1500", "price_display": "$1,500",
        "price_type": "Monthly", "square_feet": "900", "address": None, "cityname": "Atlanta",
        "state": "GA", "latitude": "33.78", "longitude": "-84.38", "source": "RentLingo",
        "time": "1568000000",
    }
    row.update(overrides)
    return row


def test_haversine_known_distance():
    assert haversine_km(33.7537, -84.3915, 33.6407, -84.4277) == pytest.approx(13.0, abs=0.5)


def test_listings_clean_filters_market():
    raw = pd.DataFrame([
        raw_listing(),
        raw_listing(id=2, state="TX", cityname="Austin", latitude="30.27", longitude="-97.74"),
        raw_listing(id=3, latitude="34.9", longitude="-85.3"),
        raw_listing(id=4, price_type="Weekly"),
        raw_listing(id=1),
    ])
    df = listings.clean(raw)
    assert df["id"].tolist() == [1]
    assert df["rent"].iat[0] == 1500
    assert df["km_to_downtown"].iat[0] < 5
    assert str(df["listed_at"].dt.year.iat[0]) == "2019"


def test_hud_parse_handles_multiline_headers():
    cols = ["ZIP\nCode", "HUD Area\nCode", "HUD Fair Market Rent\nArea Name"]
    for b in range(5):
        cols += [f"SAFMR\n{b}BR", f"SAFMR\n{b}BR - 90%\nPayment\nStandard", f"SAFMR\n{b}BR - 110%\nPayment\nStandard"]
    rows = [
        ["30303", "METRO12060M12060", "Atlanta"] + [f"${1000 + 100 * b:,}" if i == 0 else "0" for b in range(5) for i in range(3)],
        ["31401", "METRO42340M42340", "Savannah"] + ["900"] * 15,
    ]
    df = hud.parse(pd.DataFrame(rows, columns=cols), fiscal_year=2026)
    assert df["zip"].unique().tolist() == ["30303"]
    assert df.set_index("bedrooms")["safmr"].to_dict() == {0: 1000, 1: 1100, 2: 1200, 3: 1300, 4: 1400}


def test_census_to_frame_masks_sentinels():
    rows = [
        ["NAME", *census.VARIABLES, "zip code tabulation area"],
        ["ZCTA5 30303", "5000", "-666666666", "1200", "300000", "100", "10", "90", "60", "30303"],
    ]
    df = census.to_frame(rows, 2024)
    assert pd.isna(df["median_hh_income"].iat[0])
    assert df["renter_share"].iat[0] == pytest.approx(60 / 90)
    assert df["vacancy_rate"].iat[0] == pytest.approx(0.1)


def test_zillow_tidy_filters_and_computes_yoy():
    months = [d.strftime("%Y-%m-%d") for d in pd.date_range("2020-01-31", periods=13, freq="ME")]
    base = {"RegionID": 1, "SizeRank": 1, "RegionType": "zip", "StateName": "GA"}
    atl = {**base, "RegionName": "30303", "State": "GA", "City": "Atlanta",
           "Metro": "Atlanta-Sandy Springs-Roswell, GA", "CountyName": "Fulton County"}
    sav = {**base, "RegionName": "31401", "State": "GA", "City": "Savannah",
           "Metro": "Savannah, GA", "CountyName": "Chatham County"}
    raw = pd.DataFrame([
        {**atl, **{m: 1000 + i for i, m in enumerate(months)}},
        {**sav, **{m: 900 for m in months}},
    ])
    df = zillow.tidy(raw)
    assert df["zip"].unique().tolist() == ["30303"]
    assert df["month"].iat[0] == pd.Timestamp("2020-01-01")
    assert df["zori_yoy"].iat[-1] == pytest.approx(12 / 1000)


def test_fred_to_frame_drops_missing():
    payload = {"observations": [{"date": "2019-01-03", "value": "4.51"}, {"date": "2019-01-10", "value": "."}]}
    df = fred.to_frame("MORTGAGE30US", payload)
    assert len(df) == 1
    assert df["name"].iat[0] == "mortgage_30y"


def test_zcta_centroids_keeps_nearby():
    raw = pd.DataFrame({
        "GEOID": ["30303", "31401"], "ALAND_SQMI": [1.0, 2.0],
        "INTPTLAT": [33.75, 32.08], "INTPTLONG                                                                                                 ": [-84.39, -81.09],
    })
    df = zcta.centroids(raw)
    assert df["zcta"].tolist() == ["30303"]


def test_build_db_joins_everything(tmp_path):
    import duckdb

    raw = pd.DataFrame([raw_listing(), raw_listing(id=2, bedrooms="5", latitude="33.95", longitude="-84.55")])
    listings.clean(raw).to_parquet(tmp_path / "listings.parquet")
    pd.DataFrame({"zcta": ["30303", "30060"], "lat": [33.75, 33.95], "lon": [-84.39, -84.55],
                  "land_sqmi": [1.0, 5.0], "km_to_downtown": [0.5, 25.0]}).to_parquet(tmp_path / "zcta_centroids.parquet")
    pd.DataFrame({"zcta": ["30303"], "median_hh_income": [50000.0], "median_gross_rent": [1200.0],
                  "renter_share": [0.7], "vacancy_rate": [0.1]}).to_parquet(tmp_path / "acs_zcta.parquet")
    pd.DataFrame({"geoid": ["13121000100"]}).to_parquet(tmp_path / "acs_tract.parquet")
    pd.DataFrame({"zip": ["30303", "30060"], "bedrooms": [2, 4], "safmr": [1600.0, 2100.0]}).to_parquet(tmp_path / "hud_safmr.parquet")
    pd.DataFrame({"zip": ["30303"], "month": [pd.Timestamp("2019-09-01")], "zori": [1550.0],
                  "zori_yoy": [0.05], "county": ["Fulton County"]}).to_parquet(tmp_path / "zillow_zori.parquet")
    pd.DataFrame({"series_id": ["MORTGAGE30US"] * 2, "name": ["mortgage_30y"] * 2,
                  "date": pd.to_datetime(["2019-09-05", "2019-09-12"]), "value": [3.56, 3.56]}).to_parquet(tmp_path / "fred.parquet")

    con = duckdb.connect()
    build_db.build(con, tmp_path)
    df = con.sql("SELECT * FROM listings_enriched ORDER BY id").df()
    assert df["zip"].tolist() == ["30303", "30060"]
    assert df["hud_safmr"].tolist() == [1600.0, 2100.0]
    assert df["zori_at_listing"].iat[0] == 1550.0
    assert df["mortgage_30y_at_listing"].iat[0] == 3.56
    assert build_db.coverage(con)["listings"].iat[0] == 2
