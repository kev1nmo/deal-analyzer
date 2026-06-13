import pandas as pd
import requests

from src.config import ACS_YEARS, CENSUS_API_KEY, STATE_FIPS
from src.sources.common import get_json, save

VARIABLES = {
    "B01003_001E": "population",
    "B19013_001E": "median_hh_income",
    "B25064_001E": "median_gross_rent",
    "B25077_001E": "median_home_value",
    "B25002_001E": "housing_units",
    "B25002_003E": "vacant_units",
    "B25003_001E": "occupied_units",
    "B25003_003E": "renter_occupied_units",
}


def to_frame(rows: list[list[str]], year: int) -> pd.DataFrame:
    df = pd.DataFrame(rows[1:], columns=rows[0]).rename(columns=VARIABLES)
    for col in VARIABLES.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = pd.NA
    df["renter_share"] = df["renter_occupied_units"] / df["occupied_units"]
    df["vacancy_rate"] = df["vacant_units"] / df["housing_units"]
    df["acs_year"] = year
    return df


def fetch(year: int, geo: dict) -> pd.DataFrame:
    params = {"get": ",".join(["NAME", *VARIABLES]), **geo, "key": CENSUS_API_KEY}
    return to_frame(get_json(f"https://api.census.gov/data/{year}/acs/acs5", params), year)


def fetch_latest(geo: dict) -> pd.DataFrame:
    for year in ACS_YEARS:
        try:
            return fetch(year, geo)
        except (requests.RequestException, ValueError):
            continue
    raise RuntimeError(f"ACS request failed for years {ACS_YEARS}")


def zcta() -> pd.DataFrame:
    df = fetch_latest({"for": "zip code tabulation area:*"})
    df = df.rename(columns={"zip code tabulation area": "zcta"})
    df = df[df["zcta"].str[:2].isin(["30", "31", "39"])]
    return df.drop(columns=["NAME"]).reset_index(drop=True)


def tracts() -> pd.DataFrame:
    geo = {"for": "tract:*", "in": f"state:{STATE_FIPS}"}
    df = fetch_latest(geo)
    df["geoid"] = df["state"] + df["county"] + df["tract"]
    return df.reset_index(drop=True)


def run() -> pd.DataFrame:
    z = zcta()
    save(z, "acs_zcta")
    t = tracts()
    save(t, "acs_tract")
    print(f"  ACS {z['acs_year'].iat[0]}: {len(z)} ZCTAs, {len(t)} tracts")
    return z
