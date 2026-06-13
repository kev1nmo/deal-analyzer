import pandas as pd

from src.config import CENSUS_API_KEY, COUNTY_FIPS, PUMS_YEARS, STATE_FIPS
from src.sources.common import get_json, save

PUMA_GEO = "public use microdata area"
VARIABLES = ["SERIALNO", "SPORDER", "TEN", "RNTP", "GRNTP", "BLD", "BDSP", "RMSP", "YRBLT", "WGTP"]


def rows_to_frame(rows: list[list[str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df.loc[:, ~df.columns.duplicated()]


def atlanta_pumas(names: pd.DataFrame, counties=tuple(COUNTY_FIPS)) -> pd.DataFrame:
    pattern = "|".join(f"{c} County" for c in counties)
    keep = names["NAME"].str.contains(pattern, case=False, regex=True)
    return names.loc[keep, ["NAME", PUMA_GEO]].rename(columns={"NAME": "puma_name", PUMA_GEO: "puma"})


def tidy(raw: pd.DataFrame, pumas: pd.DataFrame, year: int) -> pd.DataFrame:
    df = raw.rename(columns={PUMA_GEO: "puma"})
    df = df[df["puma"].isin(pumas["puma"])].copy()
    for col in ["SPORDER", "TEN", "RNTP", "GRNTP", "BLD", "BDSP", "RMSP", "YRBLT", "WGTP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["SPORDER"] == 1) & (df["TEN"] == 3) & (df["RNTP"] > 0)]
    df = df.drop_duplicates("SERIALNO").merge(pumas, on="puma", how="left")
    df["year"] = year
    out = df.rename(columns={
        "SERIALNO": "serialno", "RNTP": "rent", "GRNTP": "gross_rent", "BLD": "bld",
        "BDSP": "bedrooms", "RMSP": "rooms", "YRBLT": "year_built", "WGTP": "weight",
    })
    return out[["serialno", "year", "puma", "puma_name", "rent", "gross_rent", "bld",
                "bedrooms", "rooms", "year_built", "weight"]].reset_index(drop=True)


def fetch_year(year: int) -> pd.DataFrame:
    base = f"https://api.census.gov/data/{year}/acs/acs1"
    geo = {"for": f"{PUMA_GEO}:*", "in": f"state:{STATE_FIPS}", "key": CENSUS_API_KEY}
    names = rows_to_frame(get_json(base, {"get": "NAME", **geo}))
    pumas = atlanta_pumas(names)
    params = {"get": ",".join(VARIABLES), "TEN": "3", "SPORDER": "1", **geo}
    raw = rows_to_frame(get_json(f"{base}/pums", params))
    return tidy(raw, pumas, year)


def run() -> pd.DataFrame:
    frames = [fetch_year(y) for y in PUMS_YEARS]
    df = pd.concat(frames, ignore_index=True)
    save(df, "pums_renters")
    print(f"  {len(df):,} renter households, {df['puma'].nunique()} PUMAs, years {PUMS_YEARS}")
    return df
