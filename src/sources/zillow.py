import pandas as pd

from src.sources.common import download, save

URL = "https://files.zillowstatic.com/research/public_csvs/zori/Zip_zori_uc_sfrcondomfr_sm_month.csv"
ID_COLS = ["RegionName", "City", "Metro", "CountyName"]


def tidy(raw: pd.DataFrame, metro_prefix: str = "Atlanta-Sandy Springs") -> pd.DataFrame:
    df = raw[(raw["State"] == "GA") & raw["Metro"].fillna("").str.startswith(metro_prefix)]
    date_cols = [c for c in df.columns if c[:2] in ("19", "20")]
    long = df[ID_COLS + date_cols].melt(id_vars=ID_COLS, var_name="month", value_name="zori")
    long = long.rename(columns={"RegionName": "zip", "City": "city", "Metro": "metro", "CountyName": "county"})
    long["zip"] = long["zip"].astype(str).str.zfill(5)
    long["month"] = pd.to_datetime(long["month"]).dt.to_period("M").dt.to_timestamp()
    long = long.dropna(subset=["zori"]).sort_values(["zip", "month"]).reset_index(drop=True)
    long["zori_yoy"] = long.groupby("zip")["zori"].pct_change(12, fill_method=None)
    return long


def run() -> pd.DataFrame:
    raw = pd.read_csv(download(URL, "zori_zip.csv", force=True), dtype={"RegionName": str})
    df = tidy(raw)
    save(df, "zillow_zori")
    print(f"  {df['zip'].nunique()} ZIPs, {df['month'].min():%Y-%m} to {df['month'].max():%Y-%m}")
    return df
