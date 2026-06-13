import re

import pandas as pd

from src.config import HUD_AREA_CODE, SAFMR_YEARS
from src.sources.common import download_first, save

BASE = "https://www.huduser.gov/portal/datasets/fmr"


def candidates() -> list[tuple[str, str]]:
    return [(f"{BASE}/fmr{y}/fy{y}_safmrs.xlsx", f"fy{y}_safmrs.xlsx") for y in SAFMR_YEARS]


def to_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")


def parse(raw: pd.DataFrame, area_code: str = HUD_AREA_CODE, fiscal_year: int | None = None) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    zip_col, area_col, name_col = df.columns[:3]
    rent_cols = {}
    for col in df.columns:
        m = re.fullmatch(r"SAFMR (\d)BR", col)
        if m:
            rent_cols[int(m.group(1))] = col
    if len(rent_cols) < 5:
        rent_cols = dict(enumerate(df.columns[3::3][:5]))
    df = df[df[area_col].astype(str).str.strip() == area_code]
    out = df[[zip_col, area_col, name_col]].copy()
    out.columns = ["zip", "hud_area_code", "hud_area_name"]
    out["zip"] = out["zip"].astype(str).str.strip().str.zfill(5)
    for beds, col in rent_cols.items():
        out[beds] = to_number(df[col]).values
    long = out.melt(id_vars=["zip", "hud_area_code", "hud_area_name"], var_name="bedrooms", value_name="safmr")
    long["bedrooms"] = long["bedrooms"].astype(int)
    long["fiscal_year"] = fiscal_year
    return long.dropna(subset=["safmr"]).sort_values(["zip", "bedrooms"]).reset_index(drop=True)


def run() -> pd.DataFrame:
    path, url = download_first(candidates())
    fiscal_year = int(re.search(r"fy(\d{4})", url).group(1))
    df = parse(pd.read_excel(path, dtype=str), fiscal_year=fiscal_year)
    save(df, "hud_safmr")
    print(f"  source: {url}")
    return df
