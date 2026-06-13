import zipfile
from pathlib import Path

import pandas as pd
import py7zr

from src.config import DOWNLOADS, DOWNTOWN_LATLON, MARKET_RADIUS_KM
from src.sources.common import download, haversine_km, save

URL = "https://archive.ics.uci.edu/static/public/555/apartment+for+rent+classified.zip"
STEM = "apartments_for_rent_classified_100K"

KEEP = [
    "id", "listed_at", "rent", "bedrooms", "bathrooms", "square_feet",
    "category", "title", "body", "amenities", "has_photo", "pets_allowed",
    "address", "cityname", "state", "latitude", "longitude", "km_to_downtown", "source",
]


def extract_csv(zip_path: Path) -> Path:
    out = DOWNLOADS / "uci_apartments"
    found = list(out.rglob(f"{STEM}.csv")) if out.exists() else []
    if found:
        return found[0]
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    found = list(out.rglob(f"{STEM}.csv"))
    if not found:
        with py7zr.SevenZipFile(next(out.rglob(f"{STEM}.7z"))) as s:
            s.extractall(out)
        found = list(out.rglob(f"{STEM}.csv"))
    return found[0]


def read_raw(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, sep=";", encoding="cp1252", on_bad_lines="skip", low_memory=False)


def clean(raw: pd.DataFrame, center=DOWNTOWN_LATLON, radius_km=MARKET_RADIUS_KM) -> pd.DataFrame:
    df = raw.copy()
    for col in ["price", "square_feet", "bedrooms", "bathrooms", "latitude", "longitude", "time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["state"] == "GA") & (df["price_type"] == "Monthly") & (df["currency"] == "USD")]
    df = df.dropna(subset=["price", "latitude", "longitude", "time"])
    df["km_to_downtown"] = haversine_km(center[0], center[1], df["latitude"], df["longitude"])
    df = df[df["km_to_downtown"] <= radius_km]
    df["listed_at"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"price": "rent"})
    df["id"] = df["id"].astype("int64")
    return df.drop_duplicates("id")[KEEP].sort_values("listed_at").reset_index(drop=True)


def coverage(raw: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    return {
        "us_rows": len(raw),
        "ga_rows": int((raw["state"] == "GA").sum()),
        "atlanta_city_rows": int((raw["cityname"] == "Atlanta").sum()),
        "metro_rows_kept": len(clean_df),
    }


def run() -> pd.DataFrame:
    raw = read_raw(extract_csv(download(URL, "uci_apartments.zip")))
    df = clean(raw)
    save(df, "listings")
    print("  coverage:", coverage(raw, df))
    return df
