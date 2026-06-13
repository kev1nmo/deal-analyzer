import pandas as pd
import requests

from src.config import FRED_API_KEY, FRED_SERIES
from src.sources.common import get_json, save

URL = "https://api.stlouisfed.org/fred/series/observations"


def to_frame(series_id: str, payload: dict) -> pd.DataFrame:
    df = pd.DataFrame(payload["observations"])[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series_id"] = series_id
    df["name"] = FRED_SERIES.get(series_id, series_id)
    return df.dropna(subset=["value"])


def fetch(series_id: str) -> pd.DataFrame:
    params = {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json"}
    return to_frame(series_id, get_json(URL, params))


def run() -> pd.DataFrame:
    frames = []
    for series_id in FRED_SERIES:
        try:
            frames.append(fetch(series_id))
        except requests.RequestException as e:
            print(f"  skipped {series_id}: {e}")
    df = pd.concat(frames, ignore_index=True)[["series_id", "name", "date", "value"]]
    save(df, "fred")
    print("  " + ", ".join(f"{k}={v}" for k, v in df.groupby("series_id").size().items()))
    return df
