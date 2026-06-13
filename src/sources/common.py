from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import DATA_RAW, DOWNLOADS

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; deal-analyzer/0.1)"}


def download(url: str, name: str, force: bool = False) -> Path:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    path = DOWNLOADS / name
    if path.exists() and not force:
        return path
    tmp = path.with_name(path.name + ".part")
    with requests.get(url, headers=HEADERS, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(path)
    return path


def download_first(candidates: list[tuple[str, str]]) -> tuple[Path, str]:
    errors = []
    for url, name in candidates:
        try:
            return download(url, name), url
        except requests.RequestException as e:
            errors.append(f"{url}: {e}")
    raise RuntimeError("All downloads failed:\n" + "\n".join(errors))


def get_json(url: str, params: dict) -> object:
    r = requests.get(url, params=params, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.json()


def save(df: pd.DataFrame, name: str, folder: Path = DATA_RAW) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))
