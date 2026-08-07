from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED

TARGET = "rent"


@dataclass
class MedianBaseline:
    keys: list[str] = field(default_factory=list)
    fallbacks: list[list[str]] = field(default_factory=list)
    min_count: int = 1
    tables: list[pd.DataFrame] = field(default_factory=list, init=False)
    global_median: float = field(default=np.nan, init=False)

    def fit(self, df: pd.DataFrame) -> "MedianBaseline":
        self.global_median = float(df[TARGET].median())
        self.tables = []
        for keys in [self.keys, *self.fallbacks]:
            if keys:
                stats = df.groupby(keys)[TARGET].agg(["median", "size"])
                self.tables.append(stats.loc[stats["size"] >= self.min_count, "median"].rename("pred").reset_index())
        return self

    def predict_with_level(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({"pred": np.nan, "level": len(self.tables)}, index=df.index)
        for level, table in enumerate(self.tables):
            keys = [c for c in table.columns if c != "pred"]
            hit = df[keys].merge(table, on=keys, how="left")["pred"].to_numpy()
            fill = out["pred"].isna().to_numpy() & ~np.isnan(hit)
            out.loc[fill, "pred"] = hit[fill]
            out.loc[fill, "level"] = level
        out["pred"] = out["pred"].fillna(self.global_median)
        return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_with_level(df)["pred"].to_numpy()


def baselines() -> dict[str, MedianBaseline]:
    return {
        "global_median": MedianBaseline(),
        "beds_median": MedianBaseline(["bedrooms"]),
        "zip_beds_median": MedianBaseline(["zip", "bedrooms"], fallbacks=[["bedrooms"]]),
    }


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = np.abs(np.asarray(y) - np.asarray(pred))
    return {
        "mae": float(err.mean()),
        "mape": float((err / np.asarray(y)).mean() * 100),
        "median_ae": float(np.median(err)),
    }


def bootstrap_mae_ci(y: np.ndarray, pred: np.ndarray, n: int = 2000, level: float = 0.95,
                     seed: int = RANDOM_SEED) -> tuple[float, float]:
    err = np.abs(np.asarray(y) - np.asarray(pred))
    rng = np.random.default_rng(seed)
    maes = err[rng.integers(0, len(err), size=(n, len(err)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(np.percentile(maes, tail)), float(np.percentile(maes, 100 - tail))


def cross_validate(model: MedianBaseline, df: pd.DataFrame, folds: pd.Series) -> pd.DataFrame:
    rows = []
    for fold in sorted(folds.unique()):
        val = folds == fold
        pred = model.fit(df[~val]).predict(df[val])
        rows.append({"fold": fold, "n": int(val.sum()), **metrics(df.loc[val, TARGET], pred)})
    return pd.DataFrame(rows)
