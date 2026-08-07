import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold

from src.config import PROJECTED_CRS, RANDOM_SEED

TEST_FRAC = 0.2
N_CLUSTERS = 40
N_FOLDS = 5
POINT = ["latitude", "longitude"]


def time_holdout(df: pd.DataFrame, frac: float = TEST_FRAC, time_col: str = "listed_at") -> pd.Series:
    times = df[time_col]
    candidates = np.sort(times.unique())[1:]
    shares = np.array([(times >= t).mean() for t in candidates])
    cutoff = candidates[np.argmin(np.abs(shares - frac))]
    return (times >= cutoff).rename("is_test")


def point_xy(points: pd.DataFrame) -> np.ndarray:
    geom = gpd.points_from_xy(points["longitude"], points["latitude"], crs=4326).to_crs(PROJECTED_CRS)
    return np.column_stack([geom.x, geom.y]) / 1000


def spatial_clusters(df: pd.DataFrame, n_clusters: int = N_CLUSTERS, seed: int = RANDOM_SEED) -> pd.Series:
    points = df.groupby(POINT).size().rename("listings").reset_index()
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    points["cluster"] = km.fit_predict(point_xy(points), sample_weight=points["listings"])
    return df[POINT].merge(points, on=POINT, how="left")["cluster"].set_axis(df.index)


def spatial_folds(clusters: pd.Series, n_folds: int = N_FOLDS) -> pd.Series:
    folds = pd.Series(-1, index=clusters.index, name="fold")
    for fold, (_, val) in enumerate(GroupKFold(n_splits=n_folds).split(clusters, groups=clusters)):
        folds.iloc[val] = fold
    return folds


def random_folds(index: pd.Index, n_folds: int = N_FOLDS, seed: int = RANDOM_SEED) -> pd.Series:
    folds = pd.Series(-1, index=index, name="fold")
    for fold, (_, val) in enumerate(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(index)):
        folds.iloc[val] = fold
    return folds


def leaked_groups(groups: pd.Series, folds: pd.Series) -> set:
    leaks = set()
    for fold in folds.unique():
        val = folds == fold
        leaks |= set(groups[val]) & set(groups[~val])
    return leaks


def make_splits(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["id"]].copy()
    out["is_test"] = time_holdout(df)
    out["cluster"] = spatial_clusters(df)
    train = ~out["is_test"]
    out["fold"] = -1
    out.loc[train, "fold"] = spatial_folds(out.loc[train, "cluster"])
    return out
