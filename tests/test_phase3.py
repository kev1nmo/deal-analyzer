import numpy as np
import pandas as pd
import pytest

from src.baseline import MedianBaseline, bootstrap_mae_ci, cross_validate, metrics
from src.config import DATA_PROCESSED
from src.validation import leaked_groups, make_splits, random_folds, spatial_clusters, spatial_folds, time_holdout


def synthetic_listings(n_points=60, per_point=10, seed=0):
    rng = np.random.default_rng(seed)
    lats = 33.75 + rng.uniform(-0.3, 0.3, n_points)
    lons = -84.39 + rng.uniform(-0.3, 0.3, n_points)
    rows = []
    for p in range(n_points):
        for j in range(per_point):
            rows.append({
                "id": p * per_point + j,
                "latitude": round(lats[p], 4),
                "longitude": round(lons[p], 4),
                "listed_at": pd.Timestamp("2019-01-01") + pd.Timedelta(days=int(rng.integers(0, 365))),
                "zip": f"30{p % 12:03d}",
                "bedrooms": int(rng.integers(1, 4)),
                "rent": float(rng.normal(1400, 300)),
            })
    return pd.DataFrame(rows)


def test_time_holdout_takes_newest_fifth():
    df = synthetic_listings()
    is_test = time_holdout(df)
    assert df.loc[is_test, "listed_at"].min() >= df.loc[~is_test, "listed_at"].max()
    assert 0.18 <= is_test.mean() <= 0.25


def test_time_holdout_keeps_ties_together():
    df = pd.DataFrame({"listed_at": pd.to_datetime(["2019-01-01"] * 6 + ["2019-02-01"] * 4)})
    is_test = time_holdout(df, frac=0.5)
    assert is_test.sum() == 4
    df = pd.DataFrame({"listed_at": pd.to_datetime(["2019-01-01"] * 4 + ["2019-02-01"] * 6)})
    assert time_holdout(df, frac=0.2).sum() == 6


def test_same_point_same_cluster():
    df = synthetic_listings()
    clusters = spatial_clusters(df, n_clusters=10)
    assert clusters.notna().all()
    assert (df.assign(c=clusters).groupby(["latitude", "longitude"])["c"].nunique() == 1).all()


def test_no_cluster_in_both_train_and_validation():
    df = synthetic_listings()
    clusters = spatial_clusters(df, n_clusters=15)
    folds = spatial_folds(clusters, n_folds=5)
    assert set(folds) == set(range(5))
    for fold in range(5):
        val = folds == fold
        assert not set(clusters[val]) & set(clusters[~val])
    assert leaked_groups(clusters, folds) == set()
    points = df["latitude"].astype(str) + "," + df["longitude"].astype(str)
    assert leaked_groups(points, folds) == set()


def test_random_folds_do_leak_points():
    df = synthetic_listings()
    points = df["latitude"].astype(str) + "," + df["longitude"].astype(str)
    assert leaked_groups(points, random_folds(df.index)) != set()


def test_make_splits_test_rows_have_no_fold():
    splits = make_splits(synthetic_listings())
    assert (splits.loc[splits["is_test"], "fold"] == -1).all()
    assert (splits.loc[~splits["is_test"], "fold"] >= 0).all()
    train = splits[~splits["is_test"]]
    assert leaked_groups(train["cluster"], train["fold"]) == set()


@pytest.mark.skipif(not (DATA_PROCESSED / "splits.parquet").exists(), reason="run python -m src.phase3 first")
def test_saved_splits_have_no_cluster_leak():
    splits = pd.read_parquet(DATA_PROCESSED / "splits.parquet")
    train = splits[~splits["is_test"]]
    assert leaked_groups(train["cluster"], train["fold"]) == set()
    assert splits["id"].is_unique


def test_median_baseline_lookup_and_fallbacks():
    train = pd.DataFrame({
        "zip": ["A", "A", "A", "B", "B"],
        "bedrooms": [1, 1, 2, 1, 2],
        "rent": [1000.0, 1200.0, 1500.0, 900.0, 1300.0],
    })
    model = MedianBaseline(["zip", "bedrooms"], fallbacks=[["bedrooms"]]).fit(train)
    new = pd.DataFrame({"zip": ["A", "C", "C"], "bedrooms": [1, 2, 5]})
    out = model.predict_with_level(new)
    assert out["pred"].tolist() == [1100.0, 1400.0, 1200.0]
    assert out["level"].tolist() == [0, 1, 2]


def test_median_baseline_min_count():
    train = pd.DataFrame({"zip": ["A", "B", "B"], "bedrooms": [1, 1, 1], "rent": [2000.0, 1000.0, 1100.0]})
    model = MedianBaseline(["zip", "bedrooms"], fallbacks=[["bedrooms"]], min_count=2).fit(train)
    assert model.predict(pd.DataFrame({"zip": ["A"], "bedrooms": [1]}))[0] == 1100.0


def test_metrics_values():
    m = metrics(np.array([100.0, 200.0]), np.array([110.0, 150.0]))
    assert m["mae"] == pytest.approx(30.0)
    assert m["mape"] == pytest.approx(17.5)
    assert m["median_ae"] == pytest.approx(30.0)


def test_bootstrap_ci_brackets_mae():
    rng = np.random.default_rng(1)
    y = rng.normal(1400, 300, 500)
    pred = y + rng.normal(0, 200, 500)
    lo, hi = bootstrap_mae_ci(y, pred)
    assert lo < metrics(y, pred)["mae"] < hi


def test_cross_validate_one_row_per_fold():
    df = synthetic_listings()
    folds = spatial_folds(spatial_clusters(df, n_clusters=10))
    res = cross_validate(MedianBaseline(["bedrooms"]), df, folds)
    assert len(res) == 5
    assert res["n"].sum() == len(df)
