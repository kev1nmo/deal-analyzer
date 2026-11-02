from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.api.store import Store, predictions, rent_actuals
from src.config import ROOT
from src.monitoring import Thresholds, clear_source, drift_detected, monthly_drift
from src.retrain import decide, split_recent

ANALYZER = ROOT / "models" / "rent_analyzer.joblib"


def add_month(store: Store, month: str, n: int, error: float, covered: bool, source: str = "api") -> None:
    rng = np.random.default_rng(0)
    rent = rng.uniform(1000, 1500, n)
    pred = rent - error
    half = 300 if covered else 50
    observed = datetime.fromisoformat(f"{month}-15")
    ids = store.log_many(predictions, [{
        "model_version": "v1", "source": source, "zip": "30318", "request": {}, "features": {},
        "predicted_rent": float(p), "range_low": float(p - half), "range_high": float(p + half),
        "rent_today": float(p), "known_location": True, "latency_ms": 1.0} for p in pred])
    store.log_many(rent_actuals, [{
        "prediction_id": pid, "listing_id": None, "rent": float(r), "observed_at": observed,
        "observed_month": observed.date().replace(day=1)} for pid, r in zip(ids, rent)])


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'monitor.db'}")


def test_drift_query_flags_simulated_drift_only_in_the_bad_month(store):
    add_month(store, "2019-09", 50, error=100, covered=True)
    add_month(store, "2019-10", 50, error=400, covered=False)
    report = monthly_drift(store.engine, "api", Thresholds(reference_mae=155))
    assert report["n"].tolist() == [50, 50]
    assert report["mae"].tolist() == pytest.approx([100, 400])
    assert report["coverage"].tolist() == pytest.approx([1.0, 0.0])
    assert report["drift"].tolist() == [False, True]
    assert report["mae_drift"].tolist() == [0, 1] and report["coverage_drift"].tolist() == [0, 1]
    assert drift_detected(report)


def test_drift_query_ignores_small_months_and_other_sources(store):
    add_month(store, "2019-09", 50, error=50, covered=True)
    add_month(store, "2019-10", 10, error=900, covered=False)
    add_month(store, "2019-10", 50, error=900, covered=False, source="replay")
    report = monthly_drift(store.engine, "api", Thresholds(min_n=30))
    assert len(report) == 2 and not drift_detected(report)
    assert drift_detected(monthly_drift(store.engine, "replay"))


def test_clear_source_removes_only_that_source(store):
    add_month(store, "2019-09", 5, error=0, covered=True, source="api")
    add_month(store, "2019-09", 5, error=0, covered=True, source="replay")
    clear_source(store, "replay")
    assert monthly_drift(store.engine, "replay", Thresholds(min_n=1)).empty
    assert monthly_drift(store.engine, "api", Thresholds(min_n=1))["n"].tolist() == [5]


@pytest.mark.parametrize("champion, challenger, ci_low, expected", [
    (150, 140, 0.02, True),
    (150, 140, -0.01, False),
    (150, 160, 0.01, False),
    (150, 150, 0.0, False),
])
def test_promote_only_when_challenger_clearly_wins(champion, challenger, ci_low, expected):
    assert decide(champion, challenger, ci_low) is expected


def test_split_recent_holds_out_the_newest_rows():
    rows = pd.DataFrame({"id": range(10), "observed_at": pd.date_range("2019-09-01", periods=10)[::-1]})
    older, recent = split_recent(rows, 0.3)
    assert len(older) == 7 and len(recent) == 3
    assert older["observed_at"].max() < recent["observed_at"].min()


@pytest.mark.skipif(not ANALYZER.exists(), reason="needs models/rent_analyzer.joblib")
def test_actuals_endpoint_links_rent_to_prediction(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    unit = {"latitude": 33.7866, "longitude": -84.4149, "bedrooms": 2, "bathrooms": 1, "square_feet": 950}
    with TestClient(app) as c:
        pid = c.post("/predict", json=unit).json()["id"]
        r = c.post("/actuals", json={"prediction_id": pid, "rent": 1800, "observed_at": "2026-10-03T00:00:00"})
        assert r.status_code == 201
        assert c.post("/actuals", json={"prediction_id": pid, "rent": 1800}).status_code == 409
        assert c.post("/actuals", json={"prediction_id": 999, "rent": 1800}).status_code == 404
        report = c.get("/drift").json()
        assert report[0]["month"] == "2026-10-01" and report[0]["n"] == 1
