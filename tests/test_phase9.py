import numpy as np
import pandas as pd
import pytest

from src.config import DATA_PROCESSED, ROOT
from src.model_features import add_base_features

ANALYZER = ROOT / "models" / "rent_analyzer.joblib"
FEATURES_FILE = DATA_PROCESSED / "listings_features.parquet"
pytestmark = pytest.mark.skipif(not (ANALYZER.exists() and FEATURES_FILE.exists()),
                                reason="needs models/rent_analyzer.joblib and listings_features.parquet")
RAW_COLUMNS = ["id", "listed_at", "bedrooms", "bathrooms", "square_feet", "amenities", "has_photo", "pets_allowed",
               "latitude", "longitude", "coarse_location"]
UNIT = {"latitude": 33.7866, "longitude": -84.4149, "bedrooms": 2, "bathrooms": 1, "square_feet": 950,
        "amenities": ["Dishwasher"], "pets_allowed": True}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    from src.api.main import app

    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", f"sqlite:///{tmp_path_factory.mktemp('api') / 'api.db'}")
    with TestClient(app) as c:
        yield c
    mp.undo()


def test_serving_features_match_training_features():
    from src.api.features import MODEL_INPUT, FeatureBuilder

    stored = pd.read_parquet(FEATURES_FILE).sample(300, random_state=0).reset_index(drop=True)
    rebuilt = FeatureBuilder().build(stored[RAW_COLUMNS])
    expected = add_base_features(stored)
    pd.testing.assert_frame_equal(rebuilt[MODEL_INPUT].astype(float), expected[MODEL_INPUT].astype(float),
                                  check_exact=False, rtol=1e-6)


def test_rent_ratio_uses_zip_then_metro():
    from src.api.features import FeatureBuilder

    fb = FeatureBuilder()
    start = pd.Timestamp("2019-12-15")
    z = fb.zori["30318"]
    assert fb.rent_ratio("30318", start) == pytest.approx(z.iloc[-1] / z.loc["2019-12-01"])
    metro = fb.zori_metro
    assert fb.rent_ratio("99999", start) == pytest.approx(metro.iloc[-1] / metro.loc["2019-12-01"])


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["database"] == "sqlite" and len(body["model_version"]) == 12


def test_predict_returns_range_drivers_comps_and_logs(client):
    r = client.post("/predict", json=UNIT)
    assert r.status_code == 200
    body = r.json()
    assert body["zip"] == "30318" and body["county"] == "Fulton"
    assert body["range_low"] < body["predicted_rent"] < body["range_high"]
    assert body["rent_today"] > body["predicted_rent"]
    assert len(body["drivers"]) == 5 and len(body["comps"]) == 5
    logged = client.get("/predictions", params={"limit": 1}).json()[0]
    assert logged["id"] == body["id"] and logged["predicted_rent"] == pytest.approx(body["predicted_rent"])
    assert "features" not in logged


def test_predict_matches_analyzer_directly(client):
    import joblib

    from src.api.features import FeatureBuilder
    from src.api.main import to_listing
    from src.api.schemas import Unit

    X = FeatureBuilder().build(to_listing(Unit(**UNIT)))
    expected = joblib.load(ANALYZER).analyze(X)["predicted_rent"]
    assert client.post("/predict", json=UNIT).json()["predicted_rent"] == pytest.approx(expected)


@pytest.mark.parametrize("change", [{"latitude": 40.7}, {"square_feet": 300, "bedrooms": 2}, {"bedrooms": 6},
                                    {"amenities": ["Helipad"]}])
def test_predict_rejects_out_of_domain_input(client, change):
    assert client.post("/predict", json={**UNIT, **change}).status_code == 422


def test_simulate_is_reproducible_and_compares_two_loans(client):
    deal = {"unit": UNIT, "units": 4, "price": 660_000, "n_paths": 2_000}
    a, b = client.post("/simulate", json=deal).json(), client.post("/simulate", json=deal).json()
    assert [f["name"] for f in a["financing"]] == ["FHA house hack", "Conventional 25% down"]
    assert a["financing"][0]["metrics"]["irr"] == b["financing"][0]["metrics"]["irr"]
    fha, conv = a["financing"]
    assert fha["cash_invested"] < conv["cash_invested"]
    p_neg = [f["metrics"]["P(negative_cf_year1)"]["median"] for f in a["financing"]]
    assert all(0 <= p <= 1 for p in p_neg)
    assert np.isfinite(a["growth_year1"])


def test_breakeven_endpoint_returns_tornado_and_breakevens(client):
    body = client.post("/breakeven", json={"unit": UNIT, "units": 4, "price": 660_000}).json()
    assert [f["name"] for f in body["financing"]] == ["FHA house hack", "Conventional 25% down"]
    conv = body["financing"][1]
    assert len(conv["tornado"]) == 8 and len(conv["breakevens"]) == 6
    assert all(b["summary"] for b in conv["breakevens"])
