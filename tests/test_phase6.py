import numpy as np
import pandas as pd
import pytest

from src.explain import CONFIDENCE, LEVELS, RentAnalyzer
from src.model_features import add_base_features
from tests.test_phase4 import raw_listings


@pytest.fixture(scope="module")
def fitted():
    df = raw_listings(n=400).assign(cityname="Atlanta", listed_at=pd.Timestamp("2019-03-01"))
    df = add_base_features(df)
    return RentAnalyzer({"n_estimators": 40, "max_depth": 3}, k=3).fit(df), df


def test_intervals_are_ordered_and_nested(fitted):
    analyzer, df = fitted
    for mode in ["spatial", "random", "location_aware", "mondrian"]:
        bounds = analyzer.predict_interval(df.head(20), mode)
        assert bounds.shape == (20, 2, len(LEVELS))
        assert (bounds[:, 0, :] <= bounds[:, 1, :]).all()
        widths = bounds[:, 1, :] - bounds[:, 0, :]
        assert (np.diff(widths, axis=1) >= -1e-6).all()


def test_location_aware_picks_scheme_by_known_point(fitted):
    analyzer, df = fitted
    known = df.head(5)
    new = known.assign(latitude=known["latitude"] + 0.5)
    assert analyzer.known_location(known).all()
    assert not analyzer.known_location(new).any()
    assert np.allclose(analyzer.predict_interval(known), analyzer.predict_interval(known, "random"))
    assert np.allclose(analyzer.predict_interval(new), analyzer.predict_interval(new, "spatial"))


def test_analyze_returns_everything(fitted):
    analyzer, df = fitted
    result = analyzer.analyze(df.iloc[[0]])
    lo, hi = result["interval"]
    assert lo < result["predicted_rent"] < hi
    assert result["confidence"] == CONFIDENCE
    assert len(result["drivers"]) == 5
    assert {"feature", "value", "effect_pct"} <= set(result["drivers"][0])
    assert len(result["comps"]) == 5
    assert result["comps"]["km_away"].iloc[0] == pytest.approx(0.0)


def test_shap_values_add_up_to_prediction(fitted):
    analyzer, df = fitted
    values = analyzer.shap_values(df.head(10))
    base = analyzer.explainer_.expected_value
    log_pred = np.log(analyzer.predict(df.head(10)))
    assert np.allclose(values.sum(axis=1) + base, log_pred, atol=1e-3)
