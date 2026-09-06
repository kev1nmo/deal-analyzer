import numpy as np
import pandas as pd
import pytest

from src.model_features import FEATURES, NeighborRentPerSqft, add_base_features
from src.models import build_ridge, build_xgb, cv_predict, cv_score, paired_bootstrap_gain


def raw_listings(n=240, seed=0):
    rng = np.random.default_rng(seed)
    point = rng.integers(0, 30, n)
    lat = 33.6 + (point % 6) * 0.06
    lon = -84.6 + (point // 6) * 0.08
    sqft = rng.integers(500, 1800, n)
    km = np.hypot((lat - 33.75) * 110.6, (lon + 84.39) * 92.5)
    rent = (1.9 - 0.02 * km) * sqft * rng.lognormal(0, 0.08, n)
    return pd.DataFrame({
        "id": np.arange(n), "rent": rent, "square_feet": sqft,
        "bedrooms": np.clip(sqft // 500, 0, 4), "bathrooms": 1.0 + (sqft > 1200),
        "latitude": lat, "longitude": lon, "km_to_downtown": km, "km_to_marta": km / 2,
        "coarse_location": False, "county_fips": np.where(point % 2 == 0, "121", "067"),
        "tract_median_income": rng.normal(80000, 20000, n), "tract_median_rent": rng.normal(1300, 200, n),
        "tract_renter_share": rng.uniform(0.2, 0.8, n), "tract_vacancy_rate": rng.uniform(0, 0.1, n),
        "zip_median_income": np.where(point % 5 == 0, np.nan, 75000.0), "zip_median_rent": 1300.0,
        "zip_renter_share": 0.5, "zip_vacancy_rate": 0.05, "hud_safmr": 1500,
        "zori_at_listing": np.where(point % 3 == 0, np.nan, 1500.0), "zori_metro_at_listing": 1450.0,
        "zori_yoy_at_listing": np.where(point % 3 == 0, np.nan, 0.04), "zori_yoy_metro_at_listing": 0.03,
        "mortgage_30y_at_listing": 4.2, "amenities": np.where(point % 4 == 0, "Pool,Gym", None),
        "pets_allowed": np.where(point % 2 == 0, "Cats,Dogs", "None"), "has_photo": "Yes",
        "fold": point % 5,
    })


def test_base_features_fill_and_flags():
    df = add_base_features(raw_listings())
    assert df["zori"].notna().all()
    assert (df.loc[df["zori_missing"] == 1, "zori"] == 1450.0).all()
    county_cols = [c for c in df if c.startswith("county_") and c != "county_fips"]
    assert (df[county_cols].sum(axis=1) == 1).all()
    assert df.loc[df["amenities"] == "Pool,Gym", ["has_pool", "has_gym", "n_amenities"]].eq([1, 1, 2]).all().all()
    assert df.loc[df["amenities"].isna(), "n_amenities"].eq(0).all()
    assert set(df["pets_allowed_flag"]) == {0, 1}


def neighbor_frame():
    return pd.DataFrame({
        "latitude": [33.70, 33.70, 33.80, 33.90],
        "longitude": [-84.40, -84.40, -84.40, -84.40],
        "square_feet": [1000, 1000, 1000, 1000],
    })


def test_neighbor_feature_ignores_own_point_when_training():
    X = neighbor_frame()
    y = np.log([1000.0, 1000.0, 2000.0, 3000.0])
    first = NeighborRentPerSqft(k=1).fit_transform(X, y)["nbr_log_rent_psf"]
    y_changed = np.log([9000.0, 9000.0, 2000.0, 3000.0])
    second = NeighborRentPerSqft(k=1).fit_transform(X, y_changed)["nbr_log_rent_psf"]
    assert first.iloc[0] == pytest.approx(np.log(2.0))
    assert first.iloc[0] == second.iloc[0]


def test_neighbor_feature_uses_own_point_when_predicting():
    X = neighbor_frame()
    y = np.log([1000.0, 1000.0, 2000.0, 3000.0])
    out = NeighborRentPerSqft(k=1).fit(X, y).transform(X.head(1))
    assert out["nbr_log_rent_psf"].iloc[0] == pytest.approx(np.log(1.0))
    assert out["nbr_km"].iloc[0] == pytest.approx(0.0)


@pytest.mark.parametrize("build", [lambda: build_ridge(10.0, 3), lambda: build_xgb({"n_estimators": 50}, 3)])
def test_models_fit_and_predict_dollars(build):
    df = add_base_features(raw_listings())
    model = build().fit(df, df["rent"])
    pred = model.predict(df)
    assert pred.shape == (len(df),)
    assert (pred > 0).all()
    assert np.abs(pred - df["rent"]).mean() < df["rent"].std()


def test_models_need_only_raw_feature_columns():
    df = add_base_features(raw_listings())
    inputs = [f for f in FEATURES if f not in ("nbr_log_rent_psf", "nbr_km")]
    model = build_ridge(10.0, 3).fit(df[inputs], df["rent"])
    assert len(model.predict(df[inputs].head(5))) == 5


def test_cv_predict_covers_every_row_once():
    df = add_base_features(raw_listings())
    pred = cv_predict(build_ridge(10.0, 3), df, df["rent"], df["fold"])
    assert not np.isnan(pred).any()
    score = cv_score(build_ridge(10.0, 3), df, df["rent"], df["fold"])
    assert {"cv_mae", "cv_mape", "cv_median_ae", "cv_mae_fold_sd"} <= set(score)


def test_paired_bootstrap_gain():
    rng = np.random.default_rng(0)
    y = rng.normal(1400, 300, 400)
    base = y + rng.normal(0, 200, 400)
    assert paired_bootstrap_gain(y, base, base)[0] == pytest.approx(0.0)
    gain, lo, hi = paired_bootstrap_gain(y, base, y)
    assert gain == pytest.approx(1.0)
    assert paired_bootstrap_gain(y, base, y + (base - y) / 2)[0] == pytest.approx(0.5)
    gain, lo, hi = paired_bootstrap_gain(y, base, y + rng.normal(0, 150, 400))
    assert lo < gain < hi
    assert 0 < lo
