import numpy as np
import pandas as pd

from src.forecast.rent_growth import (FEATURES, HORIZON, GrowthForecast, build_features, fit_xgb, forecast_origin,
                                      sarima_growth)


def toy_panel(months: int = 72, zips: int = 6) -> pd.DataFrame:
    idx = pd.date_range("2015-01-01", periods=months, freq="MS")
    rng = np.random.default_rng(0)
    rates = np.linspace(0.002, 0.006, zips)
    data = {f"300{i:02d}": np.log(1000) + rates[i] * np.arange(months) + rng.normal(0, 0.001, months)
            for i in range(zips)}
    return pd.DataFrame(data, index=idx)


def toy_macro(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"rate": 4.0, "rate_ch12": 0.0, "cpi_rent_g12": 0.03}, index=index)


def test_target_is_future_growth_and_history_resets_after_gap():
    logz = toy_panel()
    logz.iloc[30, 0] = np.nan
    f = build_features(logz, toy_macro(logz.index)).set_index(["month", "zip"])
    z = logz.columns[1]
    m = logz.index[10]
    assert np.isclose(f.loc[(m, z), "target"], logz[z].iloc[10 + HORIZON] - logz[z].iloc[10])
    assert f.loc[(logz.index[31], logz.columns[0]), "history"] == 1
    assert set(FEATURES) <= set(f.columns)


def test_xgb_trains_only_on_targets_known_at_origin():
    logz = toy_panel()
    f = build_features(logz, toy_macro(logz.index))
    origin = logz.index[48]
    leak = f.assign(target=np.where(f["month"] > origin - pd.DateOffset(months=HORIZON), 99.0, f["target"]))
    pred = fit_xgb(leak, origin).predict(f.loc[f["month"] == origin, FEATURES])
    assert np.all(pred < 1)


def test_sarima_recovers_steady_growth():
    logz = toy_panel()
    z = logz.columns[-1]
    assert abs(sarima_growth(logz[z]) - 0.006 * HORIZON) < 0.01


def test_forecast_origin_returns_each_method():
    logz = toy_panel()
    f = build_features(logz, toy_macro(logz.index))
    out = forecast_origin(logz, f, logz.index[50], list(logz.columns))
    assert out[["no_change", "persistence", "sarima", "xgboost"]].notna().all().all()


def test_growth_paths_revert_to_long_run():
    g = GrowthForecast("x", pd.Timestamp("2020-01-01"), 0.10, 0.0, 0.03, 0.5, 0.0)
    paths = g.paths(4, 3, np.random.default_rng(0))
    assert paths.shape == (3, 4)
    assert np.allclose(np.log1p(paths[0]), [0.10, 0.065, 0.0475, 0.03875])
