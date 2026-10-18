import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from src.config import DATA_RAW, RANDOM_SEED

HORIZON = 12
MIN_HISTORY = 36
LAGS = [1, 3, 6, 12, 24]
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL = (0, 1, 1, 12)
XGB_PARAMS = {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 3, "subsample": 0.8,
              "colsample_bytree": 0.8, "min_child_weight": 5}
FEATURES = [f"g{lag}" for lag in LAGS] + ["metro_g3", "metro_g12", "rel_level", "rate", "rate_ch12",
                                          "cpi_rent_g12", "month_of_year"]
METHODS = ["no_change", "persistence", "sarima", "xgboost"]
CANDIDATES = ["no_change", "persistence", "xgboost"]
WINNER = "xgboost"
LONG_RUN_YEARS = 30
FALLBACK_ERROR_SD = 0.03


def load_zori() -> pd.DataFrame:
    z = pd.read_parquet(DATA_RAW / "zillow_zori.parquet")
    panel = z.pivot_table(index="month", columns="zip", values="zori").asfreq("MS")
    return np.log(panel)


def load_fred() -> pd.DataFrame:
    return pd.read_parquet(DATA_RAW / "fred.parquet")


def monthly(fred: pd.DataFrame, name: str) -> pd.Series:
    s = fred.loc[fred["name"] == name].set_index("date")["value"]
    return s.resample("MS").mean()


def macro_frame(fred: pd.DataFrame) -> pd.DataFrame:
    rate = monthly(fred, "mortgage_30y")
    cpi_rent = np.log(monthly(fred, "atlanta_cpi_rent").interpolate(limit_area="inside")).shift(1)
    return pd.DataFrame({"rate": rate, "rate_ch12": rate - rate.shift(12),
                         "cpi_rent_g12": cpi_rent - cpi_rent.shift(12)})


def build_features(logz: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    frames = {f"g{lag}": logz - logz.shift(lag) for lag in LAGS}
    frames["rel_level"] = logz.sub(logz.median(axis=1), axis=0)
    observed = logz.notna()
    count = observed.cumsum()
    frames["history"] = (count - count.where(~observed).ffill().fillna(0)).where(observed)
    frames["target"] = logz.shift(-HORIZON) - logz
    long = pd.concat({k: v.stack(future_stack=True) for k, v in frames.items()}, axis=1)
    long.index.names = ["month", "zip"]
    long = long.reset_index()
    metro = pd.DataFrame({"metro_g3": frames["g3"].median(axis=1), "metro_g12": frames["g12"].median(axis=1)})
    long = long.join(metro, on="month").join(macro, on="month")
    long["month_of_year"] = long["month"].dt.month
    return long.dropna(subset=["history"])


def sarima_growth(series: pd.Series, horizon: int = HORIZON) -> float:
    values = series.dropna()
    months = values.index.year * 12 + values.index.month
    start = np.flatnonzero(np.diff(months, prepend=months[0] - 2) != 1)[-1]
    values = values.to_numpy()[start:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(values, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL,
                      enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)
    return float(fit.forecast(horizon)[-1] - values[-1])


def fit_xgb(features: pd.DataFrame, origin: pd.Timestamp) -> XGBRegressor:
    cutoff = origin - pd.DateOffset(months=HORIZON)
    train = features[(features["month"] <= cutoff) & features["target"].notna() & features["g12"].notna()]
    model = XGBRegressor(**XGB_PARAMS, random_state=RANDOM_SEED, n_jobs=4)
    return model.fit(train[FEATURES], train["target"])


def forecast_origin(logz: pd.DataFrame, features: pd.DataFrame, origin: pd.Timestamp,
                    zips: list[str], methods: list[str] = METHODS) -> pd.DataFrame:
    rows = features[(features["month"] == origin) & features["zip"].isin(zips)].set_index("zip").loc[zips]
    out = pd.DataFrame({"origin": origin, "zip": zips, "actual": rows["target"].to_numpy()})
    if "no_change" in methods:
        out["no_change"] = 0.0
    if "persistence" in methods:
        out["persistence"] = rows["g12"].to_numpy()
    if "sarima" in methods:
        out["sarima"] = [sarima_growth(logz.loc[:origin, z]) for z in zips]
    if "xgboost" in methods:
        out["xgboost"] = fit_xgb(features, origin).predict(rows[FEATURES])
    return out


def eligible_zips(features: pd.DataFrame, origin: pd.Timestamp, need_target: bool = True) -> list[str]:
    rows = features[(features["month"] == origin) & (features["history"] >= MIN_HISTORY) & features["g12"].notna()]
    if need_target:
        rows = rows[rows["target"].notna()]
    return sorted(rows["zip"])


def backtest(logz: pd.DataFrame, features: pd.DataFrame, origins: list[pd.Timestamp],
             methods: list[str] = METHODS) -> pd.DataFrame:
    parts = []
    for origin in origins:
        zips = eligible_zips(features, origin)
        if zips:
            parts.append(forecast_origin(logz, features, origin, zips, methods))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["actual", *methods])


def growth_mae(bt: pd.DataFrame, methods: list[str] = METHODS) -> pd.Series:
    actual = np.expm1(bt["actual"])
    return pd.Series({m: float(np.mean(np.abs(np.expm1(bt[m]) - actual))) for m in methods})


def annual_ar1(fred: pd.DataFrame, as_of: pd.Timestamp, years: int = LONG_RUN_YEARS) -> tuple[float, float, float]:
    s = fred.loc[(fred["name"] == "atlanta_cpi_rent") & (fred["date"] <= as_of)].set_index("date")["value"]
    yearly = s.groupby(s.index.year).agg(["mean", "size"])
    yearly = yearly.loc[yearly["size"] == 12, "mean"]
    g = np.log(yearly).diff()[yearly.index.to_series().diff() == 1]
    g = g[g.index < as_of.year].tail(years)
    x, y = g.to_numpy()[:-1], g.to_numpy()[1:]
    phi = float(np.clip(np.polyfit(x, y, 1)[0], 0.0, 0.95))
    mean = float(g.mean())
    resid = (y - mean) - phi * (x - mean)
    return mean, phi, float(resid.std(ddof=2))


@dataclass(frozen=True)
class GrowthForecast:
    zip: str
    as_of: pd.Timestamp
    year1: float
    year1_sd: float
    long_run: float
    phi: float
    sd: float
    method: str = ""

    def paths(self, years: int, n: int, rng: np.random.Generator) -> np.ndarray:
        g = np.empty((n, years))
        g[:, 0] = self.year1 + self.year1_sd * rng.standard_normal(n)
        for t in range(1, years):
            g[:, t] = self.long_run + self.phi * (g[:, t - 1] - self.long_run) + self.sd * rng.standard_normal(n)
        return np.expm1(g)


def recent_backtest(logz: pd.DataFrame, features: pd.DataFrame, as_of: pd.Timestamp,
                    methods: list[str]) -> pd.DataFrame:
    origins = pd.date_range(as_of - pd.DateOffset(months=HORIZON + 21), as_of - pd.DateOffset(months=HORIZON),
                            freq="3MS")
    return backtest(logz, features, list(origins), methods)


def choose_method(recent: pd.DataFrame, candidates: list[str]) -> str:
    if len(recent) < 30:
        return "no_change"
    return str(growth_mae(recent, candidates).idxmin())


def error_sd(recent: pd.DataFrame, method: str) -> float:
    if len(recent) < 30:
        return FALLBACK_ERROR_SD
    return float((recent["actual"] - recent[method]).astype(float).std())


def forecast_all(as_of: str | pd.Timestamp, method: str = WINNER, logz: pd.DataFrame | None = None,
                 fred: pd.DataFrame | None = None) -> dict[str, GrowthForecast]:
    as_of = pd.Timestamp(as_of).to_period("M").to_timestamp()
    logz = (load_zori() if logz is None else logz).loc[:as_of]
    fred = load_fred() if fred is None else fred
    fred = fred[fred["date"] <= as_of + pd.offsets.MonthEnd(0)]
    features = build_features(logz, macro_frame(fred))
    zips = eligible_zips(features, as_of, need_target=False)
    candidates = CANDIDATES if method == "auto" else [method]
    recent = recent_backtest(logz, features, as_of, candidates)
    chosen = choose_method(recent, candidates)
    pred = forecast_origin(logz, features, as_of, zips, [chosen])[chosen].to_numpy()
    year1_sd = error_sd(recent, chosen)
    long_run, phi, sd = annual_ar1(fred, as_of)
    out = {z: GrowthForecast(z, as_of, float(p), year1_sd, long_run, phi, sd, chosen) for z, p in zip(zips, pred)}
    out["metro"] = GrowthForecast("metro", as_of, float(np.median(pred)), year1_sd, long_run, phi, sd, chosen)
    return out


def forecast_growth(zip_code: str, as_of: str | pd.Timestamp, method: str = WINNER,
                    logz: pd.DataFrame | None = None, fred: pd.DataFrame | None = None) -> GrowthForecast:
    forecasts = forecast_all(as_of, method, logz, fred)
    return forecasts.get(zip_code, forecasts["metro"])
