import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.baseline import metrics
from src.config import RANDOM_SEED
from src.model_features import FEATURES, NeighborRentPerSqft, SelectColumns


def log_target(pipeline: Pipeline) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp, check_inverse=False)


def build_ridge(alpha: float = 1.0, k: int = 5, features: list[str] = FEATURES) -> TransformedTargetRegressor:
    return log_target(Pipeline([
        ("nearby", NeighborRentPerSqft(k=k)),
        ("select", SelectColumns(features)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ]))


def build_xgb(params: dict | None = None, k: int = 5, features: list[str] = FEATURES) -> TransformedTargetRegressor:
    params = {"objective": "reg:absoluteerror", "n_estimators": 500, "learning_rate": 0.05, "max_depth": 4,
              **(params or {})}
    return log_target(Pipeline([
        ("nearby", NeighborRentPerSqft(k=k)),
        ("select", SelectColumns(features)),
        ("model", XGBRegressor(tree_method="hist", n_jobs=-1,
                               random_state=RANDOM_SEED, **params)),
    ]))


def cv_predict(model, X: pd.DataFrame, y: pd.Series, folds: pd.Series) -> np.ndarray:
    pred = np.full(len(X), np.nan)
    for fold in sorted(folds.unique()):
        val = (folds == fold).to_numpy()
        pred[val] = clone(model).fit(X[~val], y[~val]).predict(X[val])
    return pred


def cv_score(model, X: pd.DataFrame, y: pd.Series, folds: pd.Series) -> dict[str, float]:
    pred = cv_predict(model, X, y, folds)
    per_fold = [metrics(y[folds == f], pred[(folds == f).to_numpy()])["mae"] for f in sorted(folds.unique())]
    return {**{f"cv_{k}": v for k, v in metrics(y, pred).items()}, "cv_mae_fold_sd": float(np.std(per_fold, ddof=1))}


def paired_bootstrap_gain(y: np.ndarray, base_pred: np.ndarray, model_pred: np.ndarray, n: int = 2000,
                          seed: int = RANDOM_SEED) -> tuple[float, float, float]:
    base_err = np.abs(np.asarray(y) - np.asarray(base_pred))
    model_err = np.abs(np.asarray(y) - np.asarray(model_pred))
    idx = np.random.default_rng(seed).integers(0, len(base_err), size=(n, len(base_err)))
    gains = 1 - model_err[idx].mean(axis=1) / base_err[idx].mean(axis=1)
    point = 1 - model_err.mean() / base_err.mean()
    return float(point), float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))
