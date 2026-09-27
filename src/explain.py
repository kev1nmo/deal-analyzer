from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap
from mapie.regression import CrossConformalRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors

from src.config import RANDOM_SEED
from src.model_features import county_name, to_km
from src.models import build_xgb, cv_predict
from src.validation import random_folds

CONFIDENCE = 0.8
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
POINT = ["latitude", "longitude"]
MIN_GROUP = 100
COMP_COLUMNS = ["id", "rent", "bedrooms", "bathrooms", "square_feet", "cityname", "latitude", "longitude", "listed_at"]
COMP_WEIGHTS = {"km": 1.0, "log_sqft": 4.0, "bedrooms": 1.5, "bathrooms": 1.0}


def comp_space(df: pd.DataFrame) -> np.ndarray:
    xy = to_km(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    return np.column_stack([
        xy * COMP_WEIGHTS["km"],
        np.log(df["square_feet"].to_numpy()) * COMP_WEIGHTS["log_sqft"],
        df["bedrooms"].to_numpy() * COMP_WEIGHTS["bedrooms"],
        df["bathrooms"].to_numpy() * COMP_WEIGHTS["bathrooms"],
    ])


@dataclass
class RentAnalyzer:
    params: dict
    k: int
    levels: list[float] = field(default_factory=lambda: list(LEVELS))
    n_comps: int = 5

    def _conformal(self, train: pd.DataFrame, cv, groups=None) -> CrossConformalRegressor:
        model = CrossConformalRegressor(build_xgb(self.params, self.k).regressor, confidence_level=self.levels,
                                        method="plus", cv=cv)
        return model.fit_conformalize(train, np.log(train["rent"].to_numpy()), groups=groups)

    def fit(self, train: pd.DataFrame) -> "RentAnalyzer":
        y = np.log(train["rent"].to_numpy())
        self.spatial_ = self._conformal(train, GroupKFold(n_splits=train["fold"].nunique()), train["fold"].to_numpy())
        self.random_ = self._conformal(train, KFold(n_splits=train["fold"].nunique(), shuffle=True,
                                                    random_state=RANDOM_SEED))
        self.known_points_ = set(map(tuple, train[POINT].to_numpy()))
        self.model_ = build_xgb(self.params, self.k).regressor.fit(train, y)
        self.groups_ = self._region(train).value_counts().loc[lambda c: c >= MIN_GROUP].index.tolist()
        region = self.region(train)
        self.quantiles_ = {}
        for scheme, folds in [("random", random_folds(train.index)), ("spatial", train["fold"])]:
            resid = np.abs(y - np.log(cv_predict(build_xgb(self.params, self.k), train, train["rent"], folds)))
            for group in region.unique():
                r = np.sort(resid[(region == group).to_numpy()])
                self.quantiles_[(scheme, group)] = [r[min(len(r) - 1, int(np.ceil((len(r) + 1) * lv)) - 1)]
                                                    for lv in self.levels]
        self.explainer_ = shap.TreeExplainer(self.model_.named_steps["model"])
        self.comps_ = train[COMP_COLUMNS].reset_index(drop=True)
        self.nn_ = NearestNeighbors(n_neighbors=self.n_comps).fit(comp_space(train))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.model_.predict(X))

    def known_location(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([p in self.known_points_ for p in map(tuple, X[POINT].to_numpy())])

    @staticmethod
    def _region(X: pd.DataFrame) -> pd.Series:
        return county_name(X["county_fips"]).set_axis(X.index)

    def region(self, X: pd.DataFrame) -> pd.Series:
        return self._region(X).where(lambda r: r.isin(self.groups_), "Other")

    def mondrian_interval(self, X: pd.DataFrame) -> np.ndarray:
        center = self.model_.predict(X)
        known = self.known_location(X)
        q = np.array([self.quantiles_[("random" if k else "spatial", g)] for k, g in zip(known, self.region(X))])
        return np.exp(np.stack([center[:, None] - q, center[:, None] + q], axis=1))

    def predict_interval(self, X: pd.DataFrame, mode: str = "location_aware") -> np.ndarray:
        if mode == "mondrian":
            return self.mondrian_interval(X)
        bounds = {}
        for name, model in [("spatial", self.spatial_), ("random", self.random_)]:
            if mode in (name, "location_aware"):
                bounds[name] = np.exp(model.predict_interval(X)[1])
        if mode != "location_aware":
            return bounds[mode]
        return np.where(self.known_location(X)[:, None, None], bounds["random"], bounds["spatial"])

    def features(self, X: pd.DataFrame) -> pd.DataFrame:
        for _, step in self.model_.steps[:-1]:
            X = step.transform(X)
        return X

    def shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        features = self.features(X)
        return pd.DataFrame(self.explainer_.shap_values(features), columns=features.columns, index=X.index)

    def drivers(self, X: pd.DataFrame, top: int = 5) -> list[list[dict]]:
        values = self.shap_values(X)
        features = self.features(X)
        out = []
        for i in range(len(X)):
            row = values.iloc[i]
            order = row.abs().sort_values(ascending=False).index[:top]
            out.append([{"feature": f, "value": float(features.iloc[i][f]),
                         "effect_pct": float((np.exp(row[f]) - 1) * 100)} for f in order])
        return out

    def comps(self, X: pd.DataFrame) -> list[pd.DataFrame]:
        dist, idx = self.nn_.kneighbors(comp_space(X))
        query_xy = to_km(X["latitude"].to_numpy(), X["longitude"].to_numpy())
        out = []
        for i in range(len(X)):
            c = self.comps_.iloc[idx[i]].copy()
            comp_xy = to_km(c["latitude"].to_numpy(), c["longitude"].to_numpy())
            c["km_away"] = np.linalg.norm(comp_xy - query_xy[i], axis=1)
            out.append(c.reset_index(drop=True))
        return out

    def analyze(self, listing: pd.DataFrame) -> dict:
        listing = listing.head(1)
        level = self.levels.index(CONFIDENCE)
        bounds = self.predict_interval(listing)
        return {
            "predicted_rent": float(self.predict(listing)[0]),
            "interval": (float(bounds[0, 0, level]), float(bounds[0, 1, level])),
            "confidence": CONFIDENCE,
            "known_location": bool(self.known_location(listing)[0]),
            "drivers": self.drivers(listing)[0],
            "comps": self.comps(listing)[0],
        }
