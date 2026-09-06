import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

AMENITIES = {
    "Pool": "has_pool",
    "Gym": "has_gym",
    "Parking": "has_parking",
    "Washer Dryer": "has_washer_dryer",
    "Patio/Deck": "has_patio",
    "Fireplace": "has_fireplace",
    "Gated": "has_gated",
    "Wood Floors": "has_wood_floors",
    "Dishwasher": "has_dishwasher",
    "Elevator": "has_elevator",
    "Storage": "has_storage",
    "Clubhouse": "has_clubhouse",
}
COUNTIES = {"121": "Fulton", "089": "DeKalb", "067": "Cobb", "135": "Gwinnett", "063": "Clayton"}
LAT0 = 33.75
KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(LAT0))

SIZE = ["bedrooms", "bathrooms", "square_feet", "log_sqft", "sqft_per_bedroom"]
LOCATION = ["latitude", "longitude", "km_to_downtown", "km_to_marta", "coarse_location",
            *[f"county_{c}" for c in [*COUNTIES.values(), "Other"]]]
NEIGHBORHOOD = ["tract_median_income", "tract_median_rent", "tract_renter_share", "tract_vacancy_rate",
                "zip_median_income", "zip_median_rent", "zip_renter_share", "zip_vacancy_rate", "hud_safmr"]
MARKET = ["zori", "zori_yoy", "zori_missing", "mortgage_30y_at_listing"]
LISTING = [*AMENITIES.values(), "n_amenities", "pets_allowed_flag", "photo_level"]
NEIGHBOR = ["nbr_log_rent_psf", "nbr_km"]
FEATURES = [*SIZE, *LOCATION, *NEIGHBORHOOD, *MARKET, *LISTING, *NEIGHBOR]
FEATURE_GROUPS = {"size": SIZE, "location": LOCATION, "neighborhood": NEIGHBORHOOD, "market": MARKET,
                  "listing": LISTING, "nearby_rent": NEIGHBOR}


def county_name(fips: pd.Series) -> pd.Series:
    return fips.map(COUNTIES).fillna("Other")


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_sqft"] = np.log(out["square_feet"])
    out["sqft_per_bedroom"] = out["square_feet"] / out["bedrooms"].clip(lower=1)
    out["coarse_location"] = out["coarse_location"].astype(int)
    county = county_name(out["county_fips"])
    for name in [*COUNTIES.values(), "Other"]:
        out[f"county_{name}"] = (county == name).astype(int)
    out["zori_missing"] = out["zori_at_listing"].isna().astype(int)
    out["zori"] = out["zori_at_listing"].fillna(out["zori_metro_at_listing"])
    out["zori_yoy"] = out["zori_yoy_at_listing"].fillna(out["zori_yoy_metro_at_listing"])
    amenities = out["amenities"].fillna("").str.split(",")
    for label, col in AMENITIES.items():
        out[col] = amenities.apply(lambda items, label=label: int(label in items))
    out["n_amenities"] = amenities.apply(lambda items: sum(1 for i in items if i))
    out["pets_allowed_flag"] = out["pets_allowed"].fillna("None").ne("None").astype(int)
    out["photo_level"] = out["has_photo"].map({"No": 0, "Thumbnail": 1, "Yes": 2}).fillna(0).astype(int)
    return out


def to_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.column_stack([(lon - (-84.39)) * KM_PER_DEG_LON, (lat - LAT0) * KM_PER_DEG_LAT])


class NeighborRentPerSqft(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "NeighborRentPerSqft":
        log_psf = np.asarray(y) - np.log(X["square_feet"].to_numpy())
        points = (pd.DataFrame({"lat": X["latitude"].to_numpy(), "lon": X["longitude"].to_numpy(), "v": log_psf})
                  .groupby(["lat", "lon"])["v"].agg(["median", "size"]).reset_index())
        self.points_xy_ = to_km(points["lat"].to_numpy(), points["lon"].to_numpy())
        self.point_value_ = points["median"].to_numpy()
        self.point_n_ = points["size"].to_numpy()
        return self

    def _features(self, X: pd.DataFrame, exclude_own_point: bool) -> pd.DataFrame:
        xy = to_km(X["latitude"].to_numpy(), X["longitude"].to_numpy())
        dist = np.linalg.norm(xy[:, None, :] - self.points_xy_[None, :, :], axis=2)
        if exclude_own_point:
            dist = np.where(dist < 1e-6, np.inf, dist)
        k = min(self.k, dist.shape[1] - int(exclude_own_point))
        idx = np.argsort(dist, axis=1)[:, :k]
        d = np.take_along_axis(dist, idx, axis=1)
        w = self.point_n_[idx] / (1 + d)
        out = X.copy()
        out["nbr_log_rent_psf"] = (self.point_value_[idx] * w).sum(axis=1) / w.sum(axis=1)
        out["nbr_km"] = d.mean(axis=1)
        return out

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray = None, **fit_params) -> pd.DataFrame:
        return self.fit(X, y)._features(X, exclude_own_point=True)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._features(X, exclude_own_point=False)


class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None) -> "SelectColumns":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns].astype(float)
