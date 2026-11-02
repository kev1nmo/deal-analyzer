from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import delete, select, text
from sqlalchemy.engine import Engine

from src.api.features import MODEL_INPUT
from src.api.store import Store, predictions, rent_actuals
from src.explain import CONFIDENCE, RentAnalyzer

REFERENCE_MAE = 155.0
MAE_RATIO_LIMIT = 1.25
COVERAGE_FLOOR = 0.70
MIN_ACTUALS = 30

DRIFT_SQL = """
WITH scored AS (
    SELECT
        a.observed_month AS month,
        p.model_version,
        ABS(a.rent - p.predicted_rent) AS abs_error,
        (a.rent - p.predicted_rent) / a.rent AS pct_error,
        CASE WHEN a.rent BETWEEN p.range_low AND p.range_high THEN 1.0 ELSE 0.0 END AS covered
    FROM rent_actuals a
    JOIN predictions p ON p.id = a.prediction_id
    WHERE p.source = :source
),
monthly AS (
    SELECT
        month,
        model_version,
        COUNT(*) AS n,
        AVG(abs_error) AS mae,
        AVG(pct_error) AS bias_pct,
        AVG(covered) AS coverage
    FROM scored
    GROUP BY month, model_version
)
SELECT
    month,
    model_version,
    n,
    mae,
    bias_pct,
    coverage,
    CASE WHEN n >= :min_n AND mae > :mae_limit THEN 1 ELSE 0 END AS mae_drift,
    CASE WHEN n >= :min_n AND coverage < :coverage_floor THEN 1 ELSE 0 END AS coverage_drift
FROM monthly
ORDER BY month, model_version
"""


@dataclass(frozen=True)
class Thresholds:
    reference_mae: float = REFERENCE_MAE
    mae_ratio: float = MAE_RATIO_LIMIT
    coverage_floor: float = COVERAGE_FLOOR
    min_n: int = MIN_ACTUALS

    @property
    def mae_limit(self) -> float:
        return self.reference_mae * self.mae_ratio


def monthly_drift(engine: Engine, source: str = "api", thresholds: Thresholds = Thresholds()) -> pd.DataFrame:
    params = {"source": source, "min_n": thresholds.min_n, "mae_limit": thresholds.mae_limit,
              "coverage_floor": thresholds.coverage_floor}
    with engine.connect() as conn:
        df = pd.read_sql(text(DRIFT_SQL), conn, params=params)
    df["month"] = pd.to_datetime(df["month"])
    df["drift"] = (df["mae_drift"] + df["coverage_drift"]) > 0
    return df


def drift_detected(report: pd.DataFrame) -> bool:
    return bool(report["drift"].any())


def clear_source(store: Store, source: str) -> None:
    with store.engine.begin() as conn:
        ids = select(predictions.c.id).where(predictions.c.source == source)
        conn.execute(delete(rent_actuals).where(rent_actuals.c.prediction_id.in_(ids)))
        conn.execute(delete(predictions).where(predictions.c.source == source))


def replay(store: Store, analyzer: RentAnalyzer, rows: pd.DataFrame, model_version: str, source: str,
           actual_rent: pd.Series | None = None) -> pd.DataFrame:
    rows = rows.sort_values(["listed_at", "id"]).reset_index(drop=True)
    actual = (rows["rent"] if actual_rent is None else actual_rent.reindex(rows["id"])).to_numpy()
    level = analyzer.levels.index(CONFIDENCE)
    pred = analyzer.predict(rows)
    bounds = analyzer.predict_interval(rows)
    known = analyzer.known_location(rows)
    feats = rows[MODEL_INPUT].astype(float)
    pred_rows = [{
        "created_at": rows.at[i, "listed_at"], "model_version": model_version, "source": source,
        "zip": rows.at[i, "zip"], "request": {"listing_id": int(rows.at[i, "id"])},
        "features": {k: (None if not np.isfinite(v) else float(v)) for k, v in feats.iloc[i].items()},
        "predicted_rent": float(pred[i]), "range_low": float(bounds[i, 0, level]),
        "range_high": float(bounds[i, 1, level]), "rent_today": float(pred[i]), "known_location": bool(known[i]),
        "latency_ms": 0.0,
    } for i in range(len(rows))]
    ids = store.log_many(predictions, pred_rows)
    observed = pd.to_datetime(rows["listed_at"])
    store.log_many(rent_actuals, [{
        "prediction_id": pid, "listing_id": int(rows.at[i, "id"]), "rent": float(actual[i]),
        "observed_at": observed[i].to_pydatetime(), "observed_month": observed[i].to_period("M").to_timestamp().date(),
    } for i, pid in enumerate(ids)])
    return rows.assign(prediction_id=ids, actual_rent=actual)
