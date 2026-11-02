import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine

from src.api.features import MODEL_INPUT
from src.api.main import file_version
from src.api.store import predictions, rent_actuals
from src.config import MLFLOW_TRACKING_URI, ROOT
from src.explain import CONFIDENCE, RentAnalyzer
from src.models import paired_bootstrap_gain
from src.monitoring import drift_detected, monthly_drift
from src.phase4 import TRUSTED_TYPES, load

REGISTERED_MODEL = "rent-model"
EXPERIMENT = "rent-retrain"
HOLDOUT_SHARE = 0.3
MODELS = ROOT / "models"
PRODUCTION = MODELS / "rent_analyzer.joblib"
LOCAL_DATABASE = "postgresql+psycopg://deals:deals@localhost:5432/deals"


def labeled_rows(engine: Engine, source: str, pool: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    query = (select(rent_actuals.c.listing_id, rent_actuals.c.rent.label("actual_rent"), rent_actuals.c.observed_at)
             .join(predictions, predictions.c.id == rent_actuals.c.prediction_id)
             .where(predictions.c.source == source, rent_actuals.c.listing_id.is_not(None)))
    with engine.connect() as conn:
        actuals = pd.read_sql(query, conn)
    rows = pool.merge(actuals, left_on="id", right_on="listing_id", how="inner", validate="one_to_one")
    rows = rows[~rows["id"].isin(train["id"])]
    fold_of_cluster = train.groupby("cluster")["fold"].first()
    rows["fold"] = rows["cluster"].map(fold_of_cluster).fillna(0).astype(int)
    rows["rent"] = rows["actual_rent"]
    return rows.drop(columns=["listing_id", "actual_rent"]).sort_values(["observed_at", "id"]).reset_index(drop=True)


def split_recent(rows: pd.DataFrame, share: float = HOLDOUT_SHARE) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = rows.sort_values(["observed_at", "id"]).reset_index(drop=True)
    cut = len(rows) - int(np.ceil(len(rows) * share))
    return rows.iloc[:cut], rows.iloc[cut:]


def decide(champion_mae: float, challenger_mae: float, gain_ci_low: float) -> bool:
    return challenger_mae < champion_mae and gain_ci_low > 0


def coverage(analyzer: RentAnalyzer, rows: pd.DataFrame) -> float:
    level = analyzer.levels.index(CONFIDENCE)
    b = analyzer.predict_interval(rows)
    y = rows["rent"].to_numpy()
    return float(((y >= b[:, 0, level]) & (y <= b[:, 1, level])).mean())


def compare(champion: RentAnalyzer, challenger: RentAnalyzer, holdout: pd.DataFrame) -> dict:
    y = holdout["rent"].to_numpy()
    champ, chall = champion.predict(holdout), challenger.predict(holdout)
    gain, lo, hi = paired_bootstrap_gain(y, champ, chall)
    out = {"holdout_n": len(holdout), "champion_mae": float(np.abs(y - champ).mean()),
           "challenger_mae": float(np.abs(y - chall).mean()), "gain": gain, "gain_ci_low": lo, "gain_ci_high": hi,
           "champion_coverage": coverage(champion, holdout), "challenger_coverage": coverage(challenger, holdout)}
    out["promote"] = decide(out["champion_mae"], out["challenger_mae"], lo)
    return out


def register(analyzer: RentAnalyzer, path: Path, model_name: str, alias: str, example: pd.DataFrame,
             tags: dict) -> str:
    info = mlflow.sklearn.log_model(analyzer.model_, name="model", input_example=example[MODEL_INPUT].head(3),
                                    skops_trusted_types=TRUSTED_TYPES, registered_model_name=model_name)
    mlflow.log_artifact(str(path))
    version = str(info.registered_model_version)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, version)
    for k, v in {**tags, "analyzer_version": file_version(path), "target": "log_rent"}.items():
        client.set_model_version_tag(model_name, version, k, str(v))
    return version


def ensure_champion(model_name: str, champion: RentAnalyzer, champion_path: Path, example: pd.DataFrame) -> str:
    client = MlflowClient()
    try:
        return client.get_model_version_by_alias(model_name, "champion").version
    except MlflowException:
        with mlflow.start_run(run_name="register-champion"):
            return register(champion, champion_path, model_name, "champion", example, {"role": "initial champion"})


def retrain(engine: Engine, source: str, champion_path: Path, model_name: str, out_dir: Path,
            reason: str) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    champion = joblib.load(champion_path)
    train, test = load()
    pool = pd.concat([train, test], ignore_index=True)
    rows = labeled_rows(engine, source, pool, train)
    recent_train, holdout = split_recent(rows)
    champion_version = ensure_champion(model_name, champion, champion_path, holdout)

    challenger = RentAnalyzer(champion.params, champion.k).fit(pd.concat([train, recent_train], ignore_index=True))
    result = {"source": source, "reason": reason, "new_rows": len(rows), "added_to_training": len(recent_train),
              "champion_version": champion_version, **compare(champion, challenger, holdout)}

    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = out_dir / "challenger.joblib"
    joblib.dump(challenger, candidate)
    with mlflow.start_run(run_name=f"challenger-{source}"):
        mlflow.log_params({"source": source, "reason": reason, "holdout_share": HOLDOUT_SHARE,
                           "train_rows": len(train) + len(recent_train), "champion_version": champion_version})
        mlflow.log_metrics({k: float(v) for k, v in result.items()
                            if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)})
        mlflow.log_metric("promoted", int(result["promote"]))
        result["challenger_version"] = register(challenger, candidate, model_name, "challenger", holdout,
                                                {"role": "challenger", "promoted": result["promote"]})
    if result["promote"]:
        promote(model_name, result["challenger_version"], candidate, champion_path)
    return result


def promote(model_name: str, version: str, candidate: Path, champion_path: Path) -> None:
    archive = champion_path.parent / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    if champion_path.exists():
        shutil.copy2(champion_path, archive / f"rent_analyzer_{file_version(champion_path)}.joblib")
    shutil.copy2(candidate, champion_path)
    MlflowClient().set_registered_model_alias(model_name, "champion", version)


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain the rent model when monitoring flags drift.")
    parser.add_argument("--source", default="api")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    engine = create_engine(os.getenv("DATABASE_URL", LOCAL_DATABASE))
    report = monthly_drift(engine, args.source)
    print(report.round(3).to_string(index=False))
    if not (drift_detected(report) or args.force):
        print("No drift flagged; champion stays.")
        return 0
    result = retrain(engine, args.source, PRODUCTION, REGISTERED_MODEL, MODELS / "candidates",
                     "drift" if drift_detected(report) else "forced")
    print(pd.Series(result).to_string())
    if result["promote"]:
        print("Promoted. Restart the API to serve it: docker compose restart api")
    return 0


if __name__ == "__main__":
    sys.exit(main())
