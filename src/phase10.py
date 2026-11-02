import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.api.main import file_version
from src.api.store import Store
from src.config import MLFLOW_TRACKING_URI, ROOT
from src.monitoring import Thresholds, clear_source, drift_detected, monthly_drift, replay
from src.phase4 import load
from src.phase6 import ANALYZER
from src.retrain import HOLDOUT_SHARE, LOCAL_DATABASE, retrain

SHOCK = 0.15
DRILL_DIR = ROOT / "models" / "drill"
SCENARIOS = {
    "replay": "late-2019 listings replayed as live traffic, real rents",
    "replay-shock": f"same listings, every actual rent raised {SHOCK:.0%} (simulated market jump)",
}


def table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.astype(object).to_markdown(index=False, floatfmt=floatfmt)


def reset_registry(name: str) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        MlflowClient().delete_registered_model(name)
    except MlflowException:
        pass


def run_scenario(store: Store, analyzer, version: str, test: pd.DataFrame, source: str) -> tuple[pd.DataFrame, dict]:
    clear_source(store, source)
    actual = test.set_index("id")["rent"] * (1 + SHOCK if source.endswith("shock") else 1)
    replay(store, analyzer, test, version, source, actual)
    report = monthly_drift(store.engine, source)
    model_name = f"rent-model-drill-{source}"
    reset_registry(model_name)
    out = DRILL_DIR / source
    out.mkdir(parents=True, exist_ok=True)
    champion = out / "champion.joblib"
    joblib.dump(analyzer, champion)
    result = retrain(store.engine, source, champion, model_name, out,
                     "drift" if drift_detected(report) else "forced (no drift flagged)")
    return report, result


def write_report(thresholds: Thresholds, results: dict) -> None:
    lines = [
        "# Phase 10 findings: monitoring and retraining",
        "",
        "## Setup",
        "",
        "- Every API prediction is logged to Postgres (`predictions`, with its model feature vector and model "
        "version). Real rents arrive later through `POST /actuals` into `rent_actuals`, linked by prediction id.",
        "- No real post-2019 rents exist for these listings, so the drill replays the newest 20% of listings "
        "(the Phase 3 test set, Sep and Dec 2019) as live traffic through the production analyzer, then records "
        "their real rents as actuals. Replay rows are tagged by `source`, so they never mix with API traffic.",
        f"- Drift rule (SQL, per month and model version, months with at least {thresholds.min_n} actuals): "
        f"MAE above {thresholds.mae_ratio:.2f} x the champion's reference MAE (${thresholds.reference_mae:.0f}, "
        f"limit ${thresholds.mae_limit:.0f}), or 80%-range coverage below {thresholds.coverage_floor:.0%}.",
        f"- Retraining: the newest {HOLDOUT_SHARE:.0%} of labeled rows (by date) are the recent holdout. The "
        "challenger is refit with the champion's hyperparameters on the original training data plus the older "
        f"{1 - HOLDOUT_SHARE:.0%}. It is registered in the MLflow Model Registry with alias `challenger`, and promoted to `champion` only if its holdout MAE "
        "is lower **and** the 95% paired-bootstrap CI of the gain is above zero.",
        "",
    ]
    for source, (report, result) in results.items():
        lines += [
            f"## Scenario `{source}`: {SCENARIOS[source]}",
            "",
            table(report.drop(columns=["model_version"]).assign(month=report["month"].dt.strftime("%Y-%m"))),
            "",
            f"Drift flagged: **{'yes' if drift_detected(report) else 'no'}**. Retrain reason: {result['reason']}. "
            f"{result['new_rows']} labeled rows; {result['added_to_training']} added to training, "
            f"{result['holdout_n']} held out.",
            "",
            "| | MAE | 80% coverage |",
            "|---|---:|---:|",
            f"| champion (v{result['champion_version']}) | ${result['champion_mae']:.0f} | "
            f"{result['champion_coverage']:.1%} |",
            f"| challenger (v{result['challenger_version']}) | ${result['challenger_mae']:.0f} | "
            f"{result['challenger_coverage']:.1%} |",
            "",
            f"Gain {result['gain']:.1%} (95% CI {result['gain_ci_low']:.1%} to {result['gain_ci_high']:.1%}). "
            f"**Decision: {'PROMOTE' if result['promote'] else 'keep champion'}.**",
            "",
        ]
    (ROOT / "reports" / "phase10_findings.md").write_text("\n".join(lines))


def main() -> int:
    store = Store(os.getenv("DATABASE_URL", LOCAL_DATABASE))
    analyzer = joblib.load(ANALYZER)
    version = file_version(ANALYZER)
    _, test = load()
    results = {}
    for source in SCENARIOS:
        report, result = run_scenario(store, analyzer, version, test, source)
        print(f"[{source}]\n{report.round(3).to_string(index=False)}\n{pd.Series(result).to_string()}\n")
        results[source] = (report, result)
    write_report(Thresholds(), results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
