import os
import sys
from itertools import product
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd

from src.baseline import TARGET, baselines, bootstrap_mae_ci, metrics
from src.config import DATA_PROCESSED, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, RANDOM_SEED, ROOT
from src.model_features import FEATURE_GROUPS, FEATURES, NEIGHBOR, add_base_features, county_name
from src.models import build_ridge, build_xgb, cv_predict, cv_score, paired_bootstrap_gain

REPORTS = ROOT / "reports"
PREDICTIONS = DATA_PROCESSED / "phase4_test_predictions.parquet"
REGISTERED_MODEL = "rent-model"
N_TRIALS = 60
RIDGE_ALPHAS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]
NEIGHBOR_K = [3, 5, 8, 12]
PRICE_BANDS = [0, 1000, 1300, 1600, np.inf]
PRICE_LABELS = ["< $1,000", "$1,000-1,299", "$1,300-1,599", "$1,600+"]
TARGET_GAIN = 0.20
MODEL_INPUT = [f for f in FEATURES if f not in NEIGHBOR]
TRUSTED_TYPES = [
    "numpy.dtype",
    "src.model_features.NeighborRentPerSqft",
    "src.model_features.SelectColumns",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBRegressor",
]


def table(df: pd.DataFrame, floatfmt: str = ".1f") -> str:
    return df.astype(object).to_markdown(index=False, floatfmt=floatfmt)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(DATA_PROCESSED / "listings_features.parquet").merge(
        pd.read_parquet(DATA_PROCESSED / "splits.parquet"), on="id", validate="one_to_one")
    df = add_base_features(df)
    return df[~df["is_test"]].reset_index(drop=True), df[df["is_test"]].reset_index(drop=True)


def baseline_cv(train: pd.DataFrame) -> dict[str, float]:
    model = baselines()["zip_beds_median"]
    pred = np.full(len(train), np.nan)
    for fold in sorted(train["fold"].unique()):
        val = (train["fold"] == fold).to_numpy()
        pred[val] = model.fit(train[~val]).predict(train[val])
    per_fold = [metrics(train.loc[train["fold"] == f, TARGET], pred[(train["fold"] == f).to_numpy()])["mae"]
                for f in sorted(train["fold"].unique())]
    return {**{f"cv_{k}": v for k, v in metrics(train[TARGET], pred).items()},
            "cv_mae_fold_sd": float(np.std(per_fold, ddof=1))}


def search_ridge(train: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = []
    with mlflow.start_run(run_name="ridge_search"):
        mlflow.set_tags({"phase": "4", "model_type": "ridge", "stage": "search"})
        for alpha, k in product(RIDGE_ALPHAS, NEIGHBOR_K):
            score = cv_score(build_ridge(alpha, k), train, train[TARGET], train["fold"])
            with mlflow.start_run(run_name=f"ridge_a{alpha}_k{k}", nested=True):
                mlflow.log_params({"alpha": alpha, "k": k})
                mlflow.log_metrics(score)
            rows.append({"alpha": alpha, "k": k, **score})
        results = pd.DataFrame(rows).sort_values("cv_mae")
        best = results.iloc[0]
        mlflow.log_params({"best_alpha": best["alpha"], "best_k": int(best["k"])})
        mlflow.log_metric("best_cv_mae", best["cv_mae"])
    return {"alpha": float(best["alpha"]), "k": int(best["k"])}, results


def xgb_params(trial: optuna.Trial) -> tuple[dict, int]:
    params = {
        "objective": trial.suggest_categorical("objective", ["reg:absoluteerror", "reg:squarederror"]),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 30, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
    }
    return params, trial.suggest_categorical("k", NEIGHBOR_K)


def search_xgb(train: pd.DataFrame, n_trials: int = N_TRIALS) -> tuple[dict, int, pd.DataFrame]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params, k = xgb_params(trial)
        score = cv_score(build_xgb(params, k), train, train[TARGET], train["fold"])
        with mlflow.start_run(run_name=f"xgb_trial_{trial.number}", nested=True):
            mlflow.log_params({**params, "k": k})
            mlflow.log_metrics(score)
        trial.set_user_attr("cv_mae_fold_sd", score["cv_mae_fold_sd"])
        return score["cv_mae"]

    with mlflow.start_run(run_name="xgb_optuna"):
        mlflow.set_tags({"phase": "4", "model_type": "xgboost", "stage": "search"})
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(objective, n_trials=n_trials)
        best = dict(study.best_params)
        k = best.pop("k")
        mlflow.log_params({f"best_{p}": v for p, v in {**best, "k": k}.items()})
        mlflow.log_metric("best_cv_mae", study.best_value)
    trials = study.trials_dataframe(attrs=("number", "value"))
    return best, k, trials.rename(columns={"value": "cv_mae"})


def ablation(train: pd.DataFrame, params: dict, k: int) -> pd.DataFrame:
    rows = []
    with mlflow.start_run(run_name="xgb_feature_ablation"):
        mlflow.set_tags({"phase": "4", "model_type": "xgboost", "stage": "ablation"})
        for group, cols in {"none (all features)": [], **FEATURE_GROUPS}.items():
            features = [f for f in FEATURES if f not in cols]
            score = cv_score(build_xgb(params, k, features), train, train[TARGET], train["fold"])
            with mlflow.start_run(run_name=f"drop_{group}", nested=True):
                mlflow.log_params({"dropped_group": group, "n_features": len(features)})
                mlflow.log_metrics(score)
            rows.append({"dropped": group, "n_features": len(features), "cv_mae": score["cv_mae"]})
    out = pd.DataFrame(rows)
    out["mae_change"] = out["cv_mae"] - out.loc[0, "cv_mae"]
    return out


def evaluate(name: str, y: pd.Series, pred: np.ndarray, base_pred: np.ndarray) -> dict:
    lo, hi = bootstrap_mae_ci(y, pred)
    gain, gain_lo, gain_hi = paired_bootstrap_gain(y, base_pred, pred)
    return {"model": name, **metrics(y, pred), "mae_ci_low": lo, "mae_ci_high": hi,
            "gain_vs_baseline": gain * 100, "gain_ci_low": gain_lo * 100, "gain_ci_high": gain_hi * 100}


def breakdown(pred: pd.DataFrame, by: str, models: list[str]) -> pd.DataFrame:
    rows = []
    for group, part in pred.groupby(by, observed=True):
        row = {by: group, "listings": len(part)}
        for m in models:
            row[f"{m}_mae"] = (part[TARGET] - part[m]).abs().mean()
            row[f"{m}_mape"] = ((part[TARGET] - part[m]).abs() / part[TARGET]).mean() * 100
        rows.append(row)
    return pd.DataFrame(rows)


def fit_final(name: str, model, train: pd.DataFrame, test: pd.DataFrame, cv: dict, params: dict,
              register: bool) -> np.ndarray:
    model.fit(train, train[TARGET])
    pred = model.predict(test)
    with mlflow.start_run(run_name=f"{name}_final"):
        mlflow.set_tags({"phase": "4", "model_type": name, "stage": "final"})
        mlflow.log_params(params)
        mlflow.log_metrics(cv)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics(test[TARGET], pred).items()})
        mlflow.sklearn.log_model(
            model, name="model", input_example=test.head(3)[MODEL_INPUT], skops_trusted_types=TRUSTED_TYPES,
            registered_model_name=REGISTERED_MODEL if register else None,
        )
    return pred



def write_report(cv_table, test_table, by_county, by_band, ablation_table, ridge_best, xgb_best, xgb_k,
                 winner, base_test_mae, path: Path) -> None:
    w = test_table.set_index("model").loc[winner]
    hit = w["mae"] <= base_test_mae * (1 - TARGET_GAIN)
    lines = [
        "# Phase 4 findings: rent model",
        "",
        "## 1. Setup",
        "",
        f"- {len(FEATURES)} features in 6 groups: size, location, neighborhood (Census and HUD), market "
        "(ZORI with metro fallback, mortgage rate), listing (amenity flags, pets, photo) and nearby rent "
        "(rent per sq ft at the nearest training map points).",
        "- Nearby rent is leakage-safe. During training each listing sees only *other* map points, never its own. "
        "At prediction time it sees every training point. It is computed inside the model pipeline, so in CV it "
        "only ever sees the training folds.",
        "- Both models predict log(rent) and convert back to dollars.",
        f"- Tuning used spatial CV on the training set only (Phase 3 folds). Ridge: {len(RIDGE_ALPHAS) * len(NEIGHBOR_K)} "
        f"grid points. XGBoost: {N_TRIALS} Optuna trials. The winner was picked on CV MAE before the test set was opened.",
        "",
        "## 2. Spatial cross-validation (training set, $)",
        "",
        table(cv_table),
        "",
        f"Best Ridge: alpha = {ridge_best['alpha']}, k = {ridge_best['k']} nearby points. "
        f"Best XGBoost: k = {xgb_k}, " + ", ".join(f"{p} = {v:.4g}" if isinstance(v, float) else f"{p} = {v}"
                                                   for p, v in xgb_best.items()) + ".",
        "",
        "## 3. Test set (newest 20%, opened once)",
        "",
        table(test_table),
        "",
        "`gain_vs_baseline`: % reduction in MAE vs the ZIP x beds median, with a 95% paired bootstrap CI "
        "(the same resampled listings for both models).",
        "",
        f"**Winner (chosen on CV): {winner}.** Test MAE ${w['mae']:.0f} vs baseline ${base_test_mae:.0f}: "
        f"{w['gain_vs_baseline']:.1f}% lower (95% CI {w['gain_ci_low']:.1f}% to {w['gain_ci_high']:.1f}%). "
        f"Target was 20% lower (MAE <= ${base_test_mae * 0.8:.0f}): **{'met' if hit else 'not met'}**.",
        "",
        "## 4. By region (county) - test MAE / MAPE",
        "",
        table(by_county),
        "",
        "## 5. By price band (actual rent) - test MAE / MAPE",
        "",
        table(by_band),
        "",
        "Bands use the *actual* rent, so every model overpredicts the cheapest band and underpredicts the "
        "priciest (regression to the middle). Compare models within a band, not bands with each other.",
        "",
        "## 6. Which feature groups matter? (XGBoost, spatial CV, one group dropped at a time)",
        "",
        table(ablation_table),
        "",
        "`mae_change`: how much CV MAE rises when the group is removed. Bigger means the group carries more "
        "information the others cannot replace.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    if not (DATA_PROCESSED / "splits.parquet").exists():
        print("Missing splits. Run: python -m src.phase3")
        return 1
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    train, test = load()
    print(f"[data] train {len(train):,}, test {len(test):,}, {len(FEATURES)} features")

    base_cv = baseline_cv(train)
    print(f"[baseline] spatial CV MAE {base_cv['cv_mae']:.1f}")

    print("[ridge] grid search")
    ridge_best, ridge_grid = search_ridge(train)
    ridge_cv = ridge_grid.iloc[0].to_dict()
    print(f"  best {ridge_best} CV MAE {ridge_cv['cv_mae']:.1f}")

    print(f"[xgboost] optuna, {N_TRIALS} trials")
    xgb_best, xgb_k, _ = search_xgb(train)
    xgb_cv = cv_score(build_xgb(xgb_best, xgb_k), train, train[TARGET], train["fold"])
    print(f"  best k={xgb_k} CV MAE {xgb_cv['cv_mae']:.1f}")

    print("[ablation]")
    abl = ablation(train, xgb_best, xgb_k)
    print(abl.round(1).to_string(index=False))

    cv_rows = [{"model": "zip_beds_median (baseline)", **base_cv},
               {"model": "ridge", **{k: ridge_cv[k] for k in base_cv}},
               {"model": "xgboost", **xgb_cv}]
    cv_table = pd.DataFrame(cv_rows)
    cv_table["gain_vs_baseline_pct"] = (1 - cv_table["cv_mae"] / base_cv["cv_mae"]) * 100
    winner = "xgboost" if xgb_cv["cv_mae"] <= ridge_cv["cv_mae"] else "ridge"
    print(cv_table.round(1).to_string(index=False))
    print(f"[winner on CV] {winner}")

    print("[test] fit on all training data, evaluate once")
    base_pred = baselines()["zip_beds_median"].fit(train).predict(test)
    ridge_pred = fit_final("ridge", build_ridge(**ridge_best), train, test,
                           {k: ridge_cv[k] for k in base_cv}, ridge_best, winner == "ridge")
    xgb_pred = fit_final("xgboost", build_xgb(xgb_best, xgb_k), train, test, xgb_cv,
                         {**xgb_best, "k": xgb_k}, winner == "xgboost")
    test_table = pd.DataFrame([
        {"model": "zip_beds_median (baseline)", **metrics(test[TARGET], base_pred),
         **dict(zip(["mae_ci_low", "mae_ci_high"], bootstrap_mae_ci(test[TARGET], base_pred)))},
        evaluate("ridge", test[TARGET], ridge_pred, base_pred),
        evaluate("xgboost", test[TARGET], xgb_pred, base_pred),
    ])
    print(test_table.round(1).to_string(index=False))

    pred = test[["id", TARGET, "bedrooms", "county_fips", "km_to_downtown", "cluster"]].assign(
        baseline=base_pred, ridge=ridge_pred, xgboost=xgb_pred,
        county=county_name(test["county_fips"]),
        price_band=pd.cut(test[TARGET], PRICE_BANDS, labels=PRICE_LABELS, right=False),
    )
    pred.to_parquet(PREDICTIONS, index=False)
    models = ["baseline", "ridge", "xgboost"]
    by_county = breakdown(pred, "county", models).sort_values("listings", ascending=False)
    by_band = breakdown(pred, "price_band", models)

    path = REPORTS / "phase4_findings.md"
    write_report(cv_table, test_table, by_county, by_band, abl, ridge_best, xgb_best, xgb_k, winner,
                 test_table.iloc[0]["mae"], path)
    print(f"[report] {path.relative_to(ROOT)}")
    print(f"[mlflow] winner registered as '{REGISTERED_MODEL}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
