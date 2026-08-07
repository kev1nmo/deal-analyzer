import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import pandas as pd

from src.baseline import TARGET, baselines, bootstrap_mae_ci, cross_validate, metrics
from src.config import DATA_PROCESSED, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, ROOT
from src.validation import N_CLUSTERS, N_FOLDS, POINT, TEST_FRAC, leaked_groups, make_splits, random_folds

REPORTS = ROOT / "reports"
FEATURES = DATA_PROCESSED / "listings_features.parquet"
SPLITS = DATA_PROCESSED / "splits.parquet"
MAIN = "zip_beds_median"
LEVELS = {0: "ZIP x beds", 1: "beds only (fallback)"}


def table(df: pd.DataFrame, floatfmt: str = ".1f") -> str:
    return df.astype(object).to_markdown(index=False, floatfmt=floatfmt)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(FEATURES)
    splits = make_splits(df)
    splits.to_parquet(SPLITS, index=False)
    return df.merge(splits, on="id", validate="one_to_one"), splits


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, part in [("train", df[~df["is_test"]]), ("test", df[df["is_test"]])]:
        rows.append({
            "set": name,
            "listings": len(part),
            "from": f"{part['listed_at'].min():%Y-%m-%d}",
            "to": f"{part['listed_at'].max():%Y-%m-%d}",
            "median_rent": part[TARGET].median(),
            "share_1br": (part["bedrooms"] == 1).mean(),
            "share_3br_plus": (part["bedrooms"] >= 3).mean(),
            "share_coarse_location": part["coarse_location"].mean(),
            "map_points": part[POINT].drop_duplicates().shape[0],
            "median_km_to_downtown": part["km_to_downtown"].median(),
        })
    return pd.DataFrame(rows)


def cv_table(train: pd.DataFrame) -> pd.DataFrame:
    schemes = {"spatial": train["fold"], "random": random_folds(train.index)}
    rows = []
    for name, model in baselines().items():
        for scheme, folds in schemes.items():
            res = cross_validate(model, train, folds)
            rows.append({"baseline": name, "cv": scheme, "mae": res["mae"].mean(),
                         "mae_fold_sd": res["mae"].std(), "mape": res["mape"].mean()})
    return pd.DataFrame(rows)


def test_table(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, main_pred = [], None
    for name, model in baselines().items():
        pred = model.fit(train).predict_with_level(test)
        lo, hi = bootstrap_mae_ci(test[TARGET], pred["pred"])
        rows.append({"baseline": name, **metrics(test[TARGET], pred["pred"]), "mae_ci_low": lo, "mae_ci_high": hi})
        if name == MAIN:
            main_pred = test.join(pred)
    return pd.DataFrame(rows), main_pred


def breakdown(pred: pd.DataFrame, by: str) -> pd.DataFrame:
    pred = pred.assign(abs_err=(pred[TARGET] - pred["pred"]).abs())
    out = pred.groupby(by).agg(listings=("id", "size"), median_rent=(TARGET, "median"), mae=("abs_err", "mean"))
    return out.reset_index()


def diagnostics(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    model = baselines()[MAIN]
    zip_level = []
    for fold in sorted(train["fold"].unique()):
        val = train["fold"] == fold
        zip_level.append((model.fit(train[~val]).predict_with_level(train[val])["level"] == 0).mean())
    seen_points = set(map(tuple, train[POINT].to_numpy()))
    return {
        "cv_zip_level_share": float(pd.Series(zip_level).mean()),
        "test_seen_point_share": float(test[POINT].apply(tuple, axis=1).isin(seen_points).mean()),
        "test_seen_zip_share": float(test["zip"].isin(train["zip"]).mean()),
    }


def log_mlflow(cv: pd.DataFrame, test: pd.DataFrame, n_train: int, n_test: int) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    for name in test["baseline"]:
        with mlflow.start_run(run_name=f"baseline_{name}"):
            mlflow.set_tags({"phase": "3", "model_type": "baseline"})
            mlflow.log_params({"baseline": name, "test_frac": TEST_FRAC, "n_clusters": N_CLUSTERS,
                               "n_folds": N_FOLDS, "n_train": n_train, "n_test": n_test})
            for _, r in cv[cv["baseline"] == name].iterrows():
                mlflow.log_metrics({f"cv_{r['cv']}_mae": r["mae"], f"cv_{r['cv']}_mape": r["mape"]})
            r = test[test["baseline"] == name].iloc[0]
            mlflow.log_metrics({f"test_{k}": r[k] for k in ["mae", "mape", "median_ae", "mae_ci_low", "mae_ci_high"]})


def write_report(summary, cv, test, levels, by_beds, folds, diag, path: Path) -> None:
    main_test = test.set_index("baseline").loc[MAIN]
    main_cv = cv.set_index(["baseline", "cv"]).loc[MAIN, "mae"]
    target = main_test["mae"] * 0.8
    lines = [
        "# Phase 3 findings: baseline and validation design",
        "",
        "## 1. Train / test split (time holdout)",
        "",
        f"The newest {TEST_FRAC:.0%} of listings by date are the test set. Models never see them until final evaluation.",
        "",
        table(summary, ".3f"),
        "",
        "## 2. Spatial cross-validation",
        "",
        f"The {summary['map_points'].max()} map points are grouped into {N_CLUSTERS} location clusters with KMeans "
        f"(UTM meters, weighted by listings per point). The clusters are dealt into {N_FOLDS} folds, so every "
        "listing in a cluster lands in the same fold. Nearby listings are never split between train and validation.",
        "",
        table(folds, ".0f"),
        "",
        "## 3. Baselines",
        "",
        "- `global_median`: one number for every listing.",
        "- `beds_median`: median rent for the bedroom count.",
        "- `zip_beds_median` (the spec's baseline): median rent for the ZIP x bedroom combination, "
        "falling back to the bedroom median when that combination was not in training.",
        "",
        "### Cross-validation on the training set (mean over 5 folds, $)",
        "",
        table(cv),
        "",
        f"Random folds make `{MAIN}` look better than it is: random CV MAE ${main_cv['random']:.0f} vs spatial CV "
        f"${main_cv['spatial']:.0f}. Under random folds, listings from the same map point land in both train and "
        "validation, so the ZIP median has already seen near-copies. Spatial CV is the honest estimate for a "
        "neighborhood the model has not seen. It is the CV score used for all tuning from Phase 4 on.",
        "",
        "### What spatial CV and the test set each measure",
        "",
        f"- Under spatial CV, `{MAIN}` found its ZIP x beds cell for only {diag['cv_zip_level_share']:.0%} of "
        "validation listings. Holding out a whole cluster holds out its ZIPs too, so the baseline falls back to "
        "the bedroom median and scores about the same as `beds_median`. ZIP identity does not carry over to new "
        "areas. Phase 4 models need continuous location features (distances, tract and ZIP statistics) to beat it.",
        f"- The test set is a different question: {diag['test_seen_point_share']:.0%} of test listings sit on a map "
        f"point seen in training and {diag['test_seen_zip_share']:.0%} in a seen ZIP. It measures *later listings in "
        "mostly familiar places*. Spatial CV measures *new places*. Report both.",
        "- The test period skews suburban (see the median km to downtown above) and its median rent is lower, "
        "even with more 3+ bedroom units. Late-2019 listings are a shifted sample, not a random one.",
        "",
        "### Test set (fit on all training data, $)",
        "",
        table(test),
        "",
        "`mae_ci_low` / `mae_ci_high`: 95% bootstrap interval for test MAE (2,000 resamples).",
        "",
        f"**Baseline to beat: `{MAIN}` test MAE ${main_test['mae']:.0f}** "
        f"(95% CI ${main_test['mae_ci_low']:.0f}-${main_test['mae_ci_high']:.0f}), "
        f"MAPE {main_test['mape']:.1f}%. Phase 4 target (20% lower): **MAE <= ${target:.0f}**.",
        "",
        "How test listings were predicted by the main baseline:",
        "",
        table(levels, ".3f"),
        "",
        "Test MAE by bedrooms (main baseline):",
        "",
        table(by_beds),
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    if not FEATURES.exists():
        print("Missing data/processed/listings_features.parquet. Run: python -m src.phase2")
        return 1
    REPORTS.mkdir(exist_ok=True)

    print("[splits]")
    df, splits = load()
    train, test = df[~df["is_test"]], df[df["is_test"]]
    leaks = leaked_groups(train["cluster"], train["fold"])
    if leaks:
        raise RuntimeError(f"Clusters leak across folds: {sorted(leaks)}")
    summary = split_summary(df)
    print(summary.to_string(index=False))
    folds = (train.groupby("fold")
             .agg(listings=("id", "size"), clusters=("cluster", "nunique"), median_rent=(TARGET, "median"))
             .reset_index())

    print("[cross-validation]")
    cv = cv_table(train)
    print(cv.round(1).to_string(index=False))

    print("[test]")
    test_res, main_pred = test_table(train, test)
    print(test_res.round(1).to_string(index=False))
    levels = (main_pred["level"].map(LEVELS).value_counts(normalize=True)
              .rename_axis("prediction_source").rename("share").reset_index())
    by_beds = breakdown(main_pred, "bedrooms")

    diag = diagnostics(train, test)
    print({k: round(v, 3) for k, v in diag.items()})
    log_mlflow(cv, test_res, len(train), len(test))
    print(f"[mlflow] logged {len(test_res)} runs to experiment '{MLFLOW_EXPERIMENT}'")

    path = REPORTS / "phase3_findings.md"
    write_report(summary, cv, test_res, levels, by_beds, folds, diag, path)
    print(f"[report] {path.relative_to(ROOT)}")
    print(f"[splits] {SPLITS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
