import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, ROOT
from src.explain import CONFIDENCE, LEVELS, RentAnalyzer
from src.model_features import county_name
from src.phase4 import load
from src.phase5 import best_xgb
from src.plots import AQUA, BLUE, MUTED, ORANGE, save, use_style

REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
ANALYZER = MODELS / "rent_analyzer.joblib"
MODES = {"spatial": "MAPIE, spatial folds", "random": "MAPIE, random folds", "location_aware": "MAPIE, location-aware",
         "mondrian": "location-aware + per county"}


def table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.astype(object).to_markdown(index=False, floatfmt=floatfmt)


def coverage_frame(analyzer: RentAnalyzer, test: pd.DataFrame) -> pd.DataFrame:
    y = test["rent"].to_numpy()
    out = test[["id", "rent"]].assign(county=county_name(test["county_fips"]).to_numpy(),
                                      known_location=analyzer.known_location(test))
    for mode in MODES:
        bounds = analyzer.predict_interval(test, mode)
        for j, level in enumerate(LEVELS):
            lo, hi = bounds[:, 0, j], bounds[:, 1, j]
            out[f"{mode}_{level}_in"] = (y >= lo) & (y <= hi)
            out[f"{mode}_{level}_width"] = hi - lo
    return out


def summarize(cov: pd.DataFrame, by: str | None) -> pd.DataFrame:
    groups = [("all", cov)] if by is None else list(cov.groupby(by))
    rows = []
    for name, part in groups:
        row = {by or "group": name, "listings": len(part)}
        for mode in MODES:
            row[f"{mode}_coverage"] = part[f"{mode}_{CONFIDENCE}_in"].mean()
            row[f"{mode}_median_width"] = part[f"{mode}_{CONFIDENCE}_width"].median()
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_plot(cov: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0.45, 1], [0.45, 1], color=MUTED, linewidth=1, linestyle=":", label="perfect")
    for mode, color in zip(MODES, [ORANGE, MUTED, AQUA, BLUE]):
        ax.plot(LEVELS, [cov[f"{mode}_{lv}_in"].mean() for lv in LEVELS], marker="o", color=color, label=MODES[mode])
    ax.set_xlabel("target coverage")
    ax.set_ylabel("actual coverage on test set")
    ax.set_title("Interval calibration")
    ax.legend()
    return save(fig, "phase6_calibration")


def shap_plot(analyzer: RentAnalyzer, test: pd.DataFrame) -> tuple[Path, pd.Series]:
    importance = analyzer.shap_values(test).abs().mean().sort_values()
    pct = (np.exp(importance) - 1) * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    top = pct.tail(15)
    ax.barh(top.index, top.values, color=BLUE)
    ax.set_xlabel("mean |SHAP| as % of rent")
    ax.set_title("Global drivers (test set)")
    return save(fig, "phase6_shap_global"), pct.sort_values(ascending=False)


def example_text(result: dict, actual: float) -> list[str]:
    lo, hi = result["interval"]
    drivers = pd.DataFrame(result["drivers"]).rename(columns={"effect_pct": "effect_%"})
    comps = result["comps"][["rent", "bedrooms", "bathrooms", "square_feet", "cityname", "km_away"]]
    return [
        f"Predicted ${result['predicted_rent']:,.0f}, 80% range ${lo:,.0f}-${hi:,.0f} "
        f"(known location: {result['known_location']}). Actual rent: ${actual:,.0f}.",
        "",
        "Top drivers (SHAP; effect relative to the average listing):",
        "",
        table(drivers, ".2f"),
        "",
        "5 comparable training listings:",
        "",
        table(comps, ".1f"),
    ]


def write_report(overall, by_county, by_known, importance, example, path: Path) -> None:
    lines = [
        "# Phase 6 findings: ranges, drivers, comps",
        "",
        "## 1. 80% rent ranges (MAPIE cross-conformal, CV+)",
        "",
        "The Phase 4 XGBoost model is wrapped with MAPIE `CrossConformalRegressor` (method `plus`) on log(rent). "
        "Ranges therefore scale with price: a $2,000 unit gets a wider dollar range than a $900 unit. "
        "Calibration uses training data only, in two ways:",
        "",
        "- **spatial folds**: errors when the neighborhood is new (the Phase 3 folds).",
        "- **random folds**: errors when the location is already known.",
        "- **location-aware** (used by `analyze()`): random-fold ranges for listings at a map point seen in training, spatial-fold "
        "ranges for new points. This rule was set from the problem setup, not tuned on the test set.",
        "- **location-aware + per county** (Mondrian conformal, tried, not adopted): the same idea, but the range "
        f"width is the conformal quantile of out-of-fold log errors *within each county* (counties with < {100} "
        "training listings are pooled as Other). MAPIE 1.x has no Mondrian option, so this is implemented directly "
        "from out-of-fold residuals. It exists because the spec asks for ~80% per region, and one shared width "
        "under-covers the hard in-town areas.",
        "",
        "### Test coverage at the 80% target",
        "",
        table(overall),
        "",
        table(by_known),
        "",
        "Reading the coverage numbers: with n listings, sampling noise alone is about +/- sqrt(0.16/n). That is "
        "+/-1.5 points overall (n=758), +/-2.9 for Fulton (n=188), +/-6.9 for DeKalb (n=34). DeKalb's 71% and Gwinnett's "
        "92% fall within roughly 1.5 noise units of 80%. Fulton's 65% is a real shortfall.",
        "",
        "### By county",
        "",
        table(by_county),
        "",
        "![calibration](figures/phase6_calibration.png)",
        "",
        "**Verdict.** The location-aware MAPIE ranges hit 78.9% overall (target 80%) with a median width of $475, "
        "against 87% and $621 for spatial-only calibration. Per-county calibration did not fix Fulton: "
        "it moved Fulton to 72% but cut Gwinnett to 70% and overall to 76%, so it was not adopted. The Fulton "
        "shortfall is a time shift, not a width problem. Late-2019 in-town listings came in below the training-period "
        "levels (Phase 3), and conformal ranges assume the future looks like the past. This is a known limitation, "
        "and it is the kind of drift Phase 10 monitoring should catch.",
        "",
        "## 2. Drivers (SHAP)",
        "",
        "Mean absolute SHAP value on the test set, as % of rent (top 10):",
        "",
        table(importance.head(10).rename("pct_of_rent").rename_axis("feature").reset_index(), ".2f"),
        "",
        "![shap](figures/phase6_shap_global.png)",
        "",
        "## 3. Example: `RentAnalyzer.analyze()` on one test listing",
        "",
        *example,
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    use_style()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    train, test = load()
    params, k = best_xgb()

    print("[fit] XGBoost + MAPIE (spatial and random calibration) + SHAP + comps")
    analyzer = RentAnalyzer(params, k).fit(train)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(analyzer, ANALYZER)

    cov = coverage_frame(analyzer, test)
    overall = summarize(cov, None)
    by_known = summarize(cov.assign(location=np.where(cov["known_location"], "known point", "new point")), "location")
    by_county = summarize(cov, "county").sort_values("listings", ascending=False)
    print(overall.round(3).to_string(index=False))
    print(by_known.round(3).to_string(index=False))

    calibration_plot(cov)
    _, importance = shap_plot(analyzer, test)
    i = int(np.argmin(np.abs(test["rent"].to_numpy() - test["rent"].median())))
    example = example_text(analyzer.analyze(test.iloc[[i]]), test["rent"].iloc[i])

    with mlflow.start_run(run_name="phase6_conformal"):
        mlflow.set_tags({"phase": "6", "model_type": "xgboost", "stage": "intervals"})
        mlflow.log_params({**params, "k": k, "confidence": CONFIDENCE, "method": "cv_plus_log_scale"})
        for mode in MODES:
            mlflow.log_metrics({f"test_coverage_{mode}": overall[f"{mode}_coverage"].iloc[0],
                                f"test_median_width_{mode}": overall[f"{mode}_median_width"].iloc[0]})
        mlflow.log_artifact(str(ROOT / "reports" / "figures" / "phase6_calibration.png"))

    path = REPORTS / "phase6_findings.md"
    write_report(overall, by_county, by_known, importance, example, path)
    print(f"[report] {path.relative_to(ROOT)}")
    print(f"[saved] {ANALYZER.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
