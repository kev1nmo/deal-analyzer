import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, MLFLOW_TRACKING_URI, RANDOM_SEED, ROOT
from src.forecast.rent_growth import (CANDIDATES, HORIZON, METHODS, SARIMA_ORDER, SARIMA_SEASONAL, XGB_PARAMS,
                                      backtest, build_features, choose_method, load_fred, load_zori, macro_frame,
                                      recent_backtest)
from src.plots import AQUA, BLUE, INK, MUTED, ORANGE, save, use_style

EXPERIMENT = "rent-growth"
ORIGINS = list(pd.date_range("2021-12-01", "2023-12-01", freq="3MS"))
BACKTEST = DATA_PROCESSED / "phase7_backtest.parquet"
LABELS = {"no_change": "no change (0%)", "persistence": "last 12 months repeat", "sarima": "SARIMA per ZIP",
          "xgboost": "XGBoost, lag features", "auto": "auto: best method of the last 2 years"}
COLUMNS = [*METHODS, "auto"]


def table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    return df.astype(object).to_markdown(index=False, floatfmt=floatfmt)


def abs_errors(bt: pd.DataFrame) -> pd.DataFrame:
    actual = np.expm1(bt["actual"].astype(float))
    return pd.DataFrame({m: np.abs(np.expm1(bt[m].astype(float)) - actual) * 100 for m in COLUMNS})


def bootstrap_diff(bt: pd.DataFrame, a: str, b: str, n: int = 2000) -> tuple[float, float, float]:
    err = abs_errors(bt).assign(zip=bt["zip"].to_numpy())
    by_zip = err.groupby("zip")[[a, b]].agg(["sum", "count"])
    sums_a, sums_b, counts = by_zip[(a, "sum")].to_numpy(), by_zip[(b, "sum")].to_numpy(), by_zip[(a, "count")].to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.integers(0, len(counts), (n, len(counts)))
    diffs = (sums_a[idx].sum(1) - sums_b[idx].sum(1)) / counts[idx].sum(1)
    point = (sums_a.sum() - sums_b.sum()) / counts.sum()
    return float(point), *np.percentile(diffs, [2.5, 97.5])


def mae_table(bt: pd.DataFrame, by: str | None) -> pd.DataFrame:
    err = abs_errors(bt)
    if by is None:
        return pd.DataFrame([{"forecasts": len(err), **err.mean().to_dict()}])
    keys = bt[by].to_numpy()
    out = err.groupby(keys).mean().assign(forecasts=err.groupby(keys).size())
    return out.rename_axis(by).reset_index()[[by, "forecasts", *COLUMNS]]


def plot_by_origin(bt: pd.DataFrame) -> None:
    err = abs_errors(bt).groupby(bt["origin"].to_numpy()).mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, color in zip(COLUMNS, [MUTED, ORANGE, AQUA, BLUE, INK]):
        ax.plot(err.index, err[m], marker="o", color=color, label=LABELS[m])
    ax.set_ylabel("MAE, percentage points")
    ax.set_xlabel("forecast made in (12 months ahead)")
    ax.set_title("12-month ZIP rent growth: backtest error")
    ax.legend()
    save(fig, "phase7_backtest_mae")


def add_auto(logz: pd.DataFrame, features: pd.DataFrame, bt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bt = bt.copy()
    bt["auto"] = np.nan
    picks = []
    for origin in ORIGINS:
        chosen = choose_method(recent_backtest(logz, features, origin, CANDIDATES), CANDIDATES)
        rows = bt["origin"] == origin
        bt.loc[rows, "auto"] = bt.loc[rows, chosen].astype(float)
        picks.append({"origin": f"{origin:%b %Y}", "picked": LABELS[chosen]})
    return bt, pd.DataFrame(picks)


def write_report(bt, overall, by_year, diff, picks, today_pick) -> None:
    point, lo, hi = diff
    actual = np.expm1(bt["actual"].astype(float)) * 100
    lines = [
        "# Phase 7 findings: rent-growth forecast",
        "",
        f"Task: forecast each ZIP's rent growth over the next {HORIZON} months, using Zillow ZORI by ZIP. Forecasts were "
        f"made every quarter from {ORIGINS[0]:%b %Y} to {ORIGINS[-1]:%b %Y}, so the forecast years cover 2022-2024. "
        f"Each forecast uses only data up to its start month. That gives {len(bt):,} forecasts over "
        f"{bt['zip'].nunique()} ZIPs with at least 3 years of history. Actual 12-month growth averaged "
        f"{actual.mean():.1f}% (range {actual.min():.1f}% to {actual.max():.1f}%).",
        "",
        "Methods:",
        "",
        f"- **SARIMA** (statsmodels): one model per ZIP on log rent, order {SARIMA_ORDER} x seasonal {SARIMA_SEASONAL}, refit at each origin.",
        "- **XGBoost**: one model for all ZIPs, refit at each origin. Features: past 1/3/6/12/24-month growth, metro "
        "growth, the ZIP's rent level relative to the metro, the 30-year mortgage rate and its 12-month change, "
        "Atlanta CPI rent growth (lagged a month), and calendar month. Hyperparameters were fixed up front, not tuned: "
        f"{XGB_PARAMS}.",
        "- Two simple reference points: no change (0%), and \"the last 12 months repeat\".",
        "",
        "## Backtest MAE (percentage points of 12-month growth)",
        "",
        table(overall),
        "",
        "By forecast year (the year the forecast was made in):",
        "",
        table(by_year),
        "",
        f"XGBoost minus SARIMA MAE: {point:+.2f} pp (95% CI {lo:+.2f} to {hi:+.2f}; bootstrap over ZIPs).",
        "",
        "![backtest](figures/phase7_backtest_mae.png)",
        "",
        f"**XGBoost vs. SARIMA** (the comparison the spec asks for): XGBoost wins clearly. **But the plain \"no change\" "
        f"forecast ({overall['no_change'].iloc[0]:.2f} pp) beat both models.** Both models learned from 2016-2021, "
        "when Atlanta rents rose every year, and kept predicting growth through the 2022-2024 flattening. Tree models "
        "cannot predict outside the range they were trained on.",
        "",
        "## Choosing what to ship",
        "",
        "Three candidates were judged on two tests: the 2022-2024 backtest above (a flat market) and the Phase 8 2019 "
        "backtest (a boom). The second test is the share of ZIPs whose actual 2019-2024 rent growth fell inside the "
        "simulated 80% band.",
        "",
        "| Option | 2022-24 MAE (pp) | 2019 backtest: ZIPs inside 80% band |",
        "|---|---|---|",
        f"| No change | {overall['no_change'].iloc[0]:.2f} | 2% |",
        f"| XGBoost | {overall['xgboost'].iloc[0]:.2f} | 44% |",
        f"| Auto-select (best method of the prior 2 years) | {overall['auto'].iloc[0]:.2f} | 44% (it picks XGBoost) |",
        "",
        "- **No change** wins only in a flat market. Started in Dec 2018, it would have predicted flat rents into the "
        "2019-2021 boom, and nearly every ZIP ended above its range.",
        "- **Auto-select** scores no-change, persistence and XGBoost on the 2 years before each forecast date and uses "
        "the best one, with no look-ahead. It did worse than XGBoost alone, because the market turned faster than a "
        "2-year lookback can notice: in early 2022 it picked \"last 12 months repeat\" right as the boom ended.",
        "",
        table(picks),
        "",
        "- **XGBoost (shipped)** is the only option that holds up in both a flat market and a boom. Auto-select picks "
        f"it today ({today_pick[0]:%b %Y}) and in Dec 2018 as well.",
        "",
        "Lesson: in a regime change, the year-1 point forecast is unreliable whichever method makes it. What protects "
        "the deal analysis is a realistic uncertainty band (Phase 8) and break-evens (Phase 11), not a cleverer forecast.",
        "",
        "It is exposed as `src.forecast.forecast_growth(zip, as_of)`. That returns the method used, the year-1 "
        "forecast, its error sd (from the same 2-year window), and a mean-reverting AR(1) for later years. The AR(1) is "
        "fit to annual Atlanta CPI rent growth (1999 onward, up to `as_of`). Phase 8 draws rent-growth paths from it.",
        "",
        "Limitations: ZORI is smoothed and Zillow revises its history, so the backtest uses today's version of past data. "
        "2022 was an unusual year: rents decelerated sharply after the 2021 spike, which no method saw coming.",
        "",
    ]
    (ROOT / "reports" / "phase7_findings.md").write_text("\n".join(lines))


def main() -> int:
    use_style()
    logz, fred = load_zori(), load_fred()
    features = build_features(logz, macro_frame(fred))
    if BACKTEST.exists() and "--rerun" not in sys.argv:
        bt = pd.read_parquet(BACKTEST)[["origin", "zip", "actual", *METHODS]]
        print(f"[backtest] cached {BACKTEST.name} (--rerun to refit SARIMA, ~4 min)")
    else:
        print(f"[backtest] {len(ORIGINS)} origins, methods {METHODS}")
        bt = backtest(logz, features, ORIGINS)
    bt, picks = add_auto(logz, features, bt)
    bt.to_parquet(BACKTEST, index=False)
    today = logz.index[-1]
    today_pick = (today, choose_method(recent_backtest(logz, features, today, CANDIDATES), CANDIDATES))

    overall = mae_table(bt, None)
    by_year = mae_table(bt.assign(year=bt["origin"].dt.year), "year")
    diff = bootstrap_diff(bt, "xgboost", "sarima")
    print(overall.round(2).to_string(index=False))
    print(by_year.round(2).to_string(index=False))
    print(picks.to_string(index=False))
    print(f"xgboost - sarima: {diff}; today picks {today_pick[1]}")

    plot_by_origin(bt)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="phase7_backtest"):
        mlflow.set_tags({"phase": "7", "shipped": "auto", "today_pick": today_pick[1]})
        mlflow.log_params({"horizon": HORIZON, "origins": len(ORIGINS), "sarima_order": str(SARIMA_ORDER),
                           "sarima_seasonal": str(SARIMA_SEASONAL), "auto_candidates": ",".join(CANDIDATES),
                           **{f"xgb_{k}": v for k, v in XGB_PARAMS.items()}})
        mlflow.log_metrics({f"backtest_mae_pp_{m}": float(overall[m].iloc[0]) for m in COLUMNS})
        mlflow.log_artifact(str(ROOT / "reports" / "figures" / "phase7_backtest_mae.png"))

    write_report(bt, overall, by_year, diff, picks, today_pick)
    print("[report] reports/phase7_findings.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
