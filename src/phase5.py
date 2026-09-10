import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.baseline import TARGET, metrics
from src.config import (DATA_PROCESSED, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, ROOT, TEXT_HISTORY,
                        TEXT_MODEL_NAME, TEXT_SCORES)
from src.model_features import FEATURES
from src.models import build_xgb, cv_score, paired_bootstrap_gain
from src.phase4 import load

REPORTS = ROOT / "reports"
VARIANTS = {"no text (Phase 4)": None, "+ frozen MiniLM score": "text_frozen", "+ fine-tuned MiniLM score": "text_finetuned"}


def table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.astype(object).to_markdown(index=False, floatfmt=floatfmt)


def log_psf(df: pd.DataFrame) -> np.ndarray:
    return np.log(df[TARGET].to_numpy() / df["square_feet"].to_numpy())


def text_quality(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, part in [("train (out-of-fold)", train), ("test", test)]:
        y = log_psf(part)
        rows.append({"set": name, "model": "predict the mean", "mae_log_psf": np.abs(y - log_psf(train).mean()).mean(),
                     "spearman": np.nan})
        for col, label in [("text_frozen", "frozen MiniLM + ridge"), ("text_finetuned", "fine-tuned MiniLM")]:
            rows.append({"set": name, "model": label, "mae_log_psf": np.abs(y - part[col]).mean(),
                         "spearman": spearmanr(y, part[col])[0]})
    return pd.DataFrame(rows)


def best_xgb() -> tuple[dict, int]:
    run = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT],
                             filter_string="tags.mlflow.runName = 'xgb_optuna'").iloc[0]
    params = {k[len("params.best_"):]: v for k, v in run.items() if k.startswith("params.best_") and pd.notna(v)}
    k = int(params.pop("k"))
    typed = {}
    for name, value in params.items():
        if name == "objective":
            typed[name] = value
        elif name in ("n_estimators", "max_depth"):
            typed[name] = int(float(value))
        else:
            typed[name] = float(value)
    return typed, k


def ablation(train: pd.DataFrame, test: pd.DataFrame, params: dict, k: int) -> tuple[pd.DataFrame, dict]:
    rows, preds = [], {}
    for label, col in VARIANTS.items():
        features = FEATURES + ([col] if col else [])
        model = build_xgb(params, k, features)
        cv = cv_score(model, train, train[TARGET], train["fold"])
        pred = model.fit(train, train[TARGET]).predict(test)
        preds[label] = pred
        test_m = metrics(test[TARGET], pred)
        with mlflow.start_run(run_name=f"phase5_xgb_{col or 'no_text'}"):
            mlflow.set_tags({"phase": "5", "model_type": "xgboost", "stage": "ablation"})
            mlflow.log_params({**params, "k": k, "text_feature": col or "none", "text_model": TEXT_MODEL_NAME})
            mlflow.log_metrics({**cv, **{f"test_{m}": v for m, v in test_m.items()}})
        rows.append({"variant": label, "cv_mae": cv["cv_mae"], "test_mae": test_m["mae"], "test_mape": test_m["mape"]})
    out = pd.DataFrame(rows)
    base = preds["no text (Phase 4)"]
    gains = [paired_bootstrap_gain(test[TARGET], base, preds[label]) for label in VARIANTS]
    out["test_gain_vs_no_text_pct"] = [g[0] * 100 for g in gains]
    out["gain_ci_low"] = [g[1] * 100 for g in gains]
    out["gain_ci_high"] = [g[2] * 100 for g in gains]
    return out, preds


def write_report(quality: pd.DataFrame, abl: pd.DataFrame, history: list[dict], path: Path) -> None:
    tuned = abl.set_index("variant").loc["+ fine-tuned MiniLM score"]
    lines = [
        "# Phase 5 findings: text feature from listing descriptions",
        "",
        "## 1. Approach",
        "",
        f"- Text: listing title + description, with **every number replaced by `[num]`**. 15% of descriptions state "
        "the exact rent, and many give square footage or prices, so leaving numbers in would leak the answer.",
        f"- Model: `{TEXT_MODEL_NAME}` (22M parameters), fine-tuned in PyTorch with a mean-pooling + linear head to predict "
        "log(rent per sq ft). Huber loss, AdamW (encoder lr 3e-5, head lr 1e-3), linear warmup and decay, "
        "3 epochs, max 160 tokens, Apple GPU (MPS).",
        "- Comparison: the same model frozen (no training). Its embeddings go into a ridge regression with the "
        "penalty picked by internal CV.",
        "- No leakage: a training listing's text score comes from a model fit on the *other* 4 spatial folds. "
        "Test scores come from a model fit on all training data. The score is one extra column for XGBoost. "
        "XGBoost keeps its Phase 4 settings, so the ablation changes only the feature.",
        "",
        "## 2. How well does text alone predict rent per sq ft?",
        "",
        table(quality),
        "",
        "`mae_log_psf` is on the log scale: 0.17 means the typical miss is about 17% of rent per sq ft.",
        "",
        "## 3. Ablation: rent model with vs without the text feature",
        "",
        table(abl.round(3), ".1f"),
        "",
        "`test_gain_vs_no_text_pct`: % MAE reduction vs the Phase 4 model, with a 95% paired bootstrap CI.",
        "",
        f"**Result:** the fine-tuned text score changes test MAE from ${abl.iloc[0]['test_mae']:.0f} to "
        f"${tuned['test_mae']:.0f} ({tuned['test_gain_vs_no_text_pct']:.1f}%, 95% CI {tuned['gain_ci_low']:.1f}% to "
        f"{tuned['gain_ci_high']:.1f}%).",
        "",
        "Caveat on the CV column: a training listing's text score came from a model that saw the other folds, "
        "including whichever fold is the XGBoost validation fold at that moment. That is a small second-order leak, "
        "so the CV gain is slightly optimistic. The test column has no such leak and is the headline.",
        "",
        "## 4. Fine-tuning loss (normalized Huber loss per epoch)",
        "",
        table(pd.DataFrame([{"fit": h["fit"], **{f"epoch_{i + 1}": v for i, v in enumerate(h["loss_by_epoch"])}}
                            for h in history])),
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5: text feature + ablation")
    parser.add_argument("--retrain", action="store_true", help="refit the text models even if scores are cached")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    train, test = load()

    if args.retrain or not TEXT_SCORES.exists():
        subprocess.run([sys.executable, "-m", "src.text_scores"], check=True, cwd=ROOT)
    else:
        print(f"[text] using cached {TEXT_SCORES.relative_to(ROOT)} (pass --retrain to refit)")
    scores = pd.read_parquet(TEXT_SCORES)
    history = [{**h, "loss_by_epoch": list(h["loss_by_epoch"])} for h in pd.read_parquet(TEXT_HISTORY).to_dict("records")]

    train = train.merge(scores, on="id", how="left", validate="one_to_one")
    test = test.merge(scores, on="id", how="left", validate="one_to_one")
    quality = text_quality(train, test)
    print(quality.round(3).to_string(index=False))

    print("[ablation] XGBoost with Phase 4 settings")
    params, k = best_xgb()
    abl, preds = ablation(train, test, params, k)
    print(abl.round(1).to_string(index=False))

    path = REPORTS / "phase5_findings.md"
    write_report(quality, abl, history, path)
    test[["id", TARGET]].assign(**{f"pred_{c or 'no_text'}": preds[l] for l, c in VARIANTS.items()}).to_parquet(
        DATA_PROCESSED / "phase5_test_predictions.parquet", index=False)
    print(f"[report] {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
