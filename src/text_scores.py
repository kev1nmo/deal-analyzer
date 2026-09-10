import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from src.config import DATA_PROCESSED, ROOT, TEXT_HISTORY, TEXT_SCORES
from src.text_model import FineTunedText, frozen_embeddings, listing_texts, text_target

RIDGE_ALPHAS = np.logspace(-2, 3, 12)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(DATA_PROCESSED / "listings_features.parquet").merge(
        pd.read_parquet(DATA_PROCESSED / "splits.parquet"), on="id", validate="one_to_one")
    return df[~df["is_test"]].reset_index(drop=True), df[df["is_test"]].reset_index(drop=True)


def out_of_fold_scores(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    tr_text, te_text = listing_texts(train), listing_texts(test)
    y = text_target(train)
    folds = train["fold"].to_numpy()
    emb_tr, emb_te = frozen_embeddings(tr_text), frozen_embeddings(te_text)

    frozen_tr, tuned_tr = np.full(len(train), np.nan), np.full(len(train), np.nan)
    history = []
    for fold in sorted(np.unique(folds)):
        val = folds == fold
        frozen_tr[val] = RidgeCV(alphas=RIDGE_ALPHAS).fit(emb_tr[~val], y[~val]).predict(emb_tr[val])
        model = FineTunedText().fit([t for t, v in zip(tr_text, val) if not v], y[~val])
        tuned_tr[val] = model.predict([t for t, v in zip(tr_text, val) if v])
        history.append({"fit": f"fold {fold} held out", "loss_by_epoch": model.history_})
        print(f"  fold {fold}: train loss by epoch {np.round(model.history_, 3).tolist()}", flush=True)

    frozen_te = RidgeCV(alphas=RIDGE_ALPHAS).fit(emb_tr, y).predict(emb_te)
    model = FineTunedText().fit(tr_text, y)
    tuned_te = model.predict(te_text)
    history.append({"fit": "all training data", "loss_by_epoch": model.history_})

    scores = pd.concat([
        pd.DataFrame({"id": train["id"], "text_frozen": frozen_tr, "text_finetuned": tuned_tr}),
        pd.DataFrame({"id": test["id"], "text_frozen": frozen_te, "text_finetuned": tuned_te}),
    ], ignore_index=True)
    return scores, history


def main() -> int:
    train, test = load()
    print("[text] fine-tuning MiniLM: 5 out-of-fold fits + 1 full fit", flush=True)
    scores, history = out_of_fold_scores(train, test)
    scores.to_parquet(TEXT_SCORES, index=False)
    pd.DataFrame(history).to_parquet(TEXT_HISTORY, index=False)
    print(f"[text] saved {TEXT_SCORES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
