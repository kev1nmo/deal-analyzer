import re

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from src.config import RANDOM_SEED, TEXT_MODEL_NAME

MODEL_NAME = TEXT_MODEL_NAME
MAX_LEN = 160
NUMBER = re.compile(r"\$?\d[\d,.]*")


def clean_text(title: str | None, body: str | None) -> str:
    text = f"{title or ''}. {body or ''}"
    return re.sub(r"\s+", " ", NUMBER.sub(" [num] ", text)).strip()


def listing_texts(df: pd.DataFrame) -> list[str]:
    return [clean_text(t, b) for t, b in zip(df["title"], df["body"])]


def text_target(df: pd.DataFrame) -> np.ndarray:
    return np.log(df["rent"].to_numpy() / df["square_feet"].to_numpy())


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class TextRegressor(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.encoder.config.hidden_size, 1))

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return mean_pool(out.last_hidden_state, attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids, attention_mask)).squeeze(-1)


def tokenize(texts: list[str], tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


class FineTunedText:
    def __init__(self, epochs: int = 3, lr: float = 3e-5, head_lr: float = 1e-3, batch_size: int = 32,
                 seed: int = RANDOM_SEED):
        self.epochs, self.lr, self.head_lr, self.batch_size, self.seed = epochs, lr, head_lr, batch_size, seed

    def fit(self, texts: list[str], y: np.ndarray) -> "FineTunedText":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device_ = device()
        self.tokenizer_ = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model_ = TextRegressor().to(self.device_)
        self.y_mean_, self.y_std_ = float(np.mean(y)), float(np.std(y))
        ids, mask = tokenize(texts, self.tokenizer_)
        target = torch.tensor((y - self.y_mean_) / self.y_std_, dtype=torch.float32)
        loader = DataLoader(TensorDataset(ids, mask, target), batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.seed))
        opt = torch.optim.AdamW([
            {"params": self.model_.encoder.parameters(), "lr": self.lr},
            {"params": self.model_.head.parameters(), "lr": self.head_lr},
        ], weight_decay=0.01)
        steps = self.epochs * len(loader)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, 1 - s / steps))
        loss_fn = nn.HuberLoss(delta=1.0)
        self.history_ = []
        self.model_.train()
        for _ in range(self.epochs):
            total = 0.0
            for b_ids, b_mask, b_y in loader:
                b_ids, b_mask, b_y = b_ids.to(self.device_), b_mask.to(self.device_), b_y.to(self.device_)
                opt.zero_grad()
                loss = loss_fn(self.model_(b_ids, b_mask), b_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                sched.step()
                total += loss.item() * len(b_y)
            self.history_.append(total / len(target))
        return self

    @torch.no_grad()
    def predict(self, texts: list[str]) -> np.ndarray:
        self.model_.eval()
        ids, mask = tokenize(texts, self.tokenizer_)
        out = []
        for i in range(0, len(ids), 128):
            out.append(self.model_(ids[i:i + 128].to(self.device_), mask[i:i + 128].to(self.device_)).cpu())
        return torch.cat(out).numpy() * self.y_std_ + self.y_mean_


@torch.no_grad()
def frozen_embeddings(texts: list[str], batch_size: int = 128) -> np.ndarray:
    dev = device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(dev).eval()
    ids, mask = tokenize(texts, tokenizer)
    out = []
    for i in range(0, len(ids), batch_size):
        b_ids, b_mask = ids[i:i + batch_size].to(dev), mask[i:i + batch_size].to(dev)
        out.append(mean_pool(encoder(input_ids=b_ids, attention_mask=b_mask).last_hidden_state, b_mask).cpu())
    return torch.cat(out).numpy()
