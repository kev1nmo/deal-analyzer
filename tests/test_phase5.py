import numpy as np
import pandas as pd
import pytest
import torch

from src.text_model import clean_text, listing_texts, mean_pool, text_target


def test_clean_text_masks_every_number():
    text = clean_text("$1,250 / Two BR - 1,100 sq ft", "Rent is $1,250.00 per month. Unit 6406, call 404-555-1234.")
    assert not any(ch.isdigit() for ch in text)
    assert "$" not in text
    assert "[num]" in text
    assert "Two BR" in text


def test_clean_text_handles_missing_parts():
    assert clean_text(None, "Nice unit") == ". Nice unit"
    assert clean_text("Title", None) == "Title."


def test_listing_texts_one_per_row():
    df = pd.DataFrame({"title": ["A 2 BR", None], "body": ["Rent 1200", "Pool"]})
    texts = listing_texts(df)
    assert len(texts) == 2
    assert all("1200" not in t and "2 BR" not in t for t in texts)


def test_text_target_is_log_rent_per_sqft():
    df = pd.DataFrame({"rent": [1000.0, 2000.0], "square_feet": [1000, 1000]})
    assert text_target(df) == pytest.approx(np.log([1.0, 2.0]))


def test_mean_pool_ignores_padding():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert mean_pool(hidden, mask).tolist() == [[2.0, 3.0]]
