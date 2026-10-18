import time

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

from src.forecast import GrowthForecast
from src.simulation import finance as fin
from src.simulation.monte_carlo import AR1, CONVENTIONAL, FHA, Deal, Market, run


def test_mortgage_payment_matches_known_value():
    assert fin.mortgage_payment(200_000, 0.06, 30) == pytest.approx(1199.10, abs=0.01)


def test_loan_balance_after_full_term_is_zero_and_after_zero_months_is_principal():
    assert fin.loan_balance(200_000, 0.06, 30, 0) == pytest.approx(200_000)
    assert fin.loan_balance(200_000, 0.06, 30, 360) == pytest.approx(0, abs=1e-6)
    assert fin.loan_balance(200_000, 0.06, 30, 12) == pytest.approx(197_543.98, abs=0.5)


def test_fha_loan_and_mip():
    base, total = fin.fha_loan(300_000)
    assert base == pytest.approx(289_500)
    assert total == pytest.approx(294_566.25)
    assert fin.fha_monthly_mip(base) == pytest.approx(132.69, abs=0.01)


def test_self_sufficiency_applies_to_three_and_four_units_only():
    assert fin.self_sufficiency(4 * 1500, 4400, 4)
    assert not fin.self_sufficiency(4 * 1500, 4600, 4)
    assert fin.self_sufficiency(2 * 1000, 5000, 2)


def test_dscr_cap_rate_cash_on_cash():
    assert fin.dscr(60_000, 50_000) == pytest.approx(1.2)
    assert fin.cap_rate(35_000, 500_000) == pytest.approx(0.07)
    assert fin.cash_on_cash(5_000, 100_000) == pytest.approx(0.05)


def test_irr_matches_scipy_root_and_handles_no_sign_change():
    flows = np.array([[-100, 10, 10, 110], [-1000, 300, 400, 500], [-100, -10, -10, -10]], dtype=float)
    result = fin.irr(flows)
    assert result[0] == pytest.approx(0.10, abs=1e-6)
    expected = brentq(lambda r: fin.npv(r, flows[1])[0], -0.5, 1)
    assert result[1] == pytest.approx(expected, abs=1e-6)
    assert np.isnan(result[2])


def toy_market() -> Market:
    growth = GrowthForecast("x", pd.Timestamp("2026-01-01"), 0.03, 0.02, 0.03, 0.6, 0.02)
    return Market(pd.Timestamp("2026-01-01"), growth, AR1(0.07, 0.8, 0.01, 0.07), AR1(0.025, 0.4, 0.01, 0.03),
                  0.065, 0.007)


def test_ten_thousand_paths_run_in_seconds_and_fha_is_more_leveraged():
    deal = Deal(600_000, 4, 1500, 1200, 1850)
    start = time.perf_counter()
    fha = run(deal, FHA, toy_market())
    conv = run(deal, CONVENTIONAL, toy_market())
    assert time.perf_counter() - start < 5
    assert fha["irr"].shape == (10_000,)
    assert fha["cash_invested"] < conv["cash_invested"]
    assert np.median(fha["noi_year1"]) < np.median(conv["noi_year1"])
    assert np.nanstd(fha["irr"]) > np.nanstd(conv["irr"])
