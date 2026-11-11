from dataclasses import replace

import numpy as np
import pytest

from src.breakeven import (EQUITY_WIPED, base_scenario, breakevens, describe, evaluate, market_ranges,
                           solve_breakeven, tornado)
from src.forecast import GrowthForecast
from src.simulation.monte_carlo import AR1, CONVENTIONAL, FHA, Deal, Market

DEAL = Deal(price=660_000, units=4, rent=1_900, rent_low=1_450, rent_high=2_450)
MARKET = Market(as_of=None, growth=GrowthForecast("x", None, 0.03, 0.03, 0.03, 0.5, 0.02),
                vacancy=AR1(0.07, 0.8, 0.005, 0.07), inflation=AR1(0.025, 0.6, 0.01, 0.03), rate=0.068,
                rate_change_sd=0.008)


@pytest.fixture(scope="module")
def ranges():
    return market_ranges(DEAL, MARKET)


def test_ranges_are_ordered_and_rent_range_matches_conformal_band(ranges):
    for lo, mid, hi in ranges.values():
        assert lo <= mid <= hi
    lo, _, hi = ranges["rent"]
    assert lo == pytest.approx(DEAL.rent_low, rel=0.03) and hi == pytest.approx(DEAL.rent_high, rel=0.03)


def test_breakeven_root_zeroes_the_metric(ranges):
    base = base_scenario(ranges)
    for name in ["rent", "vacancy", "rate", "price"]:
        x = solve_breakeven(DEAL, CONVENTIONAL, base, name, "cash_flow_year1")
        assert x is not None
        assert evaluate(DEAL, CONVENTIONAL, replace(base, **{name: x}))["cash_flow_year1"] == pytest.approx(0, abs=1)


def test_breakeven_directions_make_economic_sense(ranges):
    base = base_scenario(ranges)
    be = breakevens(DEAL, CONVENTIONAL, ranges).set_index(["metric", "input"])["breakeven"]
    assert be[("cash_flow_year1", "rent")] < base.rent
    assert be[("cash_flow_year1", "vacancy")] > base.vacancy
    assert be[("cash_flow_year1", "rate")] > base.rate
    assert be[("cash_flow_year1", "price")] > base.price
    assert be[("irr", "rent")] < be[("cash_flow_year1", "rent")]


def test_no_breakeven_returns_none_and_says_so(ranges):
    base = base_scenario(ranges)
    assert solve_breakeven(DEAL, FHA, base, "vacancy", "cash_flow_year1") is None
    row = breakevens(DEAL, FHA, ranges).query("input == 'vacancy' and metric == 'cash_flow_year1'").iloc[0]
    assert describe(row) == "year-1 cash flow stays negative at any vacancy"


def test_tornado_is_sorted_by_swing_and_brackets_base(ranges):
    t = tornado(DEAL, CONVENTIONAL, ranges)
    assert list(t["swing"]) == sorted(t["swing"], reverse=True)
    rent = t.set_index("input").loc["rent"]
    assert rent["at_low"] < rent["base"] < rent["at_high"]


def test_irr_is_minus_100_percent_when_equity_is_wiped(ranges):
    base = base_scenario(ranges)
    crash = replace(base, growth=-0.15)
    assert evaluate(DEAL, FHA, crash)["irr"] == EQUITY_WIPED
    assert np.isfinite(evaluate(DEAL, CONVENTIONAL, base)["irr"])
