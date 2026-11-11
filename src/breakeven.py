from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.simulation.monte_carlo import Deal, Drivers, Financing, Market, draw_drivers, simulate

N_RANGE_PATHS = 10_000
EQUITY_WIPED = -1.0
PRICE_SWING = 0.10
RATE_SWING = 0.01
LABELS = {
    "rent": "Rent per unit ($/month)",
    "vacancy": "Vacancy (avg/yr)",
    "growth": "Rent growth (avg/yr)",
    "inflation": "Expense inflation (avg/yr)",
    "capex": "Capex ($/yr)",
    "exit_rate_change": "Rate change by exit (pp)",
    "rate": "Mortgage rate at purchase",
    "price": "Purchase price ($)",
}
SOLVE_BOUNDS = {
    "rent": (100.0, 10_000.0),
    "vacancy": (0.0, 0.9),
    "rate": (0.001, 0.25),
    "price": (20_000.0, 20_000_000.0),
    "growth": (-0.3, 0.3),
}
HIGHER_HURTS = {"vacancy", "inflation", "capex", "exit_rate_change", "rate", "price"}
BREAKEVENS = [("cash_flow_year1", ["rent", "vacancy", "rate", "price"]), ("irr", ["growth", "rent"])]


@dataclass(frozen=True)
class Scenario:
    rent: float
    vacancy: float
    growth: float
    inflation: float
    capex: float
    exit_rate_change: float
    rate: float
    price: float


def market_ranges(deal: Deal, market: Market, seed: int = 0) -> dict[str, tuple[float, float, float]]:
    d = draw_drivers(deal, market, N_RANGE_PATHS, np.random.default_rng(seed))
    per_path = {
        "rent": d.rent,
        "vacancy": d.vacancy.mean(axis=1),
        "growth": d.growth.mean(axis=1),
        "inflation": d.inflation.mean(axis=1),
        "capex": d.capex.mean(axis=1),
        "exit_rate_change": d.rate[:, -1] - d.rate_start,
    }
    out = {k: tuple(float(x) for x in np.percentile(v, [10, 50, 90])) for k, v in per_path.items()}
    out["rate"] = (market.rate - RATE_SWING, market.rate, market.rate + RATE_SWING)
    out["price"] = (deal.price * (1 - PRICE_SWING), deal.price, deal.price * (1 + PRICE_SWING))
    return out


def base_scenario(ranges: dict[str, tuple[float, float, float]]) -> Scenario:
    return Scenario(**{k: v[1] for k, v in ranges.items()})


def evaluate(deal: Deal, financing: Financing, s: Scenario) -> dict[str, float]:
    def flat(value: float) -> np.ndarray:
        return np.full((1, deal.years), value, dtype=float)

    drivers = Drivers(rent=np.array([s.rent]), growth=flat(s.growth), vacancy=flat(s.vacancy),
                      inflation=flat(s.inflation), rate=flat(s.rate + s.exit_rate_change), capex=flat(s.capex),
                      rate_start=s.rate)
    res = simulate(replace(deal, price=s.price, rent=s.rent), financing, drivers)
    irr = float(res["irr"][0])
    return {"cash_flow_year1": float(res["cash_on_cash"][0] * res["cash_invested"]),
            "irr": EQUITY_WIPED if np.isnan(irr) else irr}


def tornado(deal: Deal, financing: Financing, ranges: dict, metric: str = "irr") -> pd.DataFrame:
    base = base_scenario(ranges)
    base_value = evaluate(deal, financing, base)[metric]
    rows = []
    for name, (lo, _, hi) in ranges.items():
        low = evaluate(deal, financing, replace(base, **{name: lo}))[metric]
        high = evaluate(deal, financing, replace(base, **{name: hi}))[metric]
        rows.append({"input": name, "label": LABELS[name], "low_input": lo, "high_input": hi,
                     "at_low": low, "at_high": high, "base": base_value, "swing": abs(high - low)})
    return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)


def solve_breakeven(deal: Deal, financing: Financing, base: Scenario, name: str, metric: str,
                    target: float = 0.0) -> float | None:
    def gap(x: float) -> float:
        value = evaluate(deal, financing, replace(base, **{name: x}))[metric]
        return value - target

    lo, hi = SOLVE_BOUNDS[name]
    grid = np.linspace(lo, hi, 41)
    values = np.array([gap(x) for x in grid])
    ok = np.isfinite(values)
    grid, values = grid[ok], values[ok]
    signs = np.sign(values)
    change = np.nonzero(signs[:-1] * signs[1:] < 0)[0]
    if len(change) == 0:
        return None
    current = getattr(base, name)
    i = change[np.argmin(np.abs(grid[change] - current))]
    return float(brentq(gap, grid[i], grid[i + 1], xtol=1e-6 * max(1.0, abs(grid[i]))))


def breakevens(deal: Deal, financing: Financing, ranges: dict) -> pd.DataFrame:
    base = base_scenario(ranges)
    base_values = evaluate(deal, financing, base)
    rows = []
    for metric, names in BREAKEVENS:
        for name in names:
            value = solve_breakeven(deal, financing, base, name, metric)
            rows.append({"metric": metric, "input": name, "label": LABELS[name], "base_input": getattr(base, name),
                         "base_metric": base_values[metric], "breakeven": value})
    return pd.DataFrame(rows)


def describe(row: pd.Series) -> str:
    name, metric = row["input"], row["metric"]
    what = "year-1 cash flow" if metric == "cash_flow_year1" else "IRR"
    label = row["label"].split(" (")[0].lower()
    if row["breakeven"] is None or pd.isna(row["breakeven"]):
        return f"{what} stays {'positive' if row['base_metric'] >= 0 else 'negative'} at any {label}"
    higher_hurts = name in HIGHER_HURTS
    if row["base_metric"] >= 0:
        verb, side = "turns negative", "above" if higher_hurts else "below"
    else:
        verb, side = "turns positive", "below" if higher_hurts else "above"
    return f"{what} {verb} {side} {fmt(name, row['breakeven'])} {label} (now {fmt(name, row['base_input'])})"


def fmt(name: str, value: float) -> str:
    if name in ("vacancy", "growth", "inflation", "rate"):
        return f"{value:.1%}"
    if name == "exit_rate_change":
        return f"{value * 100:+.1f} pp"
    return f"${value:,.0f}"
