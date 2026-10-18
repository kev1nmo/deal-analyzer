from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.forecast import GrowthForecast
from src.forecast.rent_growth import load_fred, monthly
from src.simulation import finance as fin

Z80 = 1.2816
HISTORY_YEARS = 30
MIN_EXIT_CAP = 0.03


@dataclass(frozen=True)
class Deal:
    price: float
    units: int
    rent: float
    rent_low: float
    rent_high: float
    years: int = 7
    tax_rate: float = 0.013
    insurance_per_unit: float = 1500.0
    maintenance_pct: float = 0.08
    management_pct: float = 0.08
    capex_events_per_unit: float = 0.25
    capex_median: float = 3500.0
    capex_sigma: float = 0.8
    closing_cost_pct: float = 0.03
    selling_cost_pct: float = 0.06
    exit_cap_beta: float = 0.5


@dataclass(frozen=True)
class Financing:
    name: str
    down_pct: float
    rate_premium: float = 0.0
    upfront_mi: float = 0.0
    annual_mi: float = 0.0
    owner_years: int = 0
    term_years: int = 30


FHA = Financing("FHA house hack", 0.035, 0.0, fin.FHA_UPFRONT_MIP, fin.FHA_ANNUAL_MIP, owner_years=1)
CONVENTIONAL = Financing("Conventional 25% down", 0.25, 0.0075)


@dataclass(frozen=True)
class AR1:
    mean: float
    phi: float
    sd: float
    start: float

    def paths(self, years: int, n: int, rng: np.random.Generator, low: float = -np.inf,
              high: float = np.inf) -> np.ndarray:
        out = np.empty((n, years))
        prev = np.full(n, self.start)
        for t in range(years):
            prev = np.clip(self.mean + self.phi * (prev - self.mean) + self.sd * rng.standard_normal(n), low, high)
            out[:, t] = prev
        return out


def fit_ar1(values: pd.Series) -> AR1:
    v = values.to_numpy(dtype=float)
    mean = float(v.mean())
    phi = float(np.clip(np.polyfit(v[:-1] - mean, v[1:] - mean, 1)[0], 0.0, 0.95))
    resid = (v[1:] - mean) - phi * (v[:-1] - mean)
    return AR1(mean, phi, float(resid.std(ddof=2)), float(v[-1]))


def annual(fred: pd.DataFrame, name: str, as_of: pd.Timestamp, years: int = HISTORY_YEARS) -> pd.Series:
    s = fred.loc[(fred["name"] == name) & (fred["date"] <= as_of + pd.offsets.MonthEnd(0))]
    yearly = s.groupby(s["date"].dt.year)["value"].mean()
    last_full = as_of.year if as_of.month == 12 else as_of.year - 1
    return yearly.loc[last_full - years + 1:last_full]


@dataclass(frozen=True)
class Market:
    as_of: pd.Timestamp
    growth: GrowthForecast
    vacancy: AR1
    inflation: AR1
    rate: float
    rate_change_sd: float


def fit_market(growth: GrowthForecast, fred: pd.DataFrame | None = None) -> Market:
    fred = load_fred() if fred is None else fred
    as_of = growth.as_of
    vacancy = annual(fred, "us_rental_vacancy", as_of) / 100
    inflation = np.log(annual(fred, "us_cpi", as_of, HISTORY_YEARS + 1)).diff().dropna()
    rates = annual(fred, "mortgage_30y", as_of) / 100
    rate_now = monthly(fred, "mortgage_30y").loc[:as_of].dropna().iloc[-1] / 100
    return Market(as_of, growth, fit_ar1(vacancy), fit_ar1(inflation), float(rate_now), float(rates.diff().std()))


@dataclass(frozen=True)
class Drivers:
    rent: np.ndarray
    growth: np.ndarray
    vacancy: np.ndarray
    inflation: np.ndarray
    rate: np.ndarray
    capex: np.ndarray
    rate_start: float


def draw_drivers(deal: Deal, market: Market, n: int, rng: np.random.Generator) -> Drivers:
    t = deal.years
    sigma = np.log(deal.rent_high / deal.rent_low) / (2 * Z80)
    rent = deal.rent * np.exp(sigma * rng.standard_normal(n))
    events = rng.poisson(deal.capex_events_per_unit * deal.units, (n, t))
    capex = np.zeros((n, t))
    for k in range(int(events.max())):
        capex += rng.lognormal(np.log(deal.capex_median), deal.capex_sigma, (n, t)) * (events > k)
    rate = market.rate + np.cumsum(market.rate_change_sd * rng.standard_normal((n, t)), axis=1)
    return Drivers(
        rent=rent,
        growth=market.growth.paths(t, n, rng),
        vacancy=market.vacancy.paths(t, n, rng, 0.02, 0.25),
        inflation=np.expm1(market.inflation.paths(t, n, rng)),
        rate=np.maximum(rate, 0.02),
        capex=capex,
        rate_start=market.rate,
    )


def expected_capex(deal: Deal) -> float:
    return deal.capex_events_per_unit * deal.units * deal.capex_median * np.exp(deal.capex_sigma ** 2 / 2)


def simulate(deal: Deal, financing: Financing, d: Drivers) -> dict[str, np.ndarray | float | bool]:
    t = deal.years
    years = np.arange(1, t + 1)
    rate = d.rate_start + financing.rate_premium
    base = deal.price * (1 - financing.down_pct)
    loan = base * (1 + financing.upfront_mi)
    payment = float(fin.mortgage_payment(loan, rate, financing.term_years))
    balances = fin.loan_balance(loan, rate, financing.term_years, 12 * np.arange(t + 1))
    debt_service = 12 * payment + financing.annual_mi * balances[:-1]

    rent_index = np.cumprod(np.column_stack([np.ones(len(d.rent)), 1 + d.growth]), axis=1)
    cost_index = np.cumprod(np.column_stack([np.ones(len(d.rent)), 1 + d.inflation]), axis=1)
    monthly_rent = d.rent[:, None] * rent_index
    fixed = (deal.price * deal.tax_rate + deal.insurance_per_unit * deal.units) * cost_index
    variable = deal.maintenance_pct + deal.management_pct
    vacancy = np.column_stack([d.vacancy, d.vacancy[:, -1]])

    rented = deal.units - (years <= financing.owner_years)
    egi = 12 * monthly_rent[:, :t] * rented * (1 - vacancy[:, :t])
    noi = egi * (1 - variable) - fixed[:, :t]
    cash_flow = noi - debt_service - d.capex * cost_index[:, :t]

    stabilized = 12 * monthly_rent * deal.units * (1 - vacancy) * (1 - variable) - fixed
    entry_cap = fin.cap_rate(stabilized[:, 0], deal.price)
    exit_cap = np.maximum(entry_cap + deal.exit_cap_beta * (d.rate[:, -1] - d.rate_start), MIN_EXIT_CAP)
    sale = stabilized[:, t] / exit_cap * (1 - deal.selling_cost_pct) - balances[t]

    cash_invested = deal.price * (financing.down_pct + deal.closing_cost_pct)
    flows = np.column_stack([np.full(len(d.rent), -cash_invested), cash_flow])
    flows[:, -1] += sale
    piti = payment + (fixed[0, 0] / 12) + financing.annual_mi * balances[0] / 12
    return {
        "noi_year1": noi[:, 0],
        "cap_rate": entry_cap,
        "dscr": fin.dscr(stabilized[:, 0], debt_service[0]),
        "cash_on_cash": fin.cash_on_cash(cash_flow[:, 0], cash_invested),
        "irr": fin.irr(flows),
        "negative_cf_year1": cash_flow[:, 0] < 0,
        "negative_cf_any_year": (cash_flow < 0).any(axis=1),
        "noi_final": noi[:, -1],
        "cash_flow_total": cash_flow.sum(axis=1),
        "cash_invested": cash_invested,
        "monthly_piti": piti,
        "self_sufficiency": fin.self_sufficiency(deal.rent * deal.units, piti, deal.units),
    }


def summarize(results: dict) -> pd.DataFrame:
    rows = []
    for key in ["noi_year1", "cap_rate", "dscr", "cash_on_cash", "irr"]:
        v = results[key]
        rows.append({"metric": key, "p10": np.nanpercentile(v, 10), "median": np.nanmedian(v),
                     "p90": np.nanpercentile(v, 90)})
    for key in ["negative_cf_year1", "negative_cf_any_year"]:
        rows.append({"metric": f"P({key})", "p10": np.nan, "median": results[key].mean(), "p90": np.nan})
    return pd.DataFrame(rows)


def run(deal: Deal, financing: Financing, market: Market, n: int = 10_000, seed: int = 42) -> dict:
    return simulate(deal, financing, draw_drivers(deal, market, n, np.random.default_rng(seed)))
