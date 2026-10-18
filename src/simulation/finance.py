import numpy as np

FHA_UPFRONT_MIP = 0.0175
FHA_ANNUAL_MIP = 0.0055
SELF_SUFFICIENCY_SHARE = 0.75


def mortgage_payment(principal, annual_rate, years: int = 30):
    r = np.asarray(annual_rate, dtype=float) / 12
    return principal * r / (1 - (1 + r) ** -(years * 12))


def loan_balance(principal, annual_rate, years: int, months_paid):
    r = np.asarray(annual_rate, dtype=float) / 12
    growth = (1 + r) ** months_paid
    return principal * growth - mortgage_payment(principal, annual_rate, years) * (growth - 1) / r


def fha_loan(price: float, down_pct: float = 0.035) -> tuple[float, float]:
    base = price * (1 - down_pct)
    return base, base * (1 + FHA_UPFRONT_MIP)


def fha_monthly_mip(balance, rate: float = FHA_ANNUAL_MIP):
    return np.asarray(balance) * rate / 12


def self_sufficiency(monthly_rents_all_units: float, monthly_piti: float, units: int) -> bool:
    if units < 3:
        return True
    return SELF_SUFFICIENCY_SHARE * monthly_rents_all_units >= monthly_piti


def cap_rate(noi, price):
    return np.asarray(noi) / price


def dscr(noi, annual_debt_service):
    return np.asarray(noi) / np.asarray(annual_debt_service)


def cash_on_cash(annual_cash_flow, cash_invested):
    return np.asarray(annual_cash_flow) / cash_invested


def npv(rate, flows: np.ndarray) -> np.ndarray:
    flows = np.atleast_2d(flows)
    t = np.arange(flows.shape[1])
    rate = np.asarray(rate, dtype=float).reshape(-1, 1)
    return (flows / (1 + rate) ** t).sum(axis=1)


def irr(flows: np.ndarray, low: float = -0.99, high: float = 5.0, iterations: int = 100) -> np.ndarray:
    flows = np.atleast_2d(np.asarray(flows, dtype=float))
    lo = np.full(len(flows), low)
    hi = np.full(len(flows), high)
    f_lo = npv(lo, flows)
    valid = np.sign(f_lo) != np.sign(npv(hi, flows))
    for _ in range(iterations):
        mid = (lo + hi) / 2
        f_mid = npv(mid, flows)
        left = np.sign(f_mid) == np.sign(f_lo)
        lo = np.where(left, mid, lo)
        f_lo = np.where(left, f_mid, f_lo)
        hi = np.where(left, hi, mid)
    return np.where(valid, (lo + hi) / 2, np.nan)
