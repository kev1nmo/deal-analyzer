import os
import sys
import time
from dataclasses import replace

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.config import MLFLOW_TRACKING_URI, RANDOM_SEED, ROOT
from src.forecast import forecast_all
from src.forecast.rent_growth import load_fred, load_zori
from src.phase4 import load
from src.phase6 import ANALYZER
from src.plots import AQUA, BLUE, MUTED, ORANGE, save, use_style
from src.simulation.monte_carlo import (CONVENTIONAL, FHA, Deal, Drivers, annual, draw_drivers, expected_capex,
                                        fit_market, simulate, summarize)

EXPERIMENT = "deal-simulation"
N_PATHS = 10_000
ZIP = "30318"
UNITS = 4
TODAY = pd.Timestamp("2026-09-01")
START_2019 = pd.Timestamp("2018-12-01")
BACKTEST_END = pd.Timestamp("2024-12-01")
PRICE_PER_UNIT = {"today": 165_000, "2019": 120_000}
INSURANCE_PER_UNIT = {"today": 1_500, "2019": 1_100}
CAPEX_MEDIAN = {"today": 3_500, "2019": 2_800}
FINANCINGS = [FHA, CONVENTIONAL]


def table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.astype(object).to_markdown(index=False, floatfmt=floatfmt)


def example_unit() -> tuple[pd.Series, dict]:
    _, test = load()
    pool = test[(test["zip"] == ZIP) & (test["bedrooms"] == 2)]
    unit = pool.iloc[[int(np.argmin(np.abs(pool["rent"] - pool["rent"].median())))]]
    return unit.iloc[0], joblib.load(ANALYZER).analyze(unit)


def scaled_deal(result: dict, listed: pd.Timestamp, as_of: pd.Timestamp, logz: pd.DataFrame, key: str,
                years: int) -> Deal:
    ratio = float(np.exp(logz.at[as_of, ZIP] - logz.at[listed.to_period("M").to_timestamp(), ZIP]))
    lo, hi = result["interval"]
    return Deal(price=PRICE_PER_UNIT[key] * UNITS, units=UNITS, rent=result["predicted_rent"] * ratio,
                rent_low=lo * ratio, rent_high=hi * ratio, years=years, insurance_per_unit=INSURANCE_PER_UNIT[key],
                capex_median=CAPEX_MEDIAN[key])


def results_table(results: dict) -> pd.DataFrame:
    frames = []
    for name, res in results.items():
        s = summarize(res).set_index("metric")
        frames.append(s.rename(columns=lambda c: f"{name.split()[0]} {c}"))
    return pd.concat(frames, axis=1).reset_index()


def actual_drivers(deal: Deal, logz: pd.DataFrame, fred: pd.DataFrame, rate_start: float) -> Drivers:
    years = range(START_2019.year + 1, START_2019.year + 1 + deal.years)
    dec = logz[ZIP].loc[[pd.Timestamp(f"{y}-12-01") for y in [START_2019.year, *years]]]
    vacancy = annual(fred, "us_rental_vacancy", BACKTEST_END, 40).loc[list(years)] / 100
    cpi = annual(fred, "us_cpi", BACKTEST_END, 40)
    inflation = np.exp(np.log(cpi).diff()).loc[list(years)] - 1
    rates = annual(fred, "mortgage_30y", BACKTEST_END, 40).loc[list(years)] / 100
    return Drivers(rent=np.array([deal.rent]), growth=np.expm1(np.diff(dec.to_numpy()))[None, :],
                   vacancy=vacancy.to_numpy()[None, :], inflation=inflation.to_numpy()[None, :],
                   rate=rates.to_numpy()[None, :], capex=np.full((1, deal.years), expected_capex(deal)),
                   rate_start=rate_start)


def percentile_of(sim: np.ndarray, value: float) -> float:
    sim = sim[~np.isnan(sim)]
    return float((sim < value).mean() * 100)


def zip_coverage(logz: pd.DataFrame, fred: pd.DataFrame, years: int) -> pd.DataFrame:
    forecasts = forecast_all(START_2019, logz=logz, fred=fred)
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for z, f in forecasts.items():
        if z == "metro" or pd.isna(logz.at[BACKTEST_END, z]):
            continue
        sim = np.prod(1 + f.paths(years, N_PATHS, rng), axis=1)
        actual = float(np.exp(logz.at[BACKTEST_END, z] - logz.at[START_2019, z]))
        lo, hi = np.percentile(sim, [10, 90])
        rows.append({"zip": z, "actual": actual, "p10": lo, "median": np.median(sim), "p90": hi,
                     "inside_80": lo <= actual <= hi, "percentile": percentile_of(sim, actual)})
    return pd.DataFrame(rows)


def driver_check(sim: Drivers, actual: Drivers) -> pd.DataFrame:
    checks = {
        "rent index, end of 2024 (x start)": (np.prod(1 + sim.growth, 1), np.prod(1 + actual.growth, 1)[0]),
        "average vacancy 2019-2024": (sim.vacancy.mean(1), actual.vacancy.mean()),
        "cost index, end of 2024 (x start)": (np.prod(1 + sim.inflation, 1), np.prod(1 + actual.inflation, 1)[0]),
        "30-yr mortgage rate, 2024": (sim.rate[:, -1], actual.rate[0, -1]),
    }
    return pd.DataFrame([{"driver": k, "sim_p10": np.percentile(s, 10), "sim_median": np.median(s),
                          "sim_p90": np.percentile(s, 90), "actual": a, "actual_percentile": percentile_of(s, a)}
                         for k, (s, a) in checks.items()])


def outcome_check(sims: dict, actuals: dict) -> pd.DataFrame:
    rows = []
    for name in sims:
        for key in ["noi_final", "cash_flow_total", "irr"]:
            s, a = sims[name][key], float(actuals[name][key][0])
            rows.append({"financing": name, "outcome": key, "sim_p10": np.nanpercentile(s, 10),
                         "sim_median": np.nanmedian(s), "sim_p90": np.nanpercentile(s, 90), "with_actual_drivers": a,
                         "percentile": percentile_of(s, a)})
    return pd.DataFrame(rows)


def plot_irr(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-0.4, 0.5, 60)
    for (name, res), color in zip(results.items(), [ORANGE, BLUE]):
        ax.hist(np.clip(res["irr"], -0.4, 0.5), bins=bins, alpha=0.6, color=color, label=name)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xlabel("equity IRR over the hold")
    ax.set_ylabel("simulated paths")
    ax.set_title(f"{UNITS}-unit deal in {ZIP}: 10,000 simulated futures")
    ax.legend()
    save(fig, "phase8_irr")


def plot_backtest(sim: Drivers, actual: Drivers) -> None:
    years = np.arange(START_2019.year, START_2019.year + sim.growth.shape[1] + 1)
    idx = np.column_stack([np.ones(len(sim.growth)), np.cumprod(1 + sim.growth, 1)])
    act = np.concatenate([[1], np.cumprod(1 + actual.growth[0])])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(years, *np.percentile(idx, [10, 90], axis=0), color=AQUA, alpha=0.25, label="simulated 80% band")
    ax.plot(years, np.median(idx, 0), color=AQUA, label="simulated median")
    ax.plot(years, act, color=ORANGE, marker="o", label="actual (ZORI)")
    ax.set_ylabel("rent index (Dec 2018 = 1)")
    ax.set_title(f"2019 backtest: rent in ZIP {ZIP}, simulated from Dec 2018 data")
    ax.legend()
    save(fig, "phase8_backtest_rent")


def write_report(unit, result, deal_today, market, today_table, extras, seconds, deal_2019, cov, drivers, outcomes):
    lo, hi = result["interval"]
    fha, conv = extras
    lines = [
        "# Phase 8 findings: Monte Carlo deal simulation",
        "",
        "## Setup",
        "",
        f"Example deal: a {UNITS}-unit property in ZIP {ZIP} (West Midtown, Fulton), every unit like test listing "
        f"{unit['id']} (2 BR, {unit['square_feet']:.0f} sq ft). The Phase 6 analyzer predicts "
        f"${result['predicted_rent']:,.0f}/month (80% range ${lo:,.0f}-${hi:,.0f}) at 2019 prices. ZORI growth for the "
        f"ZIP rolls that forward to ${deal_today.rent:,.0f} (${deal_today.rent_low:,.0f}-${deal_today.rent_high:,.0f}) "
        f"in {TODAY:%b %Y}. Price ${deal_today.price:,.0f} (${PRICE_PER_UNIT['today']:,}/unit) is an **illustrative "
        "assumption**: the project has no sales data.",
        "",
        f"Each of the {N_PATHS:,} paths over {deal_today.years} years draws:",
        "",
        "- **rent level** from a lognormal fitted to the 80% conformal range (one level per path).",
        f"- **rent growth** from the Phase 7 forecaster: year 1 = {market.growth.year1:+.1%} +/- "
        f"{market.growth.year1_sd:.1%}, then an AR(1) that reverts to {market.growth.long_run:.1%}/yr (phi "
        f"{market.growth.phi:.2f}, sd {market.growth.sd:.1%}), fit to Atlanta CPI rent.",
        f"- **vacancy**: an AR(1) fit to 30 years of US rental vacancy (FRED), mean {market.vacancy.mean:.1%}, "
        f"currently {market.vacancy.start:.1%}.",
        f"- **expense inflation**: an AR(1) fit to US CPI growth, mean {market.inflation.mean:.1%}. It inflates "
        "taxes, insurance and capex.",
        "- **capex shocks**: Poisson(0.25 per unit per year) events, each lognormal with a median of "
        f"${deal_today.capex_median:,.0f}.",
        f"- **interest rates**: a random walk from today's 30-yr rate ({market.rate:.2%}) with annual sd "
        f"{market.rate_change_sd:.2%} (FRED, 30 years). The loan is fixed-rate, so rates matter at exit: exit cap "
        "= entry cap + 0.5 x the rate change.",
        "",
        "Operating assumptions: property tax 1.3% of price, insurance $1,500/unit/yr, maintenance 8% and management "
        "8% of collected rent, 3% closing costs, 6% selling costs.",
        "",
        "Financing:",
        "",
        "- **FHA house hack**: 3.5% down, 1.75% upfront MIP (financed), 0.55%/yr annual MIP, and the owner lives in "
        "one unit for year 1 (no rent from it; the rent the owner avoids paying elsewhere is not counted).",
        "- **Conventional investor**: 25% down; rate +0.75 pp for an investment property.",
        "",
        f"## Results ({TODAY:%b %Y} deal)",
        "",
        f"Runtime: {N_PATHS:,} paths x 2 financing options in {seconds:.2f} s.",
        "",
        table(today_table),
        "",
        f"- FHA: cash in ${fha['cash_invested']:,.0f}, PITI+MIP ${fha['monthly_piti']:,.0f}/month. **Self-sufficiency "
        f"test (3-4 units): {'PASS' if fha['self_sufficiency'] else 'FAIL'}**. 75% of the units' rent "
        f"(${0.75 * deal_today.rent * UNITS:,.0f}) must cover PITI.",
        f"- Conventional: cash in ${conv['cash_invested']:,.0f}, PITI ${conv['monthly_piti']:,.0f}/month.",
        "",
        "![irr](figures/phase8_irr.png)",
        "",
        "## 2019 backtest",
        "",
        f"The same simulation was started in Dec 2018 using only data available then: ZORI and FRED through "
        f"Dec 2018, and the growth forecaster refit on that data. It ran {deal_2019.years} years (2019-2024), and "
        "the actual outcomes were then compared with the simulated range. The deal was priced at "
        f"${PRICE_PER_UNIT['2019']:,}/unit, with rent ${deal_2019.rent:,.0f}/unit.",
        "",
        f"**Across all ZIPs** ({len(cov)} with ZORI data from 2018 through 2024): the actual 2018-2024 rent growth "
        f"fell inside the simulated 80% band for **{cov['inside_80'].mean():.0%}** of ZIPs. The median ZIP sat at the "
        f"{cov['percentile'].median():.0f}th percentile of its simulation. Actual growth averaged "
        f"{cov['actual'].mean() - 1:+.0%} against a simulated median of {cov['median'].mean() - 1:+.0%}.",
        "",
        "**Verdict.** The simulated rent-growth range was too narrow. Nothing in 2015-2018 data foresaw the 2021-22 "
        "pandemic rent spike, so most ZIPs ended near the top of their band or above it. Expense inflation (2021-23 CPI) "
        "and mortgage rates (2022 jump) also landed in the top 5-10% of the simulation. These are tail events the "
        "history-fit AR(1)/random-walk models underweight. For the example deal, though, the outcomes with actual drivers "
        "land mid-distribution: higher rents offset higher costs, and the fixed-rate loan was locked before rates rose. "
        "Takeaway for use: treat the simulated p10-p90 as narrower than reality. Phase 11's break-evens show "
        "how far each input can move.",
        "",
        "**Drivers for this deal** (actual vs. simulated):",
        "",
        table(drivers),
        "",
        "**Deal outcomes**: the deal re-run with the actual 2019-2024 drivers (rent level = point prediction, "
        "capex = expected value), compared with the simulated range:",
        "",
        table(outcomes, ",.3f"),
        "",
        "![backtest](figures/phase8_backtest_rent.png)",
        "",
        "Limitations: the rent level comes from a model trained on 2019 listings, a few months later than the "
        "Dec 2018 start. Vacancy is national, because FRED has no Atlanta series. Prices are assumptions. The "
        "2019 check covers drivers and modeled outcomes, not a real property's books.",
        "",
    ]
    (ROOT / "reports" / "phase8_findings.md").write_text("\n".join(lines))


def main() -> int:
    use_style()
    logz, fred = load_zori(), load_fred()
    unit, result = example_unit()
    listed = pd.Timestamp(unit["listed_at"])

    market = fit_market(forecast_all(TODAY, logz=logz, fred=fred)[ZIP], fred)
    deal_today = scaled_deal(result, listed, TODAY, logz, "today", 7)
    start = time.perf_counter()
    rng = np.random.default_rng(RANDOM_SEED)
    drivers_today = draw_drivers(deal_today, market, N_PATHS, rng)
    today = {f.name: simulate(deal_today, f, drivers_today) for f in FINANCINGS}
    seconds = time.perf_counter() - start
    today_table = results_table(today)
    print(f"[today] {seconds:.2f}s\n{today_table.round(3).to_string(index=False)}")

    fred_2019 = fred[fred["date"] <= START_2019 + pd.offsets.MonthEnd(0)]
    deal_2019 = replace(scaled_deal(result, listed, START_2019, logz, "2019", 6))
    market_2019 = fit_market(forecast_all(START_2019, logz=logz, fred=fred_2019)[ZIP], fred_2019)
    sim_2019 = draw_drivers(deal_2019, market_2019, N_PATHS, np.random.default_rng(RANDOM_SEED))
    act_2019 = actual_drivers(deal_2019, logz, fred, market_2019.rate)
    sims = {f.name: simulate(deal_2019, f, sim_2019) for f in FINANCINGS}
    actuals = {f.name: simulate(deal_2019, f, act_2019) for f in FINANCINGS}
    drivers = driver_check(sim_2019, act_2019)
    outcomes = outcome_check(sims, actuals)
    cov = zip_coverage(logz, fred_2019, deal_2019.years)
    print(drivers.round(3).to_string(index=False))
    print(outcomes.round(3).to_string(index=False))
    print(f"[zip coverage] {cov['inside_80'].mean():.3f} of {len(cov)}; median percentile {cov['percentile'].median():.0f}")

    plot_irr(today)
    plot_backtest(sim_2019, act_2019)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="phase8_simulation"):
        mlflow.set_tags({"phase": "8", "zip": ZIP})
        mlflow.log_params({"paths": N_PATHS, "units": UNITS, "price": deal_today.price, "rent": round(deal_today.rent)})
        for f in FINANCINGS:
            tag = f.name.split()[0].lower()
            mlflow.log_metrics({f"{tag}_median_irr": float(np.nanmedian(today[f.name]["irr"])),
                                f"{tag}_p_negative_cf_year1": float(today[f.name]["negative_cf_year1"].mean())})
        mlflow.log_metrics({"runtime_seconds": seconds, "backtest_zip_coverage_80": float(cov["inside_80"].mean())})
        for name in ["phase8_irr", "phase8_backtest_rent"]:
            mlflow.log_artifact(str(ROOT / "reports" / "figures" / f"{name}.png"))

    write_report(unit, result, deal_today, market, today_table, [today[f.name] for f in FINANCINGS], seconds,
                 deal_2019, cov, drivers, outcomes)
    print("[report] reports/phase8_findings.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
