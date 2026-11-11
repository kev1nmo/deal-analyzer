import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.api.features import FeatureBuilder
from src.api.main import FINANCINGS, to_listing
from src.api.schemas import Unit
from src.breakeven import EQUITY_WIPED, base_scenario, breakevens, describe, evaluate, fmt, market_ranges, tornado
from src.config import ROOT
from src.forecast import forecast_all
from src.forecast.rent_growth import load_fred
from src.phase6 import ANALYZER
from src.plots import BLUE, MUTED, ORANGE, save, use_style
from src.simulation.monte_carlo import Deal, fit_market

UNIT = Unit(latitude=33.7866, longitude=-84.4149, bedrooms=2, bathrooms=1, square_feet=950)
UNITS = 4
PRICE = 660_000


def example_deal() -> tuple[Deal, object, str]:
    fb = FeatureBuilder()
    X = fb.build(to_listing(UNIT))
    result = joblib.load(ANALYZER).analyze(X)
    zip_code = X.at[0, "zip"]
    ratio = fb.rent_ratio(zip_code, X.at[0, "listed_at"])
    lo, hi = result["interval"]
    deal = Deal(price=PRICE, units=UNITS, rent=result["predicted_rent"] * ratio, rent_low=lo * ratio,
                rent_high=hi * ratio)
    fred = load_fred()
    forecasts = forecast_all(fb.latest_month, fred=fred)
    return deal, fit_market(forecasts.get(zip_code, forecasts["metro"]), fred), zip_code


def plot_tornado(tornados: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, len(tornados), figsize=(11, 4.2), sharey=False)
    for ax, (name, t) in zip(axes, tornados.items()):
        t = t.iloc[::-1]
        y = np.arange(len(t))
        base = t["base"].iloc[0]
        low = np.maximum(t["at_low"], -0.3)
        high = np.maximum(t["at_high"], -0.3)
        ax.barh(y, low - base, left=base, color=ORANGE, label="input at its p10")
        ax.barh(y, high - base, left=base, color=BLUE, label="input at its p90")
        ax.axvline(base, color=MUTED, linewidth=1)
        ax.axvline(0, color=MUTED, linewidth=1, linestyle=":")
        ax.set_yticks(y, [f"{r.label.split(' (')[0]}\n{fmt(r.input, r.low_input)} / {fmt(r.input, r.high_input)}"
                          .replace("$", r"\$") for r in t.itertuples()], fontsize=8)
        for yi, value in zip(y, t["at_low"]):
            if value <= EQUITY_WIPED:
                ax.text(-0.29, yi, "-100%: equity wiped", va="center", ha="left", fontsize=8, color="white")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlabel("equity IRR over 7 years")
        ax.set_title(f"{name} (base IRR {base:.1%})")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"What moves the IRR: {UNITS}-unit deal at ${PRICE:,}, one input at a time (p10 / p90)")
    fig.tight_layout()
    save(fig, "phase11_tornado")


def table(df: pd.DataFrame) -> str:
    return df.astype(object).to_markdown(index=False)


def write_report(deal: Deal, zip_code: str, ranges: dict, results: dict) -> None:
    range_rows = pd.DataFrame([{"input": k, "p10": fmt(k, v[0]), "median": fmt(k, v[1]), "p90": fmt(k, v[2])}
                               for k, v in ranges.items()])
    lines = [
        "# Phase 11 findings: what drives the deal, and where it breaks",
        "",
        f"Example: a {UNITS}-unit property in ZIP {zip_code} at ${PRICE:,}, every unit a 2 BR / 1 BA, "
        f"{UNIT.square_feet} sq ft. Rent today ${deal.rent:,.0f}/unit (80% range ${deal.rent_low:,.0f}-"
        f"${deal.rent_high:,.0f}).",
        "",
        "## Method",
        "",
        "- Same cash-flow engine as the Phase 8 Monte Carlo, run on one deterministic path: every driver held at "
        "its median, then **one input at a time moved to its own p10 and p90** from the 10,000 simulated paths. "
        "So the tornado's ranges are the simulation's ranges, not hand-picked +/-10% swings. Mortgage rate (+/-1 pp) "
        "and price (+/-10%) are deal terms, not forecasts, so they get fixed swings.",
        "- Break-evens: SciPy `brentq` solves for the input value where year-1 cash flow (or 7-year IRR) hits zero, "
        "holding everything else at the median. A grid scan first finds the sign change closest to today's value, "
        "so the root is bracketed. If losses exceed the down payment, IRR is undefined; it is scored as -100%.",
        "",
        "Input ranges (from the simulation):",
        "",
        table(range_rows),
        "",
        "![tornado](figures/phase11_tornado.png)",
        "",
    ]
    for name, (base, tor, be) in results.items():
        lines += [
            f"## {name}",
            "",
            f"Median path: year-1 cash flow ${base['cash_flow_year1']:,.0f}, 7-year IRR {base['irr']:.1%}.",
            "",
            "Biggest IRR swings: " + ", ".join(f"{r.label.split(' (')[0].lower()} ({r.swing:.0%} pts)"
                                               for r in tor.head(3).itertuples()) + ".",
            "",
            "Break-evens:",
            "",
            *[f"- {describe(r)}" for _, r in be.iterrows()],
            "",
        ]
    lines += [
        "## Takeaways",
        "",
        "- **Rent level and rent growth dominate.** Vacancy and expense inflation barely move the IRR because their "
        "simulated ranges are narrow (national data). That is a limitation, not proof they don't matter: the "
        "break-evens show how far vacancy would have to go.",
        "- **The two loans fail in different ways.** Conventional has a cushion on year-1 cash flow; FHA is cash-flow "
        "negative while the owner occupies a unit, and with 3.5% down, a few years of flat rents can wipe out the equity "
        "at sale.",
        "",
    ]
    (ROOT / "reports" / "phase11_findings.md").write_text("\n".join(lines))


def main() -> int:
    use_style()
    deal, market, zip_code = example_deal()
    ranges = market_ranges(deal, market)
    base = base_scenario(ranges)
    results, tornados = {}, {}
    for f in FINANCINGS:
        tor = tornado(deal, f, ranges)
        be = breakevens(deal, f, ranges)
        results[f.name] = (evaluate(deal, f, base), tor, be)
        tornados[f.name] = tor
        print(f"[{f.name}]\n{tor.round(3).to_string(index=False)}\n" + "\n".join(be.apply(describe, axis=1)))
    plot_tornado(tornados)
    write_report(deal, zip_code, ranges, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
