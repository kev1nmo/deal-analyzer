import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BUILDING_GROUPS = {
    2: "1 unit",
    3: "1 unit",
    4: "2 units",
    5: "3-4 units",
    6: "5+ units",
    7: "5+ units",
    8: "5+ units",
    9: "5+ units",
}
REFERENCE = "5+ units"
CONTROLS = "C(beds) + rooms + C(built) + C(puma) + C(year)"
TERM = f"C(building, Treatment(reference='{REFERENCE}'))"


def decade_built(year_built: pd.Series) -> pd.Series:
    decade = (year_built // 10 * 10).astype(int).astype(str)
    decade = decade.where(year_built >= 1940, "pre-1940")
    return decade.where(year_built < 2020, "2020+")


def prepare(pums: pd.DataFrame) -> pd.DataFrame:
    df = pums.copy()
    df["building"] = df["bld"].map(BUILDING_GROUPS)
    df = df.dropna(subset=["building", "rent", "bedrooms", "rooms", "year_built", "weight"])
    df = df[(df["rent"] > 0) & (df["gross_rent"] > 0) & (df["year_built"] >= 1939) & (df["weight"] > 0)].copy()
    df["log_rent"] = np.log(df["rent"])
    df["log_gross_rent"] = np.log(df["gross_rent"])
    df["beds"] = df["bedrooms"].clip(upper=4).astype(int).astype(str)
    df["rooms"] = df["rooms"].clip(upper=9)
    df["built"] = decade_built(df["year_built"])
    df["puma"] = df["puma"].astype(str)
    df["year"] = df["year"].astype(str)
    return df.reset_index(drop=True)


def fit(df: pd.DataFrame, outcome: str = "log_rent", weighted: bool = True, cluster: bool = False):
    formula = f"{outcome} ~ {TERM} + {CONTROLS}"
    model = smf.wls(formula, df, weights=df["weight"]) if weighted else smf.ols(formula, df)
    if cluster:
        return model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["puma"])[0]})
    return model.fit(cov_type="HC1")


def building_effects(result) -> pd.DataFrame:
    ci = result.conf_int()
    rows = []
    for name in result.params.index:
        if name.startswith(TERM):
            level = name.split("[T.")[1].rstrip("]")
            b = result.params[name]
            rows.append({
                "building": level,
                "coef": b,
                "pct_vs_5plus": np.expm1(b) * 100,
                "ci_low_pct": np.expm1(ci.loc[name, 0]) * 100,
                "ci_high_pct": np.expm1(ci.loc[name, 1]) * 100,
                "p_value": result.pvalues[name],
            })
    return pd.DataFrame(rows)


SPECS = {
    "Main: weighted, robust SE": {"outcome": "log_rent", "weighted": True, "cluster": False},
    "Unweighted, robust SE": {"outcome": "log_rent", "weighted": False, "cluster": False},
    "Weighted, SE clustered by PUMA": {"outcome": "log_rent", "weighted": True, "cluster": True},
    "Gross rent (incl. utilities)": {"outcome": "log_gross_rent", "weighted": True, "cluster": False},
}


def run_specs(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for label, spec in SPECS.items():
        result = fit(df, **spec)
        frames.append(building_effects(result).assign(spec=label, n=int(result.nobs), r2=result.rsquared))
    return pd.concat(frames, ignore_index=True)


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    order = np.argsort(values.to_numpy())
    v, w = values.to_numpy()[order], weights.to_numpy()[order]
    return float(v[np.searchsorted(np.cumsum(w), w.sum() / 2)])


def describe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for building, g in df.groupby("building"):
        rows.append({
            "building": building,
            "households_sampled": len(g),
            "weighted_median_rent": weighted_median(g["rent"], g["weight"]),
            "median_bedrooms": float(g["bedrooms"].median()),
        })
    return pd.DataFrame(rows)


def headline(effects: pd.DataFrame) -> str:
    row = effects[(effects["spec"] == next(iter(SPECS))) & (effects["building"] == "3-4 units")].iloc[0]
    direction = "less" if row["pct_vs_5plus"] < 0 else "more"
    verdict = "statistically significant" if row["p_value"] < 0.05 else "not statistically significant"
    return (
        f"Holding bedrooms, rooms, building age, location (PUMA) and survey year fixed, "
        f"units in 3-4 unit buildings rent for {abs(row['pct_vs_5plus']):.1f}% {direction} than units in 5+ unit buildings "
        f"(difference {row['pct_vs_5plus']:+.1f}%, 95% CI {row['ci_low_pct']:+.1f}% to {row['ci_high_pct']:+.1f}%, "
        f"p = {row['p_value']:.3g}); "
        f"this is {verdict} at the 5% level."
    )
