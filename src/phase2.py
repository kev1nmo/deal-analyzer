import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src import build_db, hypothesis
from src.clean import clean_listings
from src.config import DATA_PROCESSED, DATA_RAW, ROOT
from src.db import get_conn
from src.features import build_features, coverage
from src.sources.geo import GEO_DIR
from src.spatial import add_spatial_features

REPORTS = ROOT / "reports"
REQUIRED = [
    DATA_RAW / "listings.parquet",
    DATA_RAW / "acs_tract.parquet",
    DATA_RAW / "marta_stations.parquet",
    DATA_RAW / "pums_renters.parquet",
    GEO_DIR / "zcta_shapes.parquet",
    GEO_DIR / "tract_shapes.parquet",
]


def table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    return df.astype(object).to_markdown(index=False, floatfmt=floatfmt)


def rent_summary(features: pd.DataFrame) -> pd.DataFrame:
    return (
        features.groupby("bedrooms")
        .agg(listings=("id", "size"), median_rent=("rent", "median"),
             median_sqft=("square_feet", "median"), median_rent_per_sqft=("rent_per_sqft", "median"))
        .reset_index()
    )


def write_report(log, features, cov, desc, effects, n_pums, path: Path) -> None:
    moved = int(features["location_moved"].sum())
    coarse = int(features["coarse_location"].sum())
    lines = [
        "# Phase 2 findings",
        "",
        "## 1. Cleaning",
        "",
        table(log, ".0f"),
        "",
        f"- Locations are approximate: {features[['latitude', 'longitude']].drop_duplicates().shape[0]} unique map points "
        f"for {len(features):,} listings (neighborhood-level precision, not street addresses).",
        f"- {moved} listings had a map point that belonged to a different city; they were moved to their own city's main point.",
        f"- {coarse} listings sit on a city-wide fallback point (`coarse_location`); their spatial features are city-level only.",
        "",
        "## 2. Rent by bedrooms",
        "",
        table(rent_summary(features)),
        "",
        "## 3. Feature coverage (share of listings with a value)",
        "",
        table(cov.reset_index(names="feature")[lambda d: d["share_filled"] < 1], ".3f"),
        "",
        "Every column not listed is 100% filled.",
        "",
        f"Timing caveat: listings run from {features['listed_at'].min():%b %Y} to {features['listed_at'].max():%b %Y}, "
        "while HUD fair rents are current fiscal-year values and Census figures cover 2020-2024. Treat those as relative "
        "neighborhood signals, not rent levels at listing time. ZORI and mortgage rates are matched to the listing month.",
        "",
        "## 4. Hypothesis test: do 3-4 unit buildings rent at a per-unit discount?",
        "",
        "The listings data has no building size (all are tagged 'apartment'), so the test uses Census ACS PUMS "
        f"microdata: {n_pums:,} renter households in Fulton, DeKalb, Cobb, Gwinnett and Clayton PUMAs, survey years 2022-2024.",
        "",
        "Model: log(monthly rent) ~ building size + bedrooms + rooms + decade built + PUMA + survey year, "
        "weighted by household weight, heteroskedasticity-robust (HC1) standard errors. Reference group: 5+ unit buildings.",
        "",
        f"**Result:** {hypothesis.headline(effects)}",
        "",
        table(desc, ".0f"),
        "",
        "Effects for every building size and robustness check (percent difference vs 5+ unit buildings):",
        "",
        table(effects[["spec", "building", "pct_vs_5plus", "ci_low_pct", "ci_high_pct", "p_value", "n"]], ".3f"),
        "",
        "Caveats: PUMS has no square footage (rooms and bedrooms stand in for size), PUMAs are large areas "
        "(about 100K+ people), and rents are what current tenants pay, not new asking rents.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
    if missing:
        print("Missing inputs. Run: python -m src.pipeline --only census shapes marta pums")
        print("\n".join(f"  {m}" for m in missing))
        return 1

    REPORTS.mkdir(exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print("[clean]")
    listings, log = clean_listings(pd.read_parquet(DATA_RAW / "listings.parquet"))
    print(log.to_string(index=False))

    print("[spatial]")
    spatial = add_spatial_features(
        listings,
        gpd.read_parquet(GEO_DIR / "zcta_shapes.parquet"),
        gpd.read_parquet(GEO_DIR / "tract_shapes.parquet"),
        pd.read_parquet(DATA_RAW / "marta_stations.parquet"),
        pd.read_parquet(DATA_RAW / "acs_tract.parquet"),
    )

    print("[duckdb]")
    con = get_conn()
    build_db.build(con)
    features = build_features(con, spatial)
    con.close()
    features.to_parquet(DATA_PROCESSED / "listings_features.parquet", index=False)
    cov = coverage(features)
    print(f"  listings_features: {len(features):,} rows x {features.shape[1]} columns")

    print("[hypothesis]")
    pums = hypothesis.prepare(pd.read_parquet(DATA_RAW / "pums_renters.parquet"))
    effects = hypothesis.run_specs(pums)
    desc = hypothesis.describe(pums)
    print("  " + hypothesis.headline(effects))

    path = REPORTS / "phase2_findings.md"
    write_report(log, features, cov, desc, effects, len(pums), path)
    print(f"[report] {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
