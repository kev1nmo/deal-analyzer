import argparse
import sys

import pandas as pd

from src import build_db
from src.db import get_conn
from src.sources import census, fred, geo, hud, listings, pums, zcta, zillow

STEPS = {
    "listings": listings.run,
    "hud": hud.run,
    "census": census.run,
    "zillow": zillow.run,
    "fred": fred.run,
    "zcta": zcta.run,
    "shapes": geo.run_shapes,
    "marta": geo.run_marta,
    "pums": pums.run,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1: download Atlanta data and build DuckDB")
    parser.add_argument("--only", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--skip-db", action="store_true")
    args = parser.parse_args()

    failed = []
    for name in args.only:
        print(f"[{name}]")
        try:
            df = STEPS[name]()
            print(f"  saved {len(df):,} rows")
        except Exception as e:
            failed.append(name)
            print(f"  FAILED: {e}")

    if failed:
        print(f"\nFailed steps: {failed}. Fix them, then rerun with --only {' '.join(failed)}")
        return 1

    if not args.skip_db:
        print("[duckdb]")
        con = get_conn()
        build_db.build(con)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(build_db.coverage(con).T.to_string(header=False))
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
