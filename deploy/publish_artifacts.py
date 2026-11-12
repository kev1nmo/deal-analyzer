import os
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
REPO = os.getenv("HF_ARTIFACTS", "kev1nmo/deal-analyzer-artifacts")
FILES = [
    "models/rent_analyzer.joblib",
    "data/raw/acs_tract.parquet",
    "data/raw/acs_zcta.parquet",
    "data/raw/fred.parquet",
    "data/raw/hud_safmr.parquet",
    "data/raw/marta_stations.parquet",
    "data/raw/zillow_zori.parquet",
    "data/raw/geo/tract_shapes.parquet",
    "data/raw/geo/zcta_shapes.parquet",
]
CARD = """---
license: cc-by-4.0
tags: [real-estate, tabular-regression, xgboost, conformal-prediction]
---

# Deal Analyzer artifacts

The trained rent analyzer (XGBoost + MAPIE conformal ranges + SHAP explainer + comps) and the reference data the
[Deal Analyzer](https://github.com/kev1nmo/deal-analyzer) API needs at serving time. Used by the
live demo (Streamlit Community Cloud), which downloads them on first start.

Model card and evaluation: [findings report](https://github.com/kev1nmo/deal-analyzer/blob/main/reports/findings.md).

Data sources and attribution: rent index provided by Zillow (Zillow Observed Rent Index, free for public use with
attribution); U.S. Census Bureau ACS and TIGER shapes; HUD Small Area Fair Market Rents; FRED / Freddie Mac; MARTA GTFS;
training listings from the UCI Apartment for Rent Classified dataset (CC BY 4.0).
"""


def main() -> int:
    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "README.md"
        card.write_text(CARD)
        api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=REPO)
    for f in FILES:
        api.upload_file(path_or_fileobj=ROOT / f, path_in_repo=f, repo_id=REPO, commit_message=f"Upload {f}")
    print(f"Uploaded {len(FILES)} files to https://huggingface.co/{REPO}. Reboot the Streamlit app to load them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
