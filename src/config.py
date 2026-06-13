import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DOWNLOADS = DATA_RAW / "_downloads"
DB_PATH = ROOT / "data" / "deals.duckdb"

FRED_API_KEY = os.getenv("FRED_API_KEY")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")

MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "rent-model"

MARKET_NAME = "Atlanta-Sandy Springs-Roswell, GA"
CBSA_CODE = "12060"
STATE_FIPS = "13"
COUNTY_FIPS = {
    "Fulton": "121",
    "DeKalb": "089",
    "Cobb": "067",
    "Gwinnett": "135",
    "Clayton": "063",
}
DOWNTOWN_LATLON = (33.7537, -84.3915)
MARKET_RADIUS_KM = 50
MARKET_BBOX = (-85.05, 33.2, -83.75, 34.3)
PROJECTED_CRS = 32616

HUD_AREA_CODE = f"METRO{CBSA_CODE}M{CBSA_CODE}"
SAFMR_YEARS = [2026, 2025]
ACS_YEARS = [2024, 2023]
GAZETTEER_YEARS = [2024, 2023]
PUMS_YEARS = [2022, 2023, 2024]

FRED_SERIES = {
    "MORTGAGE30US": "mortgage_30y",
    "MORTGAGE15US": "mortgage_15y",
    "DGS10": "treasury_10y",
    "RRVRUSQ156N": "us_rental_vacancy",
    "CUURA319SEHA": "atlanta_cpi_rent",
    "CPIAUCSL": "us_cpi",
}

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_SCORES = DATA_PROCESSED / "text_scores.parquet"
TEXT_HISTORY = DATA_PROCESSED / "text_history.parquet"

RANDOM_SEED = 42
