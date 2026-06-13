from src.config import CBSA_CODE, DATA_PROCESSED, DATA_RAW
from src.db import get_conn


def test_duckdb_works():
    con = get_conn()
    assert con.execute("SELECT 1 + 1").fetchone()[0] == 2
    con.close()


def test_data_folders_exist():
    assert DATA_RAW.exists()
    assert DATA_PROCESSED.exists()


def test_market_is_atlanta():
    assert CBSA_CODE == "12060"
