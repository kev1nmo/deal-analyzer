import hashlib
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.api.features import MODEL_DATE, MODEL_INPUT, FeatureBuilder
from src.api.schemas import ActualIn, DealRequest, Prediction, Sensitivity, Simulation, Unit
from src.api.store import Store, predictions, rent_actuals, simulations
from src.breakeven import base_scenario, breakevens, describe, evaluate, market_ranges, tornado
from src.config import ROOT
from src.explain import RentAnalyzer
from src.forecast import GrowthForecast, forecast_all
from src.forecast.rent_growth import load_fred
from src.model_features import county_name
from src.monitoring import monthly_drift
from src.simulation.monte_carlo import CONVENTIONAL, FHA, Deal, Market, draw_drivers, fit_market, simulate, summarize

DEFAULT_ANALYZER = ROOT / "models" / "rent_analyzer.joblib"
DEFAULT_DATABASE = f"sqlite:///{ROOT / 'data' / 'api.db'}"
FINANCINGS = [FHA, CONVENTIONAL]
COMP_FIELDS = ["id", "rent", "bedrooms", "bathrooms", "square_feet", "cityname", "latitude", "longitude", "km_away",
               "listed_at"]


def file_version(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def clean(value):
    return None if value is None or not np.isfinite(value) else float(value)


def to_listing(unit: Unit) -> pd.DataFrame:
    return pd.DataFrame([{
        "id": 0, "listed_at": MODEL_DATE, "bedrooms": unit.bedrooms, "bathrooms": unit.bathrooms,
        "square_feet": unit.square_feet, "amenities": ",".join(unit.amenities), "has_photo": "Yes",
        "pets_allowed": "Cats,Dogs" if unit.pets_allowed else None, "latitude": unit.latitude,
        "longitude": unit.longitude, "coarse_location": False,
    }])


@dataclass
class Service:
    analyzer: RentAnalyzer
    features: FeatureBuilder
    store: Store
    model_version: str
    forecasts: dict[str, GrowthForecast]
    fred: pd.DataFrame
    markets: dict[str, Market] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Service":
        analyzer_path = Path(os.getenv("ANALYZER_PATH", DEFAULT_ANALYZER))
        database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE)
        features = FeatureBuilder()
        fred = load_fred()
        return cls(analyzer=joblib.load(analyzer_path), features=features, store=Store(database_url),
                   model_version=file_version(analyzer_path), forecasts=forecast_all(features.latest_month, fred=fred),
                   fred=fred)

    def market(self, zip_code: str | None) -> Market:
        key = zip_code if zip_code in self.forecasts else "metro"
        if key not in self.markets:
            self.markets[key] = fit_market(self.forecasts[key], self.fred)
        return self.markets[key]

    def analyze(self, unit: Unit) -> tuple[dict, pd.DataFrame]:
        X = self.features.build(to_listing(unit))
        result = self.analyzer.analyze(X)
        zip_code = X.at[0, "zip"]
        ratio = self.features.rent_ratio(zip_code, MODEL_DATE)
        lo, hi = result["interval"]
        out = {
            "model_version": self.model_version, "zip": zip_code,
            "county": county_name(X["county_fips"]).iloc[0], "known_location": result["known_location"],
            "confidence": result["confidence"], "model_date": MODEL_DATE, "predicted_rent": result["predicted_rent"],
            "range_low": lo, "range_high": hi, "as_of": self.features.latest_month,
            "rent_today": result["predicted_rent"] * ratio, "range_today_low": lo * ratio,
            "range_today_high": hi * ratio, "drivers": result["drivers"],
            "comps": result["comps"][COMP_FIELDS].to_dict("records"),
        }
        return out, X

    def predict(self, unit: Unit) -> dict:
        start = time.perf_counter()
        out, X = self.analyze(unit)
        row = {k: out[k] for k in ["model_version", "zip", "predicted_rent", "range_low", "range_high", "rent_today",
                                   "known_location"]}
        out["id"] = self.store.log(predictions, {
            **row, "request": unit.model_dump(),
            "features": {f: clean(X.at[0, f]) for f in MODEL_INPUT},
            "latency_ms": (time.perf_counter() - start) * 1000,
        })
        return out

    def deal(self, req: DealRequest) -> tuple[Deal, Market, dict]:
        pred, _ = self.analyze(req.unit)
        deal = Deal(price=req.price, units=req.units, rent=pred["rent_today"], rent_low=pred["range_today_low"],
                    rent_high=pred["range_today_high"], years=req.years)
        return deal, self.market(pred["zip"]), pred

    def sensitivity(self, req: DealRequest) -> dict:
        deal, market, pred = self.deal(req)
        ranges = market_ranges(deal, market, req.seed)
        base = base_scenario(ranges)
        financing = []
        for f in FINANCINGS:
            be = breakevens(deal, f, ranges)
            be["summary"] = be.apply(describe, axis=1)
            be["breakeven"] = be["breakeven"].astype(object).where(be["breakeven"].notna(), None)
            financing.append({"name": f.name, **evaluate(deal, f, base),
                              "tornado": tornado(deal, f, ranges).to_dict("records"),
                              "breakevens": be.to_dict("records")})
        return {"zip": pred["zip"], "rent_per_unit": deal.rent, "financing": financing}

    def simulate(self, req: DealRequest) -> dict:
        start = time.perf_counter()
        deal, market, pred = self.deal(req)
        drivers = draw_drivers(deal, market, req.n_paths, np.random.default_rng(req.seed))
        financing = []
        for f in FINANCINGS:
            res = simulate(deal, f, drivers)
            metrics = {r["metric"]: {k: clean(r[k]) for k in ["p10", "median", "p90"]}
                       for r in summarize(res).to_dict("records")}
            financing.append({"name": f.name, "cash_invested": float(res["cash_invested"]),
                              "monthly_piti": float(res["monthly_piti"]),
                              "self_sufficiency": bool(res["self_sufficiency"]), "metrics": metrics})
        seconds = time.perf_counter() - start
        out = {"model_version": self.model_version, "zip": pred["zip"], "as_of": market.as_of,
               "rent_per_unit": deal.rent, "rent_low": deal.rent_low, "rent_high": deal.rent_high,
               "growth_year1": float(np.expm1(market.growth.year1)), "growth_method": market.growth.method,
               "seconds": seconds, "financing": financing}
        out["id"] = self.store.log(simulations, {
            "model_version": self.model_version, "zip": pred["zip"], "request": req.model_dump(),
            "result": Simulation.model_validate({**out, "id": 0}).model_dump(mode="json", exclude={"id"}),
            "latency_ms": seconds * 1000,
        })
        return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = Service.load()
    yield
    app.state.service.store.engine.dispose()


app = FastAPI(title="Small Multifamily Deal Analyzer", version="0.9.0", lifespan=lifespan)


def service(request: Request) -> Service:
    return request.app.state.service


@app.get("/health")
def health(request: Request) -> dict:
    s = service(request)
    db = s.store.ping()
    if not db:
        raise HTTPException(503, "database unavailable")
    return {"status": "ok", "model_version": s.model_version, "data_as_of": s.features.latest_month.date(),
            "database": s.store.engine.dialect.name}


@app.post("/predict", response_model=Prediction)
def predict(unit: Unit, request: Request) -> dict:
    return service(request).predict(unit)


@app.post("/simulate", response_model=Simulation)
def simulate_deal(deal: DealRequest, request: Request) -> dict:
    return service(request).simulate(deal)


@app.post("/breakeven", response_model=Sensitivity)
def breakeven(deal: DealRequest, request: Request) -> dict:
    return service(request).sensitivity(deal)


@app.get("/predictions")
def recent_predictions(request: Request, limit: int = Query(20, ge=1, le=500)) -> list[dict]:
    rows = service(request).store.recent(predictions, limit)
    return [{k: v for k, v in r.items() if k != "features"} for r in rows]


@app.post("/actuals", status_code=201)
def record_actual(actual: ActualIn, request: Request) -> dict:
    store = service(request).store
    with store.engine.connect() as conn:
        if conn.execute(select(predictions.c.id).where(predictions.c.id == actual.prediction_id)).first() is None:
            raise HTTPException(404, "prediction not found")
    observed = actual.observed_at or datetime.now()
    try:
        actual_id = store.log(rent_actuals, {
            "prediction_id": actual.prediction_id, "listing_id": actual.listing_id, "rent": actual.rent,
            "observed_at": observed, "observed_month": observed.date().replace(day=1)})
    except IntegrityError:
        raise HTTPException(409, "actual already recorded for this prediction")
    return {"id": actual_id}


@app.get("/drift")
def drift(request: Request, source: str = "api") -> list[dict]:
    report = monthly_drift(service(request).store.engine, source)
    return report.assign(month=report["month"].dt.date.astype(str)).to_dict("records")
