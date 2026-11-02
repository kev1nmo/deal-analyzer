from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.config import MARKET_BBOX

Amenity = Literal["Pool", "Gym", "Parking", "Washer Dryer", "Patio/Deck", "Fireplace", "Gated", "Wood Floors",
                  "Dishwasher", "Elevator", "Storage", "Clubhouse"]
MIN_SQFT_PER_BEDROOM = 220


class Unit(BaseModel):
    latitude: float = Field(ge=MARKET_BBOX[1], le=MARKET_BBOX[3])
    longitude: float = Field(ge=MARKET_BBOX[0], le=MARKET_BBOX[2])
    bedrooms: int = Field(ge=0, le=4)
    bathrooms: float = Field(ge=0.5, le=4)
    square_feet: int = Field(ge=250, le=3500)
    amenities: list[Amenity] = []
    pets_allowed: bool = False

    @model_validator(mode="after")
    def plausible_size(self) -> "Unit":
        if self.square_feet < MIN_SQFT_PER_BEDROOM * self.bedrooms:
            raise ValueError(f"square_feet must be at least {MIN_SQFT_PER_BEDROOM} per bedroom")
        return self

    model_config = {"json_schema_extra": {"examples": [{
        "latitude": 33.7866, "longitude": -84.4149, "bedrooms": 2, "bathrooms": 1, "square_feet": 950,
        "amenities": ["Dishwasher", "Washer Dryer"], "pets_allowed": True}]}}


class Driver(BaseModel):
    feature: str
    value: float
    effect_pct: float


class Comp(BaseModel):
    id: int
    rent: float
    bedrooms: int
    bathrooms: float
    square_feet: int
    cityname: str | None
    latitude: float
    longitude: float
    km_away: float
    listed_at: datetime


class Prediction(BaseModel):
    id: int
    model_version: str
    zip: str | None
    county: str
    known_location: bool
    confidence: float
    model_date: datetime
    predicted_rent: float
    range_low: float
    range_high: float
    as_of: datetime
    rent_today: float
    range_today_low: float
    range_today_high: float
    drivers: list[Driver]
    comps: list[Comp]


class DealRequest(BaseModel):
    unit: Unit
    units: int = Field(ge=2, le=4, description="Units in the building; every unit is assumed similar to `unit`.")
    price: float = Field(gt=50_000, le=5_000_000)
    years: int = Field(7, ge=2, le=15)
    n_paths: int = Field(10_000, ge=1_000, le=50_000)
    seed: int = 42


class Metric(BaseModel):
    p10: float | None
    median: float | None
    p90: float | None


class FinancingResult(BaseModel):
    name: str
    cash_invested: float
    monthly_piti: float
    self_sufficiency: bool | None
    metrics: dict[str, Metric]


class Simulation(BaseModel):
    id: int
    model_version: str
    zip: str | None
    as_of: datetime
    rent_per_unit: float
    rent_low: float
    rent_high: float
    growth_year1: float
    growth_method: str
    seconds: float
    financing: list[FinancingResult]


class ActualIn(BaseModel):
    prediction_id: int
    rent: float = Field(gt=100, le=20_000)
    observed_at: datetime | None = None
    listing_id: int | None = None


class TornadoBar(BaseModel):
    input: str
    label: str
    low_input: float
    high_input: float
    at_low: float
    at_high: float


class Breakeven(BaseModel):
    metric: Literal["cash_flow_year1", "irr"]
    input: str
    base_input: float
    breakeven: float | None
    summary: str


class FinancingSensitivity(BaseModel):
    name: str
    cash_flow_year1: float
    irr: float
    tornado: list[TornadoBar]
    breakevens: list[Breakeven]


class Sensitivity(BaseModel):
    zip: str | None
    rent_per_unit: float
    financing: list[FinancingSensitivity]
