import os
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
API_URL = os.getenv("API_URL")
ARTIFACTS_REPO = "kev1nmo/deal-analyzer-artifacts"
AMENITIES = ["Pool", "Gym", "Parking", "Washer Dryer", "Patio/Deck", "Fireplace", "Gated", "Wood Floors",
             "Dishwasher", "Elevator", "Storage", "Clubhouse"]
GEOCODER = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
EXAMPLES = {
    "West Midtown fourplex": dict(address="1000 Marietta St NW, Atlanta, GA", bedrooms=2, bathrooms=1.0,
                                  square_feet=950, amenities=["Dishwasher"], units=4, price=660_000),
    "Decatur duplex": dict(address="101 E Court Square, Decatur, GA", bedrooms=2, bathrooms=1.0, square_feet=1000,
                           amenities=["Washer Dryer", "Wood Floors"], units=2, price=450_000),
    "Marietta triplex": dict(address="50 N Park Square, Marietta, GA", bedrooms=2, bathrooms=1.5, square_feet=1050,
                             amenities=["Parking", "Washer Dryer"], units=3, price=480_000),
}
METRIC_LABELS = {"cap_rate": "Cap rate", "dscr": "DSCR", "cash_on_cash": "Cash-on-cash (yr 1)", "irr": "IRR (7 yr)",
                 "P(negative_cf_year1)": "P(negative cash flow, yr 1)",
                 "P(negative_cf_any_year)": "P(negative cash flow, any yr)"}


@st.cache_data(show_spinner=False, ttl=86_400)
def geocode(address: str) -> tuple[float, float, str] | None:
    r = requests.get(GEOCODER, params={"address": address, "benchmark": "Public_AR_Current", "format": "json"},
                     timeout=20)
    r.raise_for_status()
    matches = r.json()["result"]["addressMatches"]
    if not matches:
        return None
    m = matches[0]
    return m["coordinates"]["y"], m["coordinates"]["x"], m["matchedAddress"]


@st.cache_resource(show_spinner=False)
def local_api():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if not (ROOT / "models" / "rent_analyzer.joblib").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(ARTIFACTS_REPO, local_dir=ROOT, allow_patterns=["models/*", "data/*"])
    from fastapi.testclient import TestClient

    from src.api.main import app

    client = TestClient(app)
    client.__enter__()
    return client


def wait_for_api(seconds: int = 90) -> None:
    if not API_URL:
        local_api()
        return
    deadline = time.time() + seconds
    while True:
        try:
            requests.get(f"{API_URL}/health", timeout=5).raise_for_status()
            return
        except requests.RequestException:
            if time.time() > deadline:
                raise
            time.sleep(2)


def call(path: str, payload: dict) -> dict:
    if API_URL:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=60)
    else:
        r = local_api().post(path, json=payload)
    if r.status_code == 422:
        detail = r.json()["detail"]
        if any(field in e["loc"] for e in detail for field in ("latitude", "longitude")):
            st.error("This tool only covers metro Atlanta. Try an address in Fulton, DeKalb, Cobb, Gwinnett or "
                     "Clayton County.")
        else:
            st.error("Please check the inputs: " + "; ".join(e["msg"] for e in detail))
        st.stop()
    r.raise_for_status()
    return r.json()


def money(v: float) -> str:
    return f"${v:,.0f}"


def simulation_table(sim: dict) -> pd.DataFrame:
    rows = {}
    for f in sim["financing"]:
        col = {"Cash invested": money(f["cash_invested"]), "Monthly PITI": money(f["monthly_piti"]),
               "FHA self-sufficiency": "pass" if f["self_sufficiency"] else "FAIL"}
        for key, label in METRIC_LABELS.items():
            m = f["metrics"][key]
            if key.startswith("P("):
                col[label] = f"{m['median']:.0%}"
            elif key == "dscr":
                col[label] = f"{m['median']:.2f} ({m['p10']:.2f}-{m['p90']:.2f})"
            else:
                col[label] = f"{m['median']:.1%} ({m['p10']:.1%} to {m['p90']:.1%})"
        rows[f["name"]] = col
    return pd.DataFrame(rows)


def tornado_chart(f: dict) -> alt.Chart:
    t = pd.DataFrame(f["tornado"])
    base = f["irr"]
    long = pd.concat([
        t.assign(end=t["at_low"].clip(lower=-0.3), side="input at p10"),
        t.assign(end=t["at_high"].clip(lower=-0.3), side="input at p90"),
    ])
    long["start"] = base
    order = t["label"].tolist()
    return alt.Chart(long).mark_bar().encode(
        y=alt.Y("label:N", sort=order, title=None),
        x=alt.X("start:Q", title="IRR (7 yr)", axis=alt.Axis(format="%")),
        x2="end:Q",
        color=alt.Color("side:N", scale=alt.Scale(range=["#eb6834", "#2a78d6"]), title=None),
        tooltip=["label", alt.Tooltip("end:Q", format=".1%", title="IRR")],
    ).properties(height=280)


st.set_page_config(page_title="Deal Analyzer", layout="wide")
st.title("Small Multifamily Deal Analyzer")
st.caption("Rent estimate, 10,000 simulated futures (FHA vs conventional) and break-evens for 2-4 unit "
           "properties in metro Atlanta. [How it works](https://github.com/kev1nmo/deal-analyzer)")

with st.sidebar:
    example = EXAMPLES[st.selectbox("Example property", list(EXAMPLES))]
    address = st.text_input("Address", value=example["address"], help="Metro Atlanta only (Census geocoder).")
    st.subheader("Typical unit")
    beds = st.number_input("Bedrooms", 0, 4, example["bedrooms"])
    baths = st.number_input("Bathrooms", 0.5, 4.0, example["bathrooms"], step=0.5)
    sqft = st.number_input("Square feet", 250, 3500, example["square_feet"], step=50)
    amenities = st.multiselect("Amenities", AMENITIES, example["amenities"])
    st.subheader("Deal")
    units = st.number_input("Units in building", 2, 4, example["units"])
    price = st.number_input("Purchase price ($)", 100_000, 5_000_000, example["price"], step=10_000)
    run = st.button("Analyze", type="primary", width="stretch")

if not run:
    st.info("Pick an example or type an address, then press **Analyze**.")
    st.stop()

try:
    found = geocode(address)
except requests.RequestException:
    st.error("The address lookup service is not responding. Try again in a minute.")
    st.stop()
if found is None:
    st.error(f"Could not find **{address}**. Try a full street address with city and state.")
    st.stop()
lat, lon, matched = found
unit = {"latitude": lat, "longitude": lon, "bedrooms": int(beds), "bathrooms": float(baths),
        "square_feet": int(sqft), "amenities": amenities}
deal = {"unit": unit, "units": int(units), "price": float(price)}
try:
    with st.spinner("Loading the model (the first visit can take about a minute)..."):
        wait_for_api()
    with st.spinner("Predicting rent, simulating 10,000 futures, solving break-evens..."):
        pred = call("/predict", unit)
        sim = call("/simulate", deal)
        sens = call("/breakeven", deal)
except requests.RequestException as e:
    st.error(f"Could not reach the API at {API_URL}: {e}. Start it with `docker compose up -d`, or unset "
             "API_URL to run the model inside the app.")
    st.stop()

st.caption(f"Matched address: {matched}")
st.header("1. Rent per unit")
c1, c2, c3 = st.columns(3)
c1.metric(f"Rent today ({pd.Timestamp(pred['as_of']):%b %Y})", money(pred["rent_today"]))
c2.metric("80% range", f"{money(pred['range_today_low'])} - {money(pred['range_today_high'])}")
c3.metric("Location", f"ZIP {pred['zip']}, {pred['county']}",
          "seen in training" if pred["known_location"] else "new location: wider range", delta_color="off")
left, right = st.columns(2)
with left:
    st.subheader("Why this rent (SHAP)")
    st.dataframe(pd.DataFrame(pred["drivers"]).assign(effect=lambda d: d["effect_pct"].map("{:+.1f}%".format))
                 [["feature", "value", "effect"]], hide_index=True, width="stretch")
with right:
    st.subheader("Nearest comps (2019 listings)")
    st.dataframe(pd.DataFrame(pred["comps"])[["rent", "bedrooms", "bathrooms", "square_feet", "cityname", "km_away"]]
                 .round({"km_away": 1}), hide_index=True, width="stretch")
comps = pd.DataFrame(pred["comps"])
points = pd.concat([
    pd.DataFrame({"lat": [lat], "lon": [lon], "color": ["#eb6834"], "size": [120]}),
    pd.DataFrame({"lat": comps["latitude"], "lon": comps["longitude"], "color": "#2a78d6", "size": 60}),
])
st.map(points, latitude="lat", longitude="lon", color="color", size="size", zoom=11)
st.caption("Orange: this property. Blue: the comparable listings.")
st.caption(f"The model was trained on 2019 listings ({money(pred['predicted_rent'])} at 2019 prices); "
           "today's rent rolls that forward with the ZIP's Zillow rent index.")

st.header("2. 10,000 simulated futures: FHA vs conventional")
st.caption(f"Median (p10 to p90). Year-1 rent growth forecast {sim['growth_year1']:+.1%} ({sim['growth_method']}); "
           f"ran in {sim['seconds']:.2f} s.")
st.dataframe(simulation_table(sim), width="stretch")

st.header("3. What moves the deal, and where it breaks")
tabs = st.tabs([f["name"] for f in sens["financing"]])
for tab, f in zip(tabs, sens["financing"]):
    with tab:
        a, b = st.columns([3, 2])
        a.altair_chart(tornado_chart(f), width="stretch")
        b.metric("Median-path year-1 cash flow", money(f["cash_flow_year1"]))
        b.metric("Median-path IRR", f"{f['irr']:.1%}")
        b.markdown("**Break-evens** (everything else at its median):")
        b.markdown("\n".join(f"- {x['summary']}" for x in f["breakevens"]))

st.caption("Purchase prices are user inputs; there is no sales data. Ranges come from the model's own backtests; "
           "see the findings report for limitations. Not financial advice.")
st.caption("Data: rent index provided by Zillow (Zillow Observed Rent Index); U.S. Census Bureau (ACS, geocoder); "
           "HUD Fair Market Rents; FRED / Freddie Mac; MARTA; UCI Apartment for Rent Classified listings (2019).")
