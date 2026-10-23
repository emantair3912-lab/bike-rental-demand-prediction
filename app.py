"""
Bike Rental Demand Prediction System - Streamlit web application
Machine Learning Fundamentals (AIC354), Assignment 3 - Task 2

The app loads the complete fitted scikit-learn pipeline saved by the notebook
(pre-processing + best algorithm), converts friendly user inputs into the exact 12
raw columns the pipeline expects, validates them, and shows the predicted number of
bike rentals for the chosen hour.
"""
import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import sklearn
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "bike_demand_model.joblib"
META_PATH = BASE_DIR / "model" / "model_metadata.json"

st.set_page_config(page_title="Bike Rental Demand Predictor", page_icon="🚲", layout="centered")


# --------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading the trained model ...")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("Model files not found. Make sure `model/bike_demand_model.joblib` and "
             "`model/model_metadata.json` are in the repository (they are created by Step 10.3 of the notebook).")
    st.stop()

try:
    model = load_model()
    meta = load_metadata()
except Exception as err:  # e.g. scikit-learn version mismatch
    st.error(f"The model file could not be loaded: {err}")
    st.info("This usually means the scikit-learn version on the server differs from the one used for training. "
            "Use the `requirements.txt` written by the notebook.")
    st.stop()

FEATURES = meta["features"]
SCALE = meta["scaling_from_readme"]
RMSE = meta["cross_validation"]["mean_rmse"]
trained_with = meta["library_versions"].get("scikit-learn")

# --------------------------------------------------------------------------- constants
import math
_r = meta["feature_ranges"]
TEMP_MIN, TEMP_MAX = float(math.ceil(_r["temp"]["min"] * SCALE["temp_divisor"])), float(math.floor(_r["temp"]["max"] * SCALE["temp_divisor"]))
HUM_MIN, HUM_MAX = int(math.ceil(_r["hum"]["min"] * SCALE["hum_divisor"])), int(math.floor(_r["hum"]["max"] * SCALE["hum_divisor"]))
WIND_MIN, WIND_MAX = 0.0, float(math.floor(_r["windspeed"]["max"] * SCALE["windspeed_divisor"]))
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]   # code 0..6 (as in the dataset)
SEASONS = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
WEATHER = {
    1: "Clear / few clouds",
    2: "Mist / cloudy",
    3: "Light snow or light rain",
    4: "Heavy rain, snow or fog",
}


def build_row(year, month, hour, weekday, holiday, weather, temp_c, humidity_pct, wind_kmh):
    """Convert friendly inputs into the 12 raw columns used for training."""
    temp = temp_c / SCALE["temp_divisor"]
    fit = meta["atemp_from_temp"]
    atemp = min(1.0, max(0.0, fit["slope"] * temp + fit["intercept"]))       # "feels like" estimated from temperature
    working = int(weekday in range(1, 6) and not holiday)                  # Mon-Fri and not a holiday
    row = {
        "yr": 1 if year == 2012 else 0,
        "mnth": month,
        "hr": hour,
        "season": meta["season_by_month"][str(month)],
        "holiday": int(holiday),
        "weekday": weekday,
        "workingday": working,
        "weathersit": weather,
        "temp": round(temp, 4),
        "atemp": round(atemp, 4),
        "hum": humidity_pct / SCALE["hum_divisor"],
        "windspeed": wind_kmh / SCALE["windspeed_divisor"],
    }
    return pd.DataFrame([row])[FEATURES]


def validate(row):
    """Return (errors, warnings). Errors block the prediction."""
    errors, warnings = [], []
    for col in FEATURES:
        lo, hi = meta["feature_ranges"][col]["min"], meta["feature_ranges"][col]["max"]
        value = float(row[col].iloc[0])
        if not (lo - 1e-9 <= value <= hi + 1e-9):
            errors.append(f"`{col}` = {value:g} is outside the range seen in the training data ({lo:g} to {hi:g}).")
    if row.isnull().any().any():
        errors.append("Some inputs are missing.")
    w = int(row["weathersit"].iloc[0])
    hours_seen = meta["weather_hours_in_training_data"].get(str(w), 0)
    if hours_seen < 100:
        warnings.append(f"This weather type appears in only {hours_seen} of the {meta['n_training_rows']:,} training hours, "
                        "so the estimate is less reliable.")
    return errors, warnings


# --------------------------------------------------------------------------- page
st.title("🚲 Bike Rental Demand Predictor")
st.write("Estimate **how many bikes will be rented in one hour** from the time and the weather. "
         "Built for the *Machine Learning Fundamentals* assignment on the UCI Bike Sharing dataset "
         "(Capital Bikeshare, Washington D.C., 2011–2012).")

col_time, col_weather = st.columns(2, gap="large")

with col_time:
    st.subheader("📅 When?")
    year = st.radio("Year", [2011, 2012], index=1, horizontal=True,
                    help="The data covers only 2011 and 2012; demand grew a lot between the two years.")
    month_name = st.selectbox("Month", MONTHS, index=5)
    weekday_name = st.selectbox("Day of the week", WEEKDAYS, index=4)
    hour = st.slider("Hour of the day", 0, 23, 17, format="%d:00",
                     help="0 = midnight, 8 = 8 am, 17 = 5 pm")
    holiday = st.checkbox("Public holiday", value=False)

with col_weather:
    st.subheader("🌤️ Weather")
    weather_label = st.selectbox("Weather type", list(WEATHER.values()), index=0)
    temp_c = st.slider("Temperature (°C, dataset scale)", TEMP_MIN, TEMP_MAX, 22.0, 0.5,
                       help="Converted with the dataset's own scale: °C ÷ 41 (see 'About the inputs' below).")
    humidity = st.slider("Humidity (%)", HUM_MIN, HUM_MAX, 50)
    wind = st.slider("Wind speed (km/h, dataset scale)", WIND_MIN, WIND_MAX, 10.0, 0.5,
                     help="Only wind speeds seen in the training data are allowed.")

month = MONTHS.index(month_name) + 1
weekday = WEEKDAYS.index(weekday_name)
weather = [k for k, v in WEATHER.items() if v == weather_label][0]

row = build_row(year, month, hour, weekday, holiday, weather, temp_c, humidity, wind)
errors, warnings = validate(row)

derived_working = "working day" if int(row["workingday"].iloc[0]) else "non-working day (weekend or holiday)"
st.caption(f"Derived automatically: **{SEASONS[int(row['season'].iloc[0])]}** season, **{derived_working}**.")

st.divider()
st.subheader("Predicted demand")

if errors:
    for e in errors:
        st.error(e)
else:
    for w in warnings:
        st.warning(w)
    try:
        raw_prediction = float(model.predict(row)[0])
        estimate = max(0, int(round(raw_prediction)))               # a count cannot be negative
        st.metric(f"Expected rentals at {hour}:00 on a {weekday_name} in {month_name} {year}", f"≈ {estimate} bikes")
        if raw_prediction < 0:
            st.caption(f"The model's raw output was {raw_prediction:.1f}; a rental count cannot be negative, so it is shown as 0 "
                       "(very quiet hours have only a handful of rentals).")
        st.caption(f"Typical size of the model's mistakes in 10-fold cross-validation: about **±{RMSE:.0f} bikes per hour** (RMSE). "
                   "This is an **estimate, not a guarantee** - real demand also depends on events and things the model cannot see.")

        # same conditions for the whole day
        day = pd.concat([row] * 24, ignore_index=True)
        day["hr"] = range(24)
        profile = pd.DataFrame({"Hour": range(24), "Predicted bikes": model.predict(day).clip(min=0).round(0)})
        line = alt.Chart(profile).mark_line(strokeWidth=2.5, color="#2a78d6").encode(
            x=alt.X("Hour:Q", title="Hour of day", scale=alt.Scale(domain=[0, 23]), axis=alt.Axis(tickMinStep=1, values=list(range(0, 24, 2)))),
            y=alt.Y("Predicted bikes:Q", title="Predicted bikes per hour"),
            tooltip=["Hour", "Predicted bikes"])
        marker = alt.Chart(profile[profile["Hour"] == hour]).mark_circle(size=140, color="#d9772b").encode(x="Hour:Q", y="Predicted bikes:Q")
        st.markdown("**Same conditions across the whole day**")
        st.altair_chart((line + marker).properties(width="container", height=280))
    except Exception as err:
        st.error(f"Prediction failed: {err}")

# --------------------------------------------------------------------------- information
with st.expander("About the inputs"):
    st.markdown(
        "- The model was trained on **normalised** values. The app converts your inputs with the scales stated in the dataset's Readme: "
        "temperature ÷ 41, humidity ÷ 100, wind speed ÷ 67. The Readme's description of these scales is not fully consistent across sources, "
        "so treat the °C and km/h numbers as **dataset-scale values**, not exact physical measurements.\n"
        "- The *feels-like* temperature is not asked for; it is estimated from the temperature with a straight line fitted on the training data.\n"
        "- *Season* and *working day* are derived from your month, weekday and holiday choice.\n"
        "- Only information available **before** the hour is used. The counts of casual / registered users are never inputs (they would leak the answer).")

with st.expander("About the model"):
    st.markdown(f"**Selected model:** {meta['model_name']} - the algorithm with the lowest mean RMSE among five compared with 10-fold cross-validation "
                f"on all {meta['n_training_rows']:,} hourly records.")
    table = pd.DataFrame(meta["all_models_cv"]).T.rename(columns={
        "mean_rmse": "Mean RMSE", "std_rmse": "Std RMSE", "mean_mae": "Mean MAE", "mean_r2": "Mean R²"}).sort_values("Mean RMSE")
    st.dataframe(table.style.format({"Mean RMSE": "{:.2f}", "Std RMSE": "{:.2f}", "Mean MAE": "{:.2f}", "Mean R²": "{:.3f}"}))
    st.caption("RMSE and MAE are in bike rentals per hour (lower is better). These are cross-validation results on held-out folds, "
               "not training errors.")
    st.markdown("**Limitations:** data from one city and two years only; neighbouring hours are similar, so real-world error on completely new "
                "days can be higher than the cross-validation figure; unusual events (storms, festivals, strikes) are not modelled.")
    if trained_with and trained_with != sklearn.__version__:
        st.warning(f"Model trained with scikit-learn {trained_with}, running on {sklearn.__version__}.")

st.caption("Dataset: Fanaee-T & Gama (2013), UCI Bike Sharing Dataset. Educational project - not for operational use.")
