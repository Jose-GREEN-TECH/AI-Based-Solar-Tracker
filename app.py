import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

KAMPALA_TZ = ZoneInfo("Africa/Kampala")
import os

# --- Page Config & Styling ---
st.set_page_config(
    page_title="Solar Tracker AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern UI design
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f, #252542);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e0e0ff !important;
        font-weight: 700;
    }
    h1 {
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 2.8rem;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }

    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 3rem !important;
        color: #00ffcc !important;
        text-shadow: 0px 0px 10px rgba(0,255,204,0.4);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.2rem !important;
        color: #a0a0c0 !important;
    }

    /* Custom prediction value display */
    .pred-label {
        font-size: 1.1rem;
        color: #a0a0c0;
        margin-bottom: 8px;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    .pred-value {
        font-size: 3.2rem;
        font-weight: 700;
        color: #00ffcc;
        text-shadow: 0px 0px 18px rgba(0,255,204,0.6);
        line-height: 1.1;
    }

    /* Container Styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        margin-bottom: 20px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-5px);
    }
    
    /* Button Customization */
    div.stButton > button {
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 12px;
        box-shadow: 0 4px 15px 0 rgba(255, 65, 108, 0.75);
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #ff4b2b, #ff416c);
        box-shadow: 0 6px 20px 0 rgba(255, 65, 108, 0.9);
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(__file__)
    tilt_model_path = os.path.join(base_dir, 'best_solar_tracker_tilt_rf.pkl')
    az_model_path = os.path.join(base_dir, 'best_solar_tracker_az_rf.pkl')
    
    try:
        rf_tilt = joblib.load(tilt_model_path)
        rf_az = joblib.load(az_model_path)
        return rf_tilt, rf_az
    except Exception as e:
        st.error(f"Error loading models. Please ensure the .pkl files are in the same directory as app.py. Details: {e}")
        return None, None

rf_tilt, rf_az = load_models()

# --- Helper Functions ---
def get_time_features(dt_obj):
    total_minutes = dt_obj.hour * 60 + dt_obj.minute
    sin_minute = np.sin(2 * np.pi * total_minutes / 1440.0)
    cos_minute = np.cos(2 * np.pi * total_minutes / 1440.0)
    
    day_of_year = dt_obj.timetuple().tm_yday
    # handle leap years approximately
    sin_day = np.sin(2 * np.pi * day_of_year / 365.25)
    cos_day = np.cos(2 * np.pi * day_of_year / 365.25)
    
    return sin_minute, cos_minute, sin_day, cos_day

def fetch_open_meteo():
    url = "https://api.open-meteo.com/v1/forecast?latitude=0.3152&longitude=32.5816&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,surface_pressure&timezone=auto"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        return current
    except Exception as e:
        st.error(f"Failed to fetch live data: {e}")
        return None

# --- Main UI ---
st.markdown("<h1>☀️ Solar Tracker Optimal Angle Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a0a0c0; font-size: 1.2rem; margin-bottom: 3rem;'>Intelligent tracking for Kampala. Powered by Random Forest AI.</p>", unsafe_allow_html=True)

# Session State for inputs
if "live_data" not in st.session_state:
    st.session_state["live_data"] = {}

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3067/3067160.png", width=100)
st.sidebar.title("Configuration")

if st.sidebar.button("🔄 Fetch Live Weather (Kampala)"):
    with st.spinner("Fetching data from Open-Meteo..."):
        weather_data = fetch_open_meteo()
        if weather_data:
            st.session_state["live_data"] = weather_data
            st.sidebar.success("Weather data updated!")

# Setup default values based on session_state or typical defaults
ld = st.session_state["live_data"]

st.sidebar.markdown("### 🌤️ Weather Parameters")
ghi = st.sidebar.slider("GHI (W/m² - Shortwave)", 0.0, 1500.0, float(ld.get("shortwave_radiation", 500.0)))
dni = st.sidebar.slider("DNI (W/m² - Direct Normal)", 0.0, 1500.0, float(ld.get("direct_normal_irradiance", 300.0)))
dhi = st.sidebar.slider("DHI (W/m² - Diffuse)", 0.0, 1000.0, float(ld.get("diffuse_radiation", 150.0)))
temperature = st.sidebar.slider("Temperature (°C)", 10.0, 45.0, float(ld.get("temperature_2m", 25.0)))
relative_humidity = st.sidebar.slider("Relative Humidity (%)", 0.0, 100.0, float(ld.get("relative_humidity_2m", 60.0)))
pressure = st.sidebar.slider("Pressure (hPa)", 800.0, 1100.0, float(ld.get("surface_pressure", 1010.0)) if ld.get("surface_pressure") else 900.0) # Kampala is high altitude (~1190m => ~880 hPa), fallback to 900 if missing
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 100.0, float(ld.get("wind_speed_10m", 10.0)))

st.sidebar.markdown("### 🔬 Derived Inputs")
precipitable_water = st.sidebar.slider("Precipitable Water (cm)", 0.0, 10.0, 3.5)
# Note: Clearness index requires extraterrestrial radiation calculation. Usually it's ~0.5 for partly cloudy. 
# If GHI is high, Kt is high (e.g. 0.7-0.8). Let's use a rough estimate if auto-filled.
suggested_kt = min(max(ghi / 1000.0, 0.1), 0.9) if ghi > 0 else 0.5
clearness_index_kt = st.sidebar.slider("Clearness Index (Kt)", 0.0, 1.0, float(suggested_kt))

st.sidebar.markdown("### ⏳ Time Selection")
use_current_time = st.sidebar.checkbox("Use Current Local Time", value=True)

if use_current_time:
    selected_time = datetime.now(KAMPALA_TZ)
    st.sidebar.info(f"Using time: {selected_time.strftime('%Y-%m-%d %H:%M')} (EAT)")
else:
    t_date = st.sidebar.date_input("Select Date", datetime.now(KAMPALA_TZ))
    t_time = st.sidebar.time_input("Select Time", datetime.now(KAMPALA_TZ))
    selected_time = datetime.combine(t_date, t_time)

sin_minute, cos_minute, sin_day, cos_day = get_time_features(selected_time)

# Feature Array Construction
# Expected order: 'GHI', 'DNI', 'DHI', 'Temperature', 'Precipitable Water', 'Relative Humidity', 'Pressure', 'Wind Speed', 'sin_minute', 'cos_minute', 'sin_day', 'cos_day', 'clearness_index_kt'
features = np.array([[
    ghi, dni, dhi, temperature, precipitable_water, relative_humidity,
    pressure, wind_speed, sin_minute, cos_minute, sin_day, cos_day, clearness_index_kt
]])

# Layout for predictions
col1, col2 = st.columns(2)

with col1:
    if rf_tilt is not None:
        try:
            pred_tilt = rf_tilt.predict(features)[0]
            st.markdown(f"""
            <div class='metric-container'>
                <div class='pred-label'>🎯 Predicted Optimal Tilt</div>
                <div class='pred-value'>{pred_tilt:.2f}°</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.warning("Model not loaded.")
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if rf_az is not None:
        try:
            pred_az = rf_az.predict(features)[0]
            st.markdown(f"""
            <div class='metric-container'>
                <div class='pred-label'>🧭 Predicted Optimal Azimuth</div>
                <div class='pred-value'>{pred_az:.2f}°</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.warning("Model not loaded.")
        st.markdown("</div>", unsafe_allow_html=True)

# Add an interactive chart or data preview below
st.markdown("### 📊 Input Feature Summary")
feature_names = ['GHI', 'DNI', 'DHI', 'Temperature', 'Precip. Water', 'Rel. Humidity', 'Pressure', 'Wind Speed', 'sin_minute', 'cos_minute', 'sin_day', 'cos_day', 'Kt']
df_features = pd.DataFrame(features, columns=feature_names)
st.dataframe(df_features, use_container_width=True)

st.markdown("<br><hr><p style='text-align: center; color: gray;'>Designed for Kampala, Uganda Solar Tracking Optimization.</p>", unsafe_allow_html=True)
