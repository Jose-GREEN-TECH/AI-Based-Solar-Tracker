from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import requests
from datetime import datetime
import os

app = FastAPI(
    title="Solar Tracker Prediction API",
    description="REST API built for ESP32 and IoT integrations to fetch optimal solar panel angles automatically."
)

# --- Load Models ---
base_dir = os.path.dirname(__file__)
tilt_model_path = os.path.join(base_dir, 'best_solar_tracker_tilt_rf.pkl')
az_model_path = os.path.join(base_dir, 'best_solar_tracker_az_rf.pkl')

try:
    rf_tilt = joblib.load(tilt_model_path)
    rf_az = joblib.load(az_model_path)
except Exception as e:
    print(f"Error loading models: {e}")
    rf_tilt = None
    rf_az = None

# --- Helper Functions ---
def get_time_features(dt_obj):
    total_minutes = dt_obj.hour * 60 + dt_obj.minute
    sin_minute = np.sin(2 * np.pi * total_minutes / 1440.0)
    cos_minute = np.cos(2 * np.pi * total_minutes / 1440.0)
    
    day_of_year = dt_obj.timetuple().tm_yday
    sin_day = np.sin(2 * np.pi * day_of_year / 365.25)
    cos_day = np.cos(2 * np.pi * day_of_year / 365.25)
    
    return sin_minute, cos_minute, sin_day, cos_day

def fetch_open_meteo():
    url = "https://api.open-meteo.com/v1/forecast?latitude=0.3152&longitude=32.5816&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,surface_pressure&timezone=auto"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("current", {})
    except Exception as e:
        print(f"Open-Meteo Error: {e}")
        return None

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Solar Tracker API is running"}

@app.get("/get_angles")
def get_angles():
    """
    Fetches real-time weather data for Kampala, computes time features, 
    and returns predicted optimal Tilt and Azimuth.
    """
    if rf_tilt is None or rf_az is None:
        raise HTTPException(status_code=500, detail="Machine Learning models failed to load. Please check the .pkl files.")
        
    weather = fetch_open_meteo()
    if not weather:
        raise HTTPException(status_code=502, detail="Failed to fetch weather data from Open-Meteo API.")
        
    # Weather parsing
    ghi = float(weather.get("shortwave_radiation", 500.0))
    dni = float(weather.get("direct_normal_irradiance", 300.0))
    dhi = float(weather.get("diffuse_radiation", 150.0))
    temperature = float(weather.get("temperature_2m", 25.0))
    relative_humidity = float(weather.get("relative_humidity_2m", 60.0))
    # Kampala is high altitude (~880 hPa), fallback if missing
    pressure = float(weather.get("surface_pressure", 880.0))
    wind_speed = float(weather.get("wind_speed_10m", 10.0))
    
    # Derived inputs
    precipitable_water = 3.5 # Default typical estimate if missing from API
    clearness_index_kt = min(max(ghi / 1000.0, 0.1), 0.9) if ghi > 0 else 0.5
    
    # Time logic
    current_time = datetime.now()
    sin_minute, cos_minute, sin_day, cos_day = get_time_features(current_time)
    
    # Feature construction (Must match training features 1-to-1)
    # Order: GHI, DNI, DHI, Temperature, Precipitable Water, Relative Humidity, Pressure, Wind Speed, sin_minute, cos_minute, sin_day, cos_day, clearness_index_kt
    features = np.array([[
        ghi, dni, dhi, temperature, precipitable_water, relative_humidity,
        pressure, wind_speed, sin_minute, cos_minute, sin_day, cos_day, clearness_index_kt
    ]])
    
    # Predictions
    pred_tilt = rf_tilt.predict(features)[0]
    pred_az = rf_az.predict(features)[0]
    
    return {
        "timestamp": current_time.isoformat(),
        "input_features": {
             "ghi": ghi,
             "dni": dni,
             "dhi": dhi,
             "temperature": temperature,
             "wind_speed": wind_speed
        },
        "optimal_tilt": round(float(pred_tilt), 2),
        "optimal_azimuth": round(float(pred_az), 2)
    }
