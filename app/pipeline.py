import requests
import time
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from app.config import NASA_API_KEY
from app.database import save_to_db

# ======================
# FETCH DATA (NeoFeed)
# ======================
def fetch_data():
    start_date = datetime.today().strftime('%Y-%m-%d')
    end_date = (datetime.today() + timedelta(days=7)).strftime('%Y-%m-%d')

    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={NASA_API_KEY}"

    response = requests.get(url)

    if response.status_code != 200:
        print("❌ API Error:", response.text)
        return pd.DataFrame()

    data = response.json()
    asteroids = []

    for date in data.get('near_earth_objects', {}):
        for obj in data['near_earth_objects'][date]:
            asteroids.append({
                "neo_reference_id": obj["neo_reference_id"],
                "name": obj["name"],
                "absolute_magnitude_h": obj["absolute_magnitude_h"],
                "estimated_diameter_max_km": obj["estimated_diameter"]["kilometers"]["estimated_diameter_max"],
                "is_potentially_hazardous_asteroid": obj["is_potentially_hazardous_asteroid"],
                "event_date": date
            })

    return asteroids

# ==============================
# STEP 2 — SBDB
# ==============================
def get_sbdb_features(spk_id):
    url = f"https://ssd-api.jpl.nasa.gov/sbdb.api?spk={spk_id}"

    try:
        data = requests.get(url).json()
        orbit = data.get("orbit", {})

        return {
            "semi_major_axis_au": float(orbit.get("a")) if orbit.get("a") else None,
            "eccentricity": float(orbit.get("e")) if orbit.get("e") else None,
            "inclination_deg": float(orbit.get("i")) if orbit.get("i") else None,
            "perihelion_distance_au": float(orbit.get("q")) if orbit.get("q") else None,
            "aphelion_distance_au": float(orbit.get("ad")) if orbit.get("ad") else None,
            "mean_motion_deg_per_day": float(orbit.get("n")) if orbit.get("n") else None,
            "mean_anomaly_deg": float(orbit.get("ma")) if orbit.get("ma") else None,
            "ascending_node_longitude_deg": float(orbit.get("om")) if orbit.get("om") else None,
            "perihelion_argument_deg": float(orbit.get("w")) if orbit.get("w") else None,
            "moid": float(orbit.get("moid")) if orbit.get("moid") else None,
            "condition_code": float(orbit.get("condition_code")) if orbit.get("condition_code") else None,
            "rms": float(orbit.get("rms")) if orbit.get("rms") else None,
            "n_obs_used": float(orbit.get("n_obs_used")) if orbit.get("n_obs_used") else None,
            "data_arc_in_days": float(orbit.get("data_arc")) if orbit.get("data_arc") else None
        }
    except:
        return {}

# ==============================
# STEP 3 — BROWSE FALLBACK
# ==============================
def get_browse_features(neo_id):
    url = f"https://api.nasa.gov/neo/rest/v1/neo/{neo_id}?api_key={NASA_API_KEY}"

    try:
        res = requests.get(url).json()
        orbit = res.get("orbital_data", {})

        return {
            "semi_major_axis_au": float(orbit.get("semi_major_axis")) if orbit.get("semi_major_axis") else None,
            "eccentricity": float(orbit.get("eccentricity")) if orbit.get("eccentricity") else None,
            "inclination_deg": float(orbit.get("inclination")) if orbit.get("inclination") else None,
            "perihelion_distance_au": float(orbit.get("perihelion_distance")) if orbit.get("perihelion_distance") else None,
            "aphelion_distance_au": float(orbit.get("aphelion_distance")) if orbit.get("aphelion_distance") else None,
            "mean_motion_deg_per_day": float(orbit.get("mean_motion")) if orbit.get("mean_motion") else None,
            "mean_anomaly_deg": float(orbit.get("mean_anomaly")) if orbit.get("mean_anomaly") else None,
            "ascending_node_longitude_deg": float(orbit.get("ascending_node_longitude")) if orbit.get("ascending_node_longitude") else None,
            "perihelion_argument_deg": float(orbit.get("perihelion_argument")) if orbit.get("perihelion_argument") else None,
        }
    except:
        return {}

# ======================
# LOAD MODEL
# ======================
def load_model():
    return joblib.load("models/model.pkl")

def load_features():
    return joblib.load("models/features.pkl")

# ==============================
# STEP 4 — CHECK MISSING
# ==============================
def is_orbit_missing(data):
    return (
        data.get("semi_major_axis_au") is None or
        data.get("eccentricity") is None or
        data.get("inclination_deg") is None
    )

# ==============================
# STEP 5 — MERGE DATA
# ==============================
def get_complete_orbit(neo_id):
    sbdb = get_sbdb_features(neo_id)

    if is_orbit_missing(sbdb):
        print(f"[FALLBACK] {neo_id} → Browse API")
        browse = get_browse_features(neo_id)

        for key in browse:
            if sbdb.get(key) is None:
                sbdb[key] = browse[key]

    return sbdb

# ==============================
# STEP 6 — FEATURE ENGINEERING
# ==============================
def feature_engineering(df):
    #df['is_potentially_hazardous_asteroid'] = df['is_potentially_hazardous_asteroid'].astype(int)
    df['cross_earth'] = ((df['perihelion_distance_au'] < 1.0) & (df['aphelion_distance_au'] > 1.0)).astype(int)
    df['elongation'] = df['aphelion_distance_au'] - df['perihelion_distance_au']
    df['orbit_ratio'] = df['aphelion_distance_au'] / df['perihelion_distance_au']
    df['size'] = 10 ** (-0.2 * df['absolute_magnitude_h'])
    df['risk_score'] = df['size'] / df['perihelion_distance_au']
    df['velocity_est'] = (1 / df['semi_major_axis_au']) ** 0.5
    df['log_a'] = np.log(df['semi_major_axis_au'])
    df['log_q'] = np.log(df['perihelion_distance_au'])
    df['moid_risk'] = 1 / (df['moid'] + 1e-6)
    df['log_moid'] = np.log1p(df['moid'])
    df['orbit_quality_score'] = 1 / (df['condition_code'] + 1)
    df['obs_density'] = df['n_obs_used'] / (df['data_arc_in_days'] + 1)
    df['rms_stability'] = 1 / (df['rms'] + 1e-6)
    df['size_risk'] = df['estimated_diameter_max_km'] * df['moid_risk']
    df['eccentricity_risk'] = df['eccentricity'] * df['moid_risk']
    df['orbital_speed_proxy'] = df['mean_motion_deg_per_day'] * df['semi_major_axis_au']
    return df

# ======================
# MAIN PIPELINE
# ======================
def run_pipeline():
    print("Fetching NeoFeed...")
    asteroids = fetch_data()

    full_data = []

    for ast in asteroids:
        neo_id = ast["neo_reference_id"]

        orbit = get_complete_orbit(neo_id)

        combined = {**ast, **orbit}
        full_data.append(combined)

        time.sleep(0.1)  # avoid rate limit

    df = pd.DataFrame(full_data)

    print("Cleaning missing data...")
    # df = df.dropna()

    print("Feature engineering...")
    df = feature_engineering(df)

    model = load_model()
    features = load_features()

    df['prediction'] = model.predict(df[features])

    # metadata
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['created_at'] = datetime.now()


    df["is_potentially_hazardous_asteroid"] = df["is_potentially_hazardous_asteroid"].astype(bool)
    df["cross_earth"] = df["cross_earth"].astype(bool)
    # remove duplicate (extra safety)
    df = df.drop_duplicates(subset=['neo_reference_id', 'event_date'])

    save_to_db(df)

    print("✅ Pipeline selesai")

    return df

if __name__ == "__main__":
    df = run_pipeline()
    save_to_db(df)
    print(f"✅ Pipeline finished. Rows: {len(df)}")