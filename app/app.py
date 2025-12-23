"""
app/app.py — US Flights Delay — ULTRA PREMIUM DARK EDITION (VERSION FINALE PATCHÉE)

- Mode sombre complet (CSS custom + animations)
- Loader animé "avion" pendant l'inférence

Onglets :
    1) 🔮 Prédiction (formulaire complet + météo + carte aéroports)
    2) 🌦️ Météo US (carte météo Plotly interactive sur les USA)
    3) 📊 Explicabilité SHAP (images PNG + force plot HTML)
    4) 🕒 Historique (session + export CSV)

- Appel API FastAPI /predict (async via httpx)
- Fallback automatique sur modèles locaux joblib
- Historique persisté dans reports/predictions_history.csv
"""

# ============================================================
# FIX IMPORT PATH (Streamlit / Windows / Docker safe)
# ============================================================

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    ROOT,
    DATA_PROCESSED_DIR,
    WEATHER_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)


# ============================================================
# 0 — Constantes & chemins
# ============================================================

st.set_page_config(
    page_title="US Flights Delay — Prediction Lab",
    page_icon="✈️",
    layout="wide",
)

SHAP_DIR = REPORTS_DIR / "shap"
HISTORY_PATH = REPORTS_DIR / "predictions_history.csv"
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

CARRIER_PRESETS = {
    "AA": "American Airlines Inc.",
    "DL": "Delta Air Lines Inc.",
    "UA": "United Air Lines Inc.",
    "9E": "Endeavor Air Inc.",
}

AIRPORT_PRESETS = {
    "JFK": "New York, NY: John F. Kennedy International",
    "ATL": "Atlanta, GA: Hartsfield-Jackson Atlanta International",
    "DHN": "Dothan, AL: Dothan Regional",
}

# session_state pour l'historique
if "history" not in st.session_state:
    if HISTORY_PATH.is_file():
        try:
            st.session_state["history"] = pd.read_csv(HISTORY_PATH).to_dict("records")
        except Exception:
            st.session_state["history"] = []
    else:
        st.session_state["history"] = []
        
from src.features import get_feature_columns

def check_api_health(api_url: str) -> bool:
    try:
        r = httpx.get(api_url.replace("/predict", "/health"), timeout=2)
        return r.status_code == 200
    except Exception:
        return False

# ============================================================
# 1 — CSS DARK + Animations + Loader
# ============================================================

DARK_CSS = """
<style>
/* GLOBAL DARK THEME */
html, body, .main, .block-container {
    background-color: #020617 !important; /* slate-950 */
    color: #e5e7eb !important;
}

.block-container {
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1500px !important;
}

/* TITLES & TEXT */
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #e5e7eb !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: radial-gradient(circle at top left, #0f172a, #020617) !important;
}
section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* INPUTS / SELECT / NUMBER / TEXT */
input, textarea {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border-radius: 6px !important;
}
div[data-baseweb="select"] > div {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border-radius: 6px !important;
    border: 1px solid #4b5563 !important;
}
ul[role="listbox"] {
    background-color: #020617 !important;
}
ul[role="listbox"] li {
    color: #e5e7eb !important;
}
ul[role="listbox"] li:hover {
    background-color: #22d3ee !important;
    color: #020617 !important;
}

/* BUTTONS */
button[kind="primary"] {
    background: linear-gradient(90deg, #22c55e, #06b6d4) !important;
    color: #020617 !important;
    border-radius: 999px !important;
    border: none !important;
    font-weight: 600 !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
button[kind="primary"]:hover {
    transform: translateY(-1px) scale(1.01);
    box-shadow: 0 10px 25px rgba(34,197,94,0.25);
}

/* METRICS CARDS */
div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
    color: #e5e7eb !important;
}
div[data-testid="stMetric"] {
    background: radial-gradient(circle at top left, #0f172a, #020617);
    padding: 1rem 1.2rem;
    border-radius: 1rem;
    border: 1px solid #1e293b;
    box-shadow: 0 18px 45px rgba(15,23,42,0.85);
    backdrop-filter: blur(10px);
}

/* TABS */
button[data-baseweb="tab"] {
    background-color: #020617 !important;
    color: #9ca3af !important;
    border-radius: 999px !important;
    margin-right: 0.5rem !important;
    border: 1px solid transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, #22c55e, #06b6d4) !important;
    color: #020617 !important;
    border-color: transparent !important;
}

/* HEADER BANNER */
.main-header {
    background: linear-gradient(120deg, #22c55e, #06b6d4);
    padding: 1.2rem 2rem;
    border-radius: 1.2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 20px 40px rgba(8,47,73,0.6);
}
.main-header-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #020617 !important;
}
.main-header-pill {
    background-color: #020617;
    color: #a5f3fc !important;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* LOADER OVERLAY */
.loader-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle at top, rgba(15,23,42,0.98), rgba(15,23,42,0.96));
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
}
.loader-content {
    text-align: center;
    color: #e5e7eb;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text";
}
.plane {
    width: 80px;
    height: 80px;
    margin: 0 auto 1rem auto;
    position: relative;
}
.plane::before {
    content: "✈️";
    font-size: 3rem;
    position: absolute;
    left: 0;
    top: 0;
    animation: fly 1.4s ease-in-out infinite;
}
@keyframes fly {
    0% { transform: translateX(-40px) translateY(20px) rotate(-15deg); opacity: 0; }
    30% { opacity: 1; }
    50% { transform: translateX(0) translateY(-10px) rotate(0deg); }
    100% { transform: translateX(60px) translateY(-30px) rotate(10deg); opacity: 0; }
}

/* PROGRESS BAR */
div[role="progressbar"] > div {
    background: linear-gradient(90deg, #22c55e, #06b6d4) !important;
}

/* DATAFRAME */
div[data-testid="stDataFrame"] {
    background-color: #020617 !important;
    border-radius: 1rem;
    border: 1px solid #1f2933;
}

/* PLOTLY BACKGROUND */
.js-plotly-plot .plotly .bg {
    fill: #020617 !important;
}

/* RISK BADGES */
.risk-wrapper {
    margin-top: 0.5rem;
}
.risk-label {
    font-size: 0.85rem;
    opacity: 0.75;
    margin-bottom: 0.2rem;
}
.risk-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    font-size: 0.9rem;
    font-weight: 600;
}
.risk-dot {
    width: 10px;
    height: 10px;
    border-radius: 999px;
}
.risk-score {
    font-size: 0.8rem;
    opacity: 0.9;
}

/* Low risk = green */
.risk-pill-low {
    background: rgba(22,163,74,0.18);
    border: 1px solid #22c55e;
    color: #bbf7d0;
}
.risk-pill-low .risk-dot {
    background: #22c55e;
    box-shadow: 0 0 10px rgba(34,197,94,0.9);
}

/* Medium risk = amber */
.risk-pill-medium {
    background: rgba(234,179,8,0.18);
    border: 1px solid #eab308;
    color: #fef9c3;
}
.risk-pill-medium .risk-dot {
    background: #eab308;
    box-shadow: 0 0 10px rgba(234,179,8,0.9);
}

/* High risk = red */
.risk-pill-high {
    background: rgba(220,38,38,0.18);
    border: 1px solid #ef4444;
    color: #fecaca;
}
.risk-pill-high .risk-dot {
    background: #ef4444;
    box-shadow: 0 0 10px rgba(239,68,68,0.9);
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ============================================================
# 2 — Utils modèles & SHAP (fichiers)
# ============================================================

_classifier: Optional[object] = None
_regressor: Optional[object] = None


def load_local_models():
    """Charge les modèles XGBoost (classifier + regressor) depuis MODELS_DIR."""
    global _classifier, _regressor

    if _classifier is None:
        clf_path = MODELS_DIR / "xgb_classifier_optuna_weather.joblib"
        if clf_path.is_file():
            _classifier = joblib.load(clf_path)
        else:
            st.error(f"❌ Modèle classifier introuvable : {clf_path}")

    if _regressor is None:
        reg_path = MODELS_DIR / "xgb_regressor_optuna_weather.joblib"
        if reg_path.is_file():
            _regressor = joblib.load(reg_path)
        else:
            st.error(f"❌ Modèle regressor introuvable : {reg_path}")

    return _classifier, _regressor


def shap_file(name: str) -> Optional[Path]:
    """Retourne le chemin d'un fichier SHAP s'il existe."""
    p = SHAP_DIR / name
    return p if p.is_file() else None


# ============================================================
# 3 — Data loaders (météo, aéroports, historique)
# ============================================================

@st.cache_data
def load_weather_df() -> Optional[pd.DataFrame]:
    """Charge flight_weather.csv (Open-Meteo) ou fallback train_weather.csv."""
    primary = WEATHER_DIR / "flight_weather.csv"
    if primary.is_file():
        try:
            return pd.read_csv(primary)
        except Exception:
            return None

    fallback = DATA_PROCESSED_DIR / "train_weather.csv"
    if fallback.is_file():
        try:
            return pd.read_csv(fallback)
        except Exception:
            return None
    return None

st.cache_data.clear()

@st.cache_data
@st.cache_data
def load_airports_df() -> Optional[pd.DataFrame]:
    csv = DATA_PROCESSED_DIR / "airports_with_coords.csv"
    if not csv.is_file():
        return None
    try:
        return pd.read_csv(csv)
    except Exception:
        return None

    df = pd.read_csv(csv)

    # NORMALISATION
    if "latitude_deg" in df.columns and "longitude_deg" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude_deg"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude_deg"], errors="coerce")
    else:
        st.error("❌ latitude_deg / longitude_deg absentes du CSV")
        return None

    # DROP INVALID
    df = df.dropna(subset=["latitude", "longitude"])

    if df.empty:
        st.error("❌ Aucune coordonnée valide après nettoyage")
        return None

    return df
        
@st.cache_data
def build_airport_lookup() -> dict:
    """
    Construit un mapping fiable :
    airport_code (IATA/FAA) -> airport_name lisible
    """
    df = load_airports_df()
    if df is None or df.empty:
        return {}

    code_cols = [c for c in ["iata_code", "faa_code", "ident"] if c in df.columns]

    lookup = {}
    for _, row in df.iterrows():
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            continue

        city = row.get("municipality")
        region = row.get("iso_region")

        label = name
        if isinstance(city, str):
            label = f"{city}: {label}"
        if isinstance(region, str):
            label = f"{label} ({region})"

        for c in code_cols:
            code = row.get(c)
            if isinstance(code, str) and code.strip():
                lookup[code.strip().upper()] = label

    return lookup


    # Normalisation latitude / longitude
    if "latitude_deg" in df.columns and "longitude_deg" in df.columns:
        df = df.rename(columns={"latitude_deg": "lat", "longitude_deg": "lon"})
    else:
        # Fallback générique si le fichier change plus tard
        for c in ["lat", "latitude", "latitude_deg"]:
            if c in df.columns:
                df = df.rename(columns={c: "lat"})
                break
        for c in ["lon", "longitude", "longitude_deg"]:
            if c in df.columns:
                df = df.rename(columns={c: "lon"})
                break

    if "lat" not in df.columns or "lon" not in df.columns:
        st.warning("⚠️ Colonnes lat/lon introuvables dans airports_with_coords.csv.")
        return df

    return df


@st.cache_data
def load_train_weather_for_shap() -> Optional[pd.DataFrame]:
    """Charge train_weather.csv pour cohérence avec la pipeline SHAP."""
    csv = DATA_PROCESSED_DIR / "train_weather.csv"
    if csv.is_file():
        try:
            return pd.read_csv(csv)
        except Exception:
            return None
    return None


def append_history(entry: dict) -> None:
    """Ajoute une ligne d'historique en session + CSV."""
    st.session_state["history"].append(entry)
    df_hist = pd.DataFrame(st.session_state["history"])
    df_hist.to_csv(HISTORY_PATH, index=False)


# ============================================================
# 4 — Appel API (async) + loader avion
# ============================================================

async def _call_api_async(payload: dict):
    async with httpx.AsyncClient(timeout=6) as client:
        resp = await client.post(API_URL, json=payload)
        if resp.status_code == 200:
            return resp.json()
        st.warning(f"⚠️ API a renvoyé le status {resp.status_code}")
        return None


def call_api_with_loader(payload: dict):
    """Affiche le loader avion pendant la requête API / fallback."""
    loader_placeholder = st.empty()
    loader_html = """
    <div class="loader-overlay">
        <div class="loader-content">
            <div class="plane"></div>
            <div style="font-size:1.1rem;margin-bottom:0.3rem;">Calcul des prédictions...</div>
            <div style="opacity:0.7;">XGBoost + Optuna + météo en cours d'exécution</div>
        </div>
    </div>
    """
    loader_placeholder.markdown(loader_html, unsafe_allow_html=True)

    try:
        try:
            result = asyncio.run(_call_api_async(payload))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(_call_api_async(payload))
    except Exception:
        result = None

    loader_placeholder.empty()
    return result

def check_mlflow_health() -> bool:
    urls = [
        os.getenv("MLFLOW_UI_URL", "http://flights_mlflow:5000"),
        "http://localhost:5000",
    ]
    for url in urls:
        try:
            r = httpx.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            continue
    return False
# ============================================================
# 5 — HEADER & Tabs
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <div>
            <div class="main-header-title">✈️ US Flights Delay — Prediction Lab</div>
            <div class="main-header-pill">XGBoost · Optuna · Météo Open-Meteo · MLflow · SHAP</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_pred, tab_weather, tab_shap, tab_history = st.tabs(
    ["🔮 Prédiction", "🌦️ Météo US", "📊 Explicabilité SHAP", "🕒 Historique"]
)

api_up = check_api_health(API_URL)

status_color = "#22c55e" if api_up else "#ef4444"
status_text  = "API UP" if api_up else "API DOWN"

st.markdown(
    f"""
    <div style="
        display:flex;
        align-items:center;
        gap:0.5rem;
        margin-bottom:0.75rem;
    ">
        <span style="
            width:10px;
            height:10px;
            border-radius:50%;
            background:{status_color};
            box-shadow:0 0 8px {status_color};
        "></span>
        <span style="font-size:0.85rem;opacity:0.85;">
            FastAPI status: <strong>{status_text}</strong>
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

mlflow_up = check_mlflow_health()

mlflow_color = "#22c55e" if mlflow_up else "#ef4444"
mlflow_text  = "MLflow CONNECTÉ" if mlflow_up else "MLflow OFFLINE"

st.markdown(
    f"""
    <div style="
        display:flex;
        align-items:center;
        gap:0.5rem;
        margin-bottom:0.75rem;
    ">
        <span style="
            width:10px;
            height:10px;
            border-radius:50%;
            background:{mlflow_color};
            box-shadow:0 0 8px {mlflow_color};
        "></span>
        <span style="font-size:0.85rem;opacity:0.85;">
            {mlflow_text}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
# ============================================================
# 6 — TAB 1 : PRÉDICTION
# ============================================================

with tab_pred:
    clf_model, reg_model = load_local_models()

    left_col, right_col = st.columns([2, 1])

    # ========================================================
    # LEFT COLUMN — FORM + SELECTEURS
    # ========================================================
    with left_col:

        # ====================================================
        # 🔥 SÉLECTEURS DYNAMIQUES (HORS FORM — OK)
        # ====================================================

        carrier_code = st.selectbox(
            "carrier",
            options=list(CARRIER_PRESETS.keys()),
            key="carrier_code"
        )
        carrier_name = CARRIER_PRESETS[carrier_code]

        airport_code = st.selectbox(
            "airport",
            options=list(AIRPORT_PRESETS.keys()),
            key="airport_code"
        )
        airport_name = AIRPORT_PRESETS[airport_code]

        st.markdown(
            f"""
            **carrier_name**  
            <span style='color:#9ca3af'>{carrier_name}</span>

            **airport_name**  
            <span style='color:#9ca3af'>{airport_name}</span>
            """,
            unsafe_allow_html=True,
        )

        # ====================================================
        # FORM — INCHANGÉ
        # ====================================================

        with st.form("flight_form"):
            st.subheader("Données de trafic agrégé")

            c1, c2, c3 = st.columns(3)
            with c1:
                year = st.number_input("year", min_value=1980, max_value=2100, value=2018)
                month = st.number_input("month", min_value=1, max_value=12, value=4)
                arr_flights = st.number_input("arr_flights", min_value=0, value=100)
                arr_del15 = st.number_input("arr_del15", min_value=0, value=10)

            with c2:
                carrier_ct = st.number_input("carrier_ct", min_value=0, value=5)
                weather_ct = st.number_input("weather_ct", min_value=0, value=5)
                nas_ct = st.number_input("nas_ct", min_value=0, value=5)
                security_ct = st.number_input("security_ct", min_value=0, value=0)
                late_aircraft_ct = st.number_input(
                    "late_aircraft_ct", min_value=0, value=5
                )

            st.markdown("### Retards (minutes)")

            cA, cB, cC = st.columns(3)
            with cA:
                arr_delay = st.number_input("arr_delay", min_value=0.0, value=50.0)
                carrier_delay = st.number_input(
                    "carrier_delay", min_value=0.0, value=10.0
                )
            with cB:
                weather_delay = st.number_input(
                    "weather_delay", min_value=0.0, value=15.0
                )
                nas_delay = st.number_input("nas_delay", min_value=0.0, value=10.0)
            with cC:
                security_delay = st.number_input(
                    "security_delay", min_value=0.0, value=0.0
                )
                late_aircraft_delay = st.number_input(
                    "late_aircraft_delay", min_value=0.0, value=15.0
                )

            st.markdown("### Météo agrégée (mois / aéroport)")

            m1, m2, m3 = st.columns(3)
            with m1:
                temperature_2m_mean = st.number_input(
                    "temperature_2m_mean (°C)",
                    min_value=-50.0,
                    max_value=60.0,
                    value=15.0,
                )
            with m2:
                precipitation_sum = st.number_input(
                    "precipitation_sum (mm)", min_value=0.0, value=0.0
                )
            with m3:
                windspeed_10m_max = st.number_input(
                    "windspeed_10m_max (km/h)", min_value=0.0, value=20.0
                )

            submitted = st.form_submit_button("🔮 Lancer la prédiction")

    # ========================================================
    # RIGHT COLUMN — 🗺️ CARTE (FIX FINAL)
    # ========================================================
    with right_col:
        st.subheader("🗺️ Carte interactive des aéroports (USA)")

        airports_df = load_airports_df()

        if airports_df is None or airports_df.empty:
            st.warning("Impossible d'afficher la carte des aéroports.")
        else:
            # ⚠️ st.map EXIGE latitude / longitude
            map_df = (
                airports_df[["latitude_deg", "longitude_deg"]]
                .rename(
                    columns={
                        "latitude_deg": "latitude",
                        "longitude_deg": "longitude",
                    }
                )
                .dropna()
            )

            st.map(map_df, use_container_width=True)

    # ========================================================
    # PAYLOAD — STRICTEMENT INCHANGÉ
    # ========================================================
    if submitted:
        payload = {
            "year": year,
            "month": month,
            "carrier": carrier_code,
            "carrier_name": carrier_name,
            "airport": airport_code,
            "airport_name": airport_name,
            "arr_flights": arr_flights,
            "arr_del15": arr_del15,
            "carrier_ct": carrier_ct,
            "weather_ct": weather_ct,
            "nas_ct": nas_ct,
            "security_ct": security_ct,
            "late_aircraft_ct": late_aircraft_ct,
            "arr_cancelled": 0,
            "arr_diverted": 0,
            "arr_delay": arr_delay,
            "carrier_delay": carrier_delay,
            "weather_delay": weather_delay,
            "nas_delay": nas_delay,
            "security_delay": security_delay,
            "late_aircraft_delay": late_aircraft_delay,
            "temperature_2m_mean": temperature_2m_mean,
            "precipitation_sum": precipitation_sum,
            "windspeed_10m_max": windspeed_10m_max,
        }

        result = call_api_with_loader(payload)


        # 1) API + loader avion
        result = call_api_with_loader(payload)

        # 2) fallback local si API HS
        if result is None:
            st.info("⛔ API indisponible → utilisation des modèles locaux.")
            if arr_flights > 0:
                delay_rate = arr_del15 / arr_flights
                avg_delay_pf = arr_delay / arr_flights
            else:
                delay_rate = 0.0
                avg_delay_pf = 0.0

            df_sample = pd.DataFrame(
                [
                    {
                        **payload,
                        "delay_rate": delay_rate,
                        "avg_delay_per_flight": avg_delay_pf,
                    }
                ]
            )

            feature_cols = get_feature_columns()
            missing = [c for c in feature_cols if c not in df_sample.columns]
            if missing:
                st.error(f"Colonnes manquantes pour le modèle (fallback) : {missing}")
            else:
                X = df_sample[feature_cols]
                proba = float(clf_model.predict_proba(X)[0, 1])
                label = int(proba >= 0.5)
                avg_delay_pred = float(reg_model.predict(X)[0])

                result = {
                    "high_delay_risk_proba": proba,
                    "high_delay_risk_label": label,
                    "avg_delay_per_flight": avg_delay_pred,
                }

        # 3) Affichage + historique
        if result is not None:
            st.success("🎯 Prédiction réussie !")

            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric(
                    "Probabilité High Delay Risk",
                    f"{result['high_delay_risk_proba']:.3f}",
                )
            with r2:
                st.metric(
                    "Label High Delay Risk", int(result["high_delay_risk_label"])
                )
            with r3:
                st.metric(
                    "Avg Delay per Flight (prédit)",
                    f"{result['avg_delay_per_flight']:.1f} min",
                )

            risk = float(result["high_delay_risk_proba"])

            # ---- Visuel tri-couleur (vert / jaune / rouge) ----
            if risk < 0.3:
                risk_text = "Risque faible"
                css_class = "risk-pill-low"
                risk_desc = "Trafic globalement fluide, retards sévères peu probables."
            elif risk < 0.7:
                risk_text = "Risque modéré"
                css_class = "risk-pill-medium"
                risk_desc = "Surveiller les retards : conditions météo / trafic moyennes."
            else:
                risk_text = "Risque élevé"
                css_class = "risk-pill-high"
                risk_desc = "Fort risque de retards importants, prévoir buffers et alertes."

            st.write("### Niveau de risque global")

            risk_html = f"""
            <div class="risk-wrapper">
                <div class="risk-label">Score modèle (high_delay_risk)</div>
                <div class="risk-pill {css_class}">
                    <span class="risk-dot"></span>
                    <span class="risk-level">{risk_text}</span>
                    <span class="risk-score">{risk*100:.1f}%</span>
                </div>
                <div class="risk-caption" style="margin-top:0.25rem;font-size:0.8rem;opacity:0.8;">
                    {risk_desc}
                </div>
            </div>
            """
            st.markdown(risk_html, unsafe_allow_html=True)

            # Barre de progression
            st.progress(risk)

            # Historique
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "year": year,
                "month": month,
                "carrier": carrier_code,
                "airport": airport_code,
                "arr_flights": arr_flights,
                "arr_del15": arr_del15,
                "high_delay_risk_proba": risk,
                "high_delay_risk_label": int(result["high_delay_risk_label"]),
                "avg_delay_per_flight": float(result["avg_delay_per_flight"]),
            }
            append_history(entry)
            st.caption(
                f"Historique mis à jour — total {len(st.session_state['history'])} prédictions."
            )

# ============================================================
# 7 — TAB 2 : MÉTÉO US (carte interactive)
# ============================================================

with tab_weather:
    st.subheader("🌦️ Météo agrégée — USA (dataset Open-Meteo)")

    weather_df = load_weather_df()
    if weather_df is None or weather_df.empty:
        st.warning(
            "Impossible de charger les données météo. "
            "Vérifie `data/weather/flight_weather.csv` ou `data/processed/train_weather.csv`."
        )
    else:
        # Aperçu brut
        st.dataframe(weather_df.head(200), width="stretch")

        # Normalisation des noms de colonnes
        if "month" not in weather_df.columns and "month_x" in weather_df.columns:
            weather_df["month"] = weather_df["month_x"]
        if "airport" not in weather_df.columns and "airport_x" in weather_df.columns:
            weather_df["airport"] = weather_df["airport_x"]

        # Colonnes métriques possibles
        metric_candidates = {
            "Température moyenne (°C)": ["temperature_2m_mean", "temp_mean"],
            "Précipitations (mm)": ["precipitation_sum", "precip_sum"],
            "Vent max (km/h)": ["windspeed_10m_max", "windspeed_max"],
        }
        metrics: dict[str, str] = {}
        for label, candidates in metric_candidates.items():
            for c in candidates:
                if c in weather_df.columns:
                    metrics[label] = c
                    break

        if not metrics:
            st.error("Aucune colonne météo reconnue pour construire la carte.")
        else:
            metric_label = st.selectbox(
                "Variable météo à visualiser",
                list(metrics.keys()),
                index=0,
            )
            metric_col = metrics[metric_label]

            # Merge avec les coordonnées aéroports si besoin
            if ("lat" not in weather_df.columns or "lon" not in weather_df.columns):
                airports_df = load_airports_df()
                if airports_df is not None and not airports_df.empty:
                    if (
                        "faa_code" in airports_df.columns
                        and "lat" in airports_df.columns
                        and "lon" in airports_df.columns
                    ):
                        airports_geo = airports_df[
                            ["faa_code", "lat", "lon"]
                        ].rename(
                            columns={
                                "faa_code": "airport",
                            }
                        )
                        weather_df = weather_df.merge(
                            airports_geo, on="airport", how="left"
                        )

            if "lat" not in weather_df.columns or "lon" not in weather_df.columns:
                st.error(
                    "Impossible d'afficher la carte météo : colonnes `lat` / `lon` manquantes "
                    "même après fusion avec airports_with_coords."
                )
            else:
                df_map = weather_df.copy()

                # Filtre par année si dispo
                if "year" in df_map.columns:
                    years = sorted(df_map["year"].dropna().unique().tolist())
                    if years:
                        default_year = max(years)
                        year_sel = st.selectbox(
                            "Année à afficher",
                            years,
                            index=years.index(default_year),
                        )
                        df_map = df_map[df_map["year"] == year_sel]

                # Agrégation par aéroport
                agg_cols = ["airport", "lat", "lon", metric_col]
                df_map = (
                    df_map[agg_cols]
                    .dropna(subset=["lat", "lon"])
                    .groupby(["airport", "lat", "lon"], as_index=False)[metric_col]
                    .mean()
                )

                st.markdown("#### 🗺️ Carte météo des aéroports (Plotly Geo)")

                fig_geo = px.scatter_geo(
                    df_map,
                    lat="lat",
                    lon="lon",
                    color=metric_col,
                    hover_name="airport",
                    size=None,
                    color_continuous_scale="Turbo",
                    scope="north america",
                    labels={metric_col: metric_label},
                )
                fig_geo.update_geos(
                    showcountries=True,
                    showland=True,
                    landcolor="#020617",
                    bgcolor="#020617",
                    coastlinecolor="#4b5563",
                    lakecolor="#020617",
                )
                fig_geo.update_layout(
                    height=500,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="#020617",
                    plot_bgcolor="#020617",
                    font=dict(color="#e5e7eb", size=12),
                    coloraxis_colorbar=dict(
                        title=metric_label,
                        thickness=14,
                        bgcolor="rgba(2,6,23,0.9)",
                        tickcolor="#e5e7eb",
                        tickfont=dict(color="#e5e7eb", size=10),
                    ),
                )
                st.plotly_chart(fig_geo, width="stretch")

                # Petits KPIs
                m1, m2, m3 = st.columns(3)
                m1.metric("Min", f"{df_map[metric_col].min():.2f}")
                m2.metric("Moyenne", f"{df_map[metric_col].mean():.2f}")
                m3.metric("Max", f"{df_map[metric_col].max():.2f}")

# ============================================================
# 8 — TAB 3 : SHAP (Explicabilité)
# ============================================================

with tab_shap:
    st.subheader("📊 Explicabilité SHAP — XGBoost (Optuna + météo)")

    # on vérifie la présence de train_weather juste pour cohérence pipeline
    train_w = load_train_weather_for_shap()
    if train_w is None or train_w.empty:
        st.error(
            "Impossible d'afficher les graphiques SHAP : "
            "❌ `train_weather.csv` introuvable ou vide dans `data/processed`."
        )
    else:
        col1, col2 = st.columns(2)

        clf_summary = shap_file("classifier_summary.png")
        reg_summary = shap_file("regressor_summary.png")

        with col1:
            if clf_summary:
                st.markdown("**SHAP summary — Classifier**")
                st.image(str(clf_summary), width="stretch")
        with col2:
            if reg_summary:
                st.markdown("**SHAP summary — Regressor**")
                st.image(str(reg_summary), width="stretch")

        # top 20 bar plots
        bar_clf = shap_file("classifier_importance_bar.png")
        bar_reg = shap_file("regressor_importance_bar.png")
        if bar_clf or bar_reg:
            st.markdown("### Top 20 features (SHAP importance)")
            c3, c4 = st.columns(2)
            if bar_clf:
                with c3:
                    st.image(str(bar_clf), width="stretch")
            if bar_reg:
                with c4:
                    st.image(str(bar_reg), width="stretch")

        # force plot HTML interactif
        force_html_path = None
        for candidate in ["force_regressor_sample.html", "force_plot_reg_idx0.html"]:
            p = shap_file(candidate)
            if p:
                force_html_path = p
                break

        if force_html_path:
            st.markdown("### Force plot (régression — observation échantillon)")
            try:
                with open(force_html_path, "r", encoding="utf-8") as f:
                    html = f.read()
                st.components.v1.html(html, height=350, scrolling=True)
            except Exception as e:
                st.warning(f"Impossible de charger le force plot HTML : {e}")

# ============================================================
# 9 — TAB 4 : HISTORIQUE
# ============================================================

with tab_history:
    st.subheader("🕒 Historique des prédictions (session + export CSV)")

    hist = st.session_state["history"]
    if not hist:
        st.info("Aucune prédiction enregistrée pour l’instant.")
    else:
        df_hist = pd.DataFrame(hist).sort_values("timestamp", ascending=False)
        st.dataframe(df_hist, width="stretch")

        csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger l’historique en CSV",
            data=csv_bytes,
            file_name="us_flights_delay_history.csv",
            mime="text/csv",
        )

        if st.button("🧹 Vider l’historique (session + CSV)", type="secondary"):
            st.session_state["history"] = []
            if HISTORY_PATH.is_file():
                try:
                    HISTORY_PATH.unlink()
                except Exception:
                    pass
            st.experimental_rerun()

st.write("---")
st.caption("US Flights Delay — Streamlit App © Faycel & Francis · XGBoost · Optuna · Weather · SHAP")