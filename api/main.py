"""
api/main.py — API FastAPI pour le projet US-FLIGHTS-DELAY (VERSION PREMIUM)

- /health : check de vie
- /predict (POST) :
    * Input : colonnes officielles du dataset US Flights Delay
      + features météo déjà pré-calculées :
        - temperature_2m_mean
        - precipitation_sum
        - windspeed_10m_max
    * Recalcule :
        - delay_rate = arr_del15 / arr_flights
        - avg_delay_per_flight = arr_delay / arr_flights
    * Utilise les modèles finaux :
        - xgb_classifier_optuna_weather.joblib
        - xgb_regressor_optuna_weather.joblib
    * Retour :
        - high_delay_risk_proba
        - high_delay_risk_label (seuil 0.5)
        - avg_delay_per_flight (prédit)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator

from src.config import MODELS_DIR
from src.features import get_feature_columns
from fastapi import FastAPI

# ============================================================
# 1 — Initialisation FastAPI
# ============================================================

app = FastAPI(
    title="US Flights Delay API",
    description=(
        "API de prédiction des retards (classification high_delay_risk + "
        "régression avg_delay_per_flight) basée sur XGBoost + Optuna + météo."
    ),
    version="1.0.0",
)

# ============================================================
# 2 — Schéma Pydantic pour l'entrée
# ============================================================


class FlightInput(BaseModel):
    """
    Ligne agrégée du dataset US Flights Delay
    + features météo déjà calculées via 04_meteo_integration.
    """

    year: int = Field(..., ge=1980, le=2100)
    month: int = Field(..., ge=1, le=12)

    carrier: str
    carrier_name: str

    airport: str
    airport_name: str

    arr_flights: int = Field(..., ge=0)
    arr_del15: int = Field(..., ge=0)

    carrier_ct: int = Field(..., ge=0)
    weather_ct: int = Field(..., ge=0)
    nas_ct: int = Field(..., ge=0)
    security_ct: int = Field(..., ge=0)
    late_aircraft_ct: int = Field(..., ge=0)

    arr_cancelled: int = Field(..., ge=0)
    arr_diverted: int = Field(..., ge=0)

    arr_delay: float
    carrier_delay: float
    weather_delay: float
    nas_delay: float
    security_delay: float
    late_aircraft_delay: float

    # Features météo (déjà en place dans train_weather / val_weather / test_weather)
    temperature_2m_mean: float = 0.0
    precipitation_sum: float = 0.0
    windspeed_10m_max: float = 0.0

    @validator(
        "arr_delay",
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "security_delay",
        "late_aircraft_delay",
        pre=True,
    )
    def non_negative_delay(cls, v):
        if v is None:
            return 0.0
        return float(v) if v >= 0 else 0.0

    @validator(
        "arr_flights",
        "arr_del15",
        "carrier_ct",
        "weather_ct",
        "nas_ct",
        "security_ct",
        "late_aircraft_ct",
        "arr_cancelled",
        "arr_diverted",
        pre=True,
    )
    def non_negative_counts(cls, v):
        if v is None:
            return 0
        return int(v) if v >= 0 else 0


class PredictionResponse(BaseModel):
    high_delay_risk_proba: float
    high_delay_risk_label: int
    avg_delay_per_flight: float


# ============================================================
# 3 — Chargement paresseux des modèles
# ============================================================

_classifier_pipeline = None
_regressor_pipeline = None


def get_classifier_pipeline():
    global _classifier_pipeline
    if _classifier_pipeline is None:
        path = MODELS_DIR / "xgb_classifier_optuna_weather.joblib"
        if not path.is_file():
            raise FileNotFoundError(
                f"❌ Modèle classifier introuvable : {path}. "
                "Lance d'abord src.train_optuna."
            )
        _classifier_pipeline = joblib.load(path)
        print(f"[API] Classifier chargé depuis {path}")
    return _classifier_pipeline


def get_regressor_pipeline():
    global _regressor_pipeline
    if _regressor_pipeline is None:
        path = MODELS_DIR / "xgb_regressor_optuna_weather.joblib"
        if not path.is_file():
            raise FileNotFoundError(
                f"❌ Modèle regressor introuvable : {path}. "
                "Lance d'abord src.train_optuna."
            )
        _regressor_pipeline = joblib.load(path)
        print(f"[API] Regressor chargé depuis {path}")
    return _regressor_pipeline


# ============================================================
# 4 — Helper features
# ============================================================


def build_features_df(flight: FlightInput) -> pd.DataFrame:
    """
    Construit le DataFrame features attendu par le pipeline
    (features.get_feature_columns()), en recalculant delay_rate &
    avg_delay_per_flight.
    """
    data = flight.dict()

    arr_flights = data["arr_flights"]
    if arr_flights > 0:
        delay_rate = data["arr_del15"] / arr_flights
        avg_delay_per_flight = data["arr_delay"] / arr_flights
    else:
        delay_rate = 0.0
        avg_delay_per_flight = 0.0

    data["delay_rate"] = delay_rate
    data["avg_delay_per_flight"] = avg_delay_per_flight

    df = pd.DataFrame([data])

    feature_cols = get_feature_columns()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"❌ Colonnes attendues par le modèle manquantes dans la requête : {missing}"
        )

    return df[feature_cols]


# ============================================================
# 5 — Endpoints
# ============================================================


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}


@app.get("/api/health", tags=["health"])
def api_health_check():
    return {
        "status": "ok",
        "service": "us-flights-delay-api"
    }


@app.post("/api/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_delay_api(flight: FlightInput):
    return _predict_logic(flight)


def _predict_logic(flight: FlightInput) -> PredictionResponse:
    X = build_features_df(flight)

    clf = get_classifier_pipeline()
    reg = get_regressor_pipeline()

    proba = float(clf.predict_proba(X)[0, 1])
    label = int(proba >= 0.5)
    avg_delay_pred = float(reg.predict(X)[0])

    return PredictionResponse(
        high_delay_risk_proba=proba,
        high_delay_risk_label=label,
        avg_delay_per_flight=avg_delay_pred,
    )

# ============================================================
# 6 — Entrée directe uvicorn
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

