"""
features.py — Construction des features et du pipeline de preprocessing
pour le projet US-FLIGHTS-DELAY.

Ce module :
- Définit les colonnes numériques & catégorielles autorisées
- Crée le ColumnTransformer :
      * StandardScaler pour les colonnes numériques
      * OneHotEncoder pour les colonnes catégorielles
- Fournit build_preprocessor() utilisé par :
      * train_optuna.py
      * inference.py
      * API FastAPI
      * Streamlit

Aucune colonne inventée.
Compatible dataset officiel + météo.
"""

from __future__ import annotations

import pandas as pd
from typing import List

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ============================================================
# 1 — Colonnes numériques brutes du dataset officiel
# ============================================================

NUMERIC_COLUMNS: List[str] = [
    "arr_flights",
    "arr_del15",
    "carrier_ct",
    "weather_ct",
    "nas_ct",
    "security_ct",
    "late_aircraft_ct",
    "arr_cancelled",
    "arr_diverted",
    "arr_delay",
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
]

# Colonnes météo ajoutées par weather_api + 04_meteo_integration
WEATHER_NUMERIC_COLUMNS: List[str] = [
    "temperature_2m_mean",
    "precipitation_sum",
    "windspeed_10m_max",
]


# ============================================================
# 2 — Colonnes catégorielles + temporelles
# ============================================================

CATEGORICAL_COLUMNS: List[str] = [
    "carrier",
    "carrier_name",
    "airport",
    "airport_name",
    "year",
    "month",
]


# ============================================================
# 3 — Features dérivées
# ============================================================

MANDATORY_FEATURES: List[str] = [
    "delay_rate",
    "avg_delay_per_flight",
    "high_delay_risk",
]

NON_FEATURE_COLUMNS = ["high_delay_risk"]


# ============================================================
# 4 — Liste EXACTE des features pour XGBoost
# ============================================================

def get_feature_columns() -> List[str]:
    """
    Renvoie la liste complète des colonnes features utilisées par XGBoost.

    - Colonnes brutes officielles
    - Colonnes dérivées essentielles
    - Colonnes météo
    - Colonnes catégorielles
    """

    return (
        NUMERIC_COLUMNS
        + WEATHER_NUMERIC_COLUMNS
        + ["delay_rate", "avg_delay_per_flight"]  # high_delay_risk exclue
        + CATEGORICAL_COLUMNS
    )


# ============================================================
# 5 — Preprocessor PREMIUM
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    StandardScaler sur numériques,
    OneHotEncoder sur catégorielles.
    """
    print("[INFO] Construction du préprocesseur StandardScaler + OneHotEncoder…")

    numeric_features = (
        NUMERIC_COLUMNS
        + WEATHER_NUMERIC_COLUMNS
        + ["delay_rate", "avg_delay_per_flight"]
    )

    categorical_features = CATEGORICAL_COLUMNS

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


# ============================================================
# 6 — Helper d'affichage
# ============================================================

def show_feature_info():
    print("────────────────────────────────────────")
    print("📌 Colonnes numériques utilisées :")
    for col in NUMERIC_COLUMNS + WEATHER_NUMERIC_COLUMNS + ["delay_rate", "avg_delay_per_flight"]:
        print("   -", col)

    print("\n📌 Colonnes catégorielles utilisées :")
    for col in CATEGORICAL_COLUMNS:
        print("   -", col)

    print("\n📌 Colonne cible (classification) : high_delay_risk")
    print("────────────────────────────────────────")


if __name__ == "__main__":
    show_feature_info()
    preprocessor = build_preprocessor()
    print("\n[INFO] Preprocessor construit avec succès.")