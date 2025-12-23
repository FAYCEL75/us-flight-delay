"""
data_preprocessing.py — Préprocessing PREMIUM pour le projet US-FLIGHTS-DELAY.

Ce module :
- Charge le dataset brut officiel (colonnes imposées).
- Vérifie l’intégrité des colonnes.
- Nettoie les valeurs incohérentes.
- Calcule les 3 features obligatoires :
    • delay_rate = arr_del15 / arr_flights
    • high_delay_risk = (delay_rate > seuil)
    • avg_delay_per_flight = arr_delay / arr_flights
- Produit un dataset propre, prêt pour le split + ajout météo.

Aucune approximation.
Aucune colonne inventée.
Compatible notebooks + train_optuna + API.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    HIGH_DELAY_RATE_THRESHOLD,
)


# ============================================================
# 1 — Colonnes imposées par le dataset officiel
# ============================================================

REQUIRED_COLUMNS: List[str] = [
    "year",
    "month",
    "carrier",
    "carrier_name",
    "airport",
    "airport_name",
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


# ============================================================
# 2 — Chargement du fichier brut
# ============================================================

def load_raw_data(file_name: str = "us_flights_delay.csv") -> pd.DataFrame:
    """
    Charge le dataset brut depuis data/raw/.

    Parameters
    ----------
    file_name : str
        Nom du fichier CSV dans data/raw/.

    Returns
    -------
    pd.DataFrame
        Dataset brut.
    """
    path = DATA_RAW_DIR / file_name
    if not path.is_file():
        raise FileNotFoundError(f"❌ Fichier brut introuvable : {path}")

    df = pd.read_csv(path)
    print(f"[INFO] Dataset brut chargé : {path} — {df.shape[0]} lignes")

    # Vérification stricte des colonnes
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Colonnes manquantes dans le dataset brut : {missing}")

    # On retourne le dataframe brut, sans modification
    return df[REQUIRED_COLUMNS].copy()


# ============================================================
# 3 — Nettoyage minimal & préparation des types
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage minimal :
    - Convertit year et month en int
    - Force arr_flights >= 0
    - Remplace divisions impossibles par 0 (arr_flights = 0)
    - Version non-destructive (retourne une copie)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset brut.

    Returns
    -------
    pd.DataFrame
        Dataset nettoyé.
    """
    df = df.copy()

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    # On sécurise arr_flights
    df["arr_flights"] = df["arr_flights"].fillna(0).clip(lower=0)

    # Remplacer NaN dans les autres colonnes numériques par 0
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


# ============================================================
# 4 — Calcul des features obligatoires
# ============================================================

def compute_mandatory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les 3 features obligatoires :
    - delay_rate = arr_del15 / arr_flights
    - high_delay_risk = delay_rate > seuil
    - avg_delay_per_flight = arr_delay / arr_flights

    Notes :
    - arr_flights == 0 → delay_rate = 0 et avg_delay_per_flight = 0
    - high_delay_risk utilise le seuil défini dans config.yml
    """
    df = df.copy()

    # Division sécurisée
    df["delay_rate"] = np.where(
        df["arr_flights"] > 0,
        df["arr_del15"] / df["arr_flights"],
        0,
    )

    df["avg_delay_per_flight"] = np.where(
        df["arr_flights"] > 0,
        df["arr_delay"] / df["arr_flights"],
        0,
    )

    # high_delay_risk basé sur le seuil défini dans config.yml
    df["high_delay_risk"] = (df["delay_rate"] > HIGH_DELAY_RATE_THRESHOLD).astype(int)

    print(f"[INFO] Features obligatoires ajoutées (seuil = {HIGH_DELAY_RATE_THRESHOLD})")

    return df


# ============================================================
# 5 — Fonction principale
# ============================================================

def prepare_base_dataset(file_name: str = "us_flights_delay.csv") -> pd.DataFrame:
    """
    Pipeline complet :
    1. Chargement brut
    2. Nettoyage minimal
    3. Calcul des features obligatoires
    4. Sauvegarde en data/processed/base_preprocessed.csv

    Returns
    -------
    pd.DataFrame
        Dataset propre avec features obligatoires.
    """
    df = load_raw_data(file_name)
    df = clean_data(df)
    df = compute_mandatory_features(df)

    output_path = DATA_PROCESSED_DIR / "base_preprocessed.csv"
    df.to_csv(output_path, index=False)

    print(f"[INFO] Dataset préprocessé sauvegardé : {output_path} — {df.shape[0]} lignes")

    return df


# ============================================================
# 6 — exécution directe
# ============================================================

if __name__ == "__main__":
    print("[INFO] Lancement du préprocessing complet…")
    prepare_base_dataset()
    print("[INFO] Terminé.")