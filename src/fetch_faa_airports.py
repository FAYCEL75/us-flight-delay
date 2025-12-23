"""
fetch_faa_airports.py — Version PREMIUM
--------------------------------------

Objectif :
- Télécharger automatiquement la base OurAirports (worldwide)
- Filtrer uniquement les aéroports US
- Normaliser les codes FAA/IATA pour correspondre au dataset US Flights Delay
- Produire un cache propre dans data/processed/airports_with_coords.csv
- Fournir des fonctions prêtes pour toute la pipeline météo

Aucune colonne inventée.
Compatible avec data_preprocessing + weather API + train_optuna.
"""

import io
import pandas as pd
import requests
from pathlib import Path
from typing import Optional, Tuple


from src.config import ROOT, DATA_PROCESSED_DIR


# ============================================================
# 1 — Source officielle OurAirports
# ============================================================

OURAIRPORTS_URL = "https://ourairports.com/data/airports.csv"

CACHE_PATH = DATA_PROCESSED_DIR / "airports_with_coords.csv"


# ============================================================
# 2 — Téléchargement de la base OurAirports
# ============================================================

def download_airports_csv() -> pd.DataFrame:
    """
    Télécharge le fichier airports.csv depuis OurAirports.

    Returns
    -------
    pd.DataFrame
        Données brutes OurAirports.
    """
    print("[INFO] Téléchargement de OurAirports (airports.csv)…")

    response = requests.get(OURAIRPORTS_URL, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"❌ Impossible de télécharger airports.csv (code HTTP {response.status_code})"
        )

    df = pd.read_csv(io.StringIO(response.text))
    print(f"[INFO] airports.csv téléchargé — {df.shape[0]} lignes")

    return df


# ============================================================
# 3 — Nettoyage & normalisation des données OurAirports
# ============================================================

def preprocess_airports(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre et normalise les aéroports OurAirports pour correspondre
    au dataset FAA.

    Conserve uniquement :
    - Les aéroports situés aux USA (iso_country == 'US')
    - Les colonnes utiles : ident, name, municipality, iso_region, iata_code, local_code, latitude_deg, longitude_deg

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # Ne garder que les aéroports américains
    df = df[df["iso_country"] == "US"]

    # Colonnes utiles
    keep_cols = [
        "ident",
        "name",
        "municipality",
        "iso_region",
        "iata_code",
        "local_code",
        "latitude_deg",
        "longitude_deg",
    ]

    df = df[keep_cols]

    # Normalisation du code aéroport utilisé par FAA
    # FAA utilise généralement le code IATA, sinon le code local.
    df["faa_code"] = df["iata_code"].fillna(df["local_code"])

    # Nettoyage final
    df = df.dropna(subset=["faa_code", "latitude_deg", "longitude_deg"])
    df = df.drop_duplicates(subset=["faa_code"])

    print(f"[INFO] Aéroports US filtrés → {df.shape[0]} lignes")

    return df


# ============================================================
# 4 — Génération et sauvegarde du cache
# ============================================================

def ensure_airport_cache(force: bool = False) -> pd.DataFrame:
    """
    Génère ou recharge le cache des aéroports US avec coordonnées.

    Parameters
    ----------
    force : bool
        Si True → force le re-téléchargement depuis OurAirports.

    Returns
    -------
    pd.DataFrame
        DataFrame avec colonnes :
        faa_code, name, latitude_deg, longitude_deg, municipality, iso_region
    """
    if CACHE_PATH.is_file() and not force:
        print(f"[INFO] Chargement du cache existant : {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH)

    print("[INFO] Cache non trouvé ou regeneration forcée. Traitement en cours…")

    df_raw = download_airports_csv()
    df_clean = preprocess_airports(df_raw)

    df_clean.to_csv(CACHE_PATH, index=False)
    print(f"[INFO] Cache généré : {CACHE_PATH}")

    return df_clean


# ============================================================
# 5 — Accès direct à un aéroport
# ============================================================

def get_airport_coords(airport_code: str) -> Optional[Tuple[float, float]]:
    """
    Renvoie (lat, lon) pour un code aéroport FAA.

    Parameters
    ----------
    airport_code : str

    Returns
    -------
    tuple(lat, lon) ou None si non trouvé
    """
    df = ensure_airport_cache()

    row = df[df["faa_code"] == airport_code]

    if row.empty:
        print(f"[WARN] Aéroport inconnu dans OurAirports : {airport_code}")
        return None

    lat = row.iloc[0]["latitude_deg"]
    lon = row.iloc[0]["longitude_deg"]

    return float(lat), float(lon)


# ============================================================
# 6 — Fonction principale pour la pipeline météo
# ============================================================

def load_airports_with_coords() -> pd.DataFrame:
    """
    Charge le cache (et le génère si besoin).

    Returns
    -------
    pd.DataFrame
        DataFrame avec au minimum :
        faa_code, latitude_deg, longitude_deg
    """
    return ensure_airport_cache()


# ============================================================
# 7 — Test manuel
# ============================================================

if __name__ == "__main__":
    print("[INFO] Test — Génération du cache OurAirports…")
    df_air = ensure_airport_cache()
    print(df_air.head())

    print("\n[INFO] Test — Lookup d'un aéroport : LAX")
    print(get_airport_coords("LAX"))