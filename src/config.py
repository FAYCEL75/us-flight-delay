"""
config.py — Configuration centrale du projet US-FLIGHTS-DELAY.

- Détecte automatiquement le ROOT du projet (dossier us-flights-delay).
- Charge config.yml.
- Expose des constantes de chemins (Path).
- Expose la configuration MLflow (URI, experiment name).
- Expose la configuration météo & training.

Ce fichier doit rester le point d'entrée unique pour toute configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# 1 — Détection du ROOT du projet
# ============================================================


def _detect_project_root() -> Path:
    """
    Détecte le dossier racine du projet (us-flights-delay).

    Logique :
    1. Si la variable d'environnement PROJECT_ROOT est définie → on l'utilise.
    2. Sinon, on remonte depuis ce fichier jusqu'à trouver un dossier contenant 'src' et 'config.yml'.

    Raises
    ------
    RuntimeError
        Si le ROOT ne peut pas être déterminé.
    """
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        root = Path(env_root).resolve()
        if not root.exists():
            raise RuntimeError(f"❌ PROJECT_ROOT pointe vers un chemin inexistant : {root}")
        return root

    # Remonter à partir de ce fichier
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "src").is_dir() and (parent / "config.yml").is_file():
            return parent

    raise RuntimeError(
        "❌ Impossible de détecter le ROOT du projet.\n"
        "Assure-toi d'exécuter le code depuis le dossier 'us-flights-delay' "
        "ou de définir la variable d'environnement PROJECT_ROOT."
    )


ROOT: Path = _detect_project_root()
SRC_DIR: Path = ROOT / "src"

print("[INFO] ROOT =", ROOT)
print("[INFO] src exists? ", SRC_DIR.is_dir())

if not SRC_DIR.is_dir():
    raise RuntimeError(
        "❌ Dossier src introuvable. Vérifie que tu exécutes bien le code dans le projet us-flights-delay."
    )

# ============================================================
# 2 — Chargement de config.yml
# ============================================================


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    """
    Charge un fichier YAML et retourne un dict Python.

    Parameters
    ----------
    path : Path
        Chemin vers config.yml.

    Raises
    ------
    FileNotFoundError
        Si le fichier config.yml n'existe pas.
    """
    if not path.is_file():
        raise FileNotFoundError(f"❌ Fichier de configuration introuvable : {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data


CONFIG_PATH: Path = ROOT / "config.yml"
CONFIG: Dict[str, Any] = _load_yaml_config(CONFIG_PATH)

print("[INFO] config.yml chargé depuis :", CONFIG_PATH)

# ============================================================
# 3 — Extraction des sections principales
# ============================================================

PROJECT_NAME: str = CONFIG.get("project", {}).get("name", "us_flights_delay")

# ----- Paths -----
_paths_cfg = CONFIG.get("paths", {})

DATA_DIR: Path = ROOT / "data"
DATA_RAW_DIR: Path = ROOT / _paths_cfg.get("data_raw", "data/raw")
DATA_PROCESSED_DIR: Path = ROOT / _paths_cfg.get("data_processed", "data/processed")

WEATHER_DIR: Path = ROOT / _paths_cfg.get("data_weather", "data/weather")
DATA_WEATHER_DIR: Path = WEATHER_DIR   # <<<<<< AJOUT ESSENTIEL

MODELS_DIR: Path = ROOT / _paths_cfg.get("models", "models")
REPORTS_DIR: Path = ROOT / _paths_cfg.get("reports", "reports")
MLRUNS_DIR: Path = ROOT / _paths_cfg.get("mlruns", "mlruns")

print("[INFO] DATA_RAW_DIR      =", DATA_RAW_DIR)
print("[INFO] DATA_PROCESSED_DIR=", DATA_PROCESSED_DIR)
print("[INFO] WEATHER_DIR       =", WEATHER_DIR)
print("[INFO] MODELS_DIR        =", MODELS_DIR)
print("[INFO] REPORTS_DIR       =", REPORTS_DIR)
print("[INFO] MLRUNS_DIR        =", MLRUNS_DIR)

# ----- MLflow -----
_mlflow_cfg = CONFIG.get("mlflow", {})

MLFLOW_EXPERIMENT_NAME: str = _mlflow_cfg.get("experiment_name", "us_flights_delay")
MLFLOW_CLASSIFIER_RUN_NAME: str = _mlflow_cfg.get("classifier_run_name", "xgb_classifier_optuna_weather")
MLFLOW_REGRESSOR_RUN_NAME: str = _mlflow_cfg.get("regressor_run_name", "xgb_regressor_optuna_weather")


def _build_local_mlflow_uri(mlruns_dir: Path) -> str:
    """
    Construit une URI file:/// absolue pour MLflow à partir d'un dossier mlruns.

    Exemple sous Windows :
    - Path('C:/Users/xxx/us-flights-delay/mlruns').as_uri()
      → 'file:///C:/Users/xxx/us-flights-delay/mlruns'
    """
    mlruns_dir = mlruns_dir.resolve()
    return mlruns_dir.as_uri()


_default_local_uri = _build_local_mlflow_uri(MLRUNS_DIR)

_env_mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_URI_CFG: str = _env_mlflow_uri or _default_local_uri

print("[INFO] MLFLOW_TRACKING_URI (cfg) =", MLFLOW_TRACKING_URI_CFG)
print("[INFO] MLflow experiment name    =", MLFLOW_EXPERIMENT_NAME)

# ----- Training -----
_training_cfg = CONFIG.get("training", {})

TARGET_CLASSIFICATION: str = _training_cfg.get("target_classification", "high_delay_risk")
TARGET_REGRESSION: str = _training_cfg.get("target_regression", "avg_delay_per_flight")

HIGH_DELAY_RATE_THRESHOLD: float = float(
    _training_cfg.get("high_delay_rate_threshold", 0.25)
)

_optuna_cfg = _training_cfg.get("optuna", {})
OPTUNA_N_TRIALS_CLASSIFIER: int = int(_optuna_cfg.get("n_trials_classifier", 10))
OPTUNA_N_TRIALS_REGRESSOR: int = int(_optuna_cfg.get("n_trials_regressor", 10))
OPTUNA_TIMEOUT_SECONDS_CLASSIFIER: int | None = _optuna_cfg.get("timeout_seconds_classifier")
OPTUNA_TIMEOUT_SECONDS_REGRESSOR: int | None = _optuna_cfg.get("timeout_seconds_regressor")

print("[INFO] HIGH_DELAY_RATE_THRESHOLD =", HIGH_DELAY_RATE_THRESHOLD)

# ----- Weather -----
_weather_cfg = CONFIG.get("weather", {})

WEATHER_PROVIDER: str = _weather_cfg.get("provider", "open-meteo")
WEATHER_CACHE_DIR: Path = ROOT / _weather_cfg.get("cache_dir", "data/weather")
WEATHER_DAILY_VARIABLES = _weather_cfg.get("daily_variables", [])

WEATHER_TIMEOUT_SECONDS: int = int(_weather_cfg.get("timeout_seconds", 30))
WEATHER_MAX_RETRIES: int = int(_weather_cfg.get("max_retries", 3))
WEATHER_RETRY_BACKOFF_SECONDS: int = int(_weather_cfg.get("retry_backoff_seconds", 5))
WEATHER_AGGREGATION_LEVEL: str = _weather_cfg.get("aggregation_level", "monthly")

WEATHER_CACHE_FILE: Path = WEATHER_CACHE_DIR / "weather_cache.json"

print("[INFO] WEATHER_CACHE_DIR         =", WEATHER_CACHE_DIR)
print("[INFO] WEATHER_CACHE_FILE        =", WEATHER_CACHE_FILE)
print("[INFO] WEATHER_DAILY_VARIABLES   =", WEATHER_DAILY_VARIABLES)

# ----- AWS (optionnel) -----
_aws_cfg = CONFIG.get("aws", {})

AWS_ENABLED: bool = bool(_aws_cfg.get("enabled", False))
AWS_S3_BUCKET: str | None = _aws_cfg.get("s3_bucket")
AWS_S3_MODELS_PREFIX: str | None = _aws_cfg.get("s3_models_prefix")
AWS_S3_MLRUNS_PREFIX: str | None = _aws_cfg.get("s3_mlruns_prefix")

# ----- API & APP -----
_api_cfg = CONFIG.get("api", {})
API_HOST: str = _api_cfg.get("host", "0.0.0.0")
API_PORT: int = int(_api_cfg.get("port", 8000))

_app_cfg = CONFIG.get("app", {})
APP_HOST: str = _app_cfg.get("host", "0.0.0.0")
APP_PORT: int = int(_app_cfg.get("port", 8501))


def print_config_summary() -> None:
    """
    Imprime un résumé compact de la configuration importante.
    Utile dans les notebooks (06_mlflow_tracking.ipynb, 04_meteo_integration.ipynb, etc.).
    """
    print("──────────────────────────")
    print("[CONFIG] Projet :", PROJECT_NAME)
    print("[CONFIG] ROOT   :", ROOT)
    print("[CONFIG] DATA_RAW_DIR       :", DATA_RAW_DIR)
    print("[CONFIG] DATA_PROCESSED_DIR :", DATA_PROCESSED_DIR)
    print("[CONFIG] WEATHER_DIR        :", WEATHER_DIR)
    print("[CONFIG] MODELS_DIR         :", MODELS_DIR)
    print("[CONFIG] REPORTS_DIR        :", REPORTS_DIR)
    print("[CONFIG] MLRUNS_DIR         :", MLRUNS_DIR)
    print("──────────────────────────")
    print("[CONFIG] MLflow tracking URI :", MLFLOW_TRACKING_URI_CFG)
    print("[CONFIG] MLflow experiment   :", MLFLOW_EXPERIMENT_NAME)
    print("[CONFIG] Target CLF          :", TARGET_CLASSIFICATION)
    print("[CONFIG] Target REG          :", TARGET_REGRESSION)
    print("[CONFIG] HIGH_DELAY_RATE_THRESHOLD :", HIGH_DELAY_RATE_THRESHOLD)
    print("──────────────────────────")
    print("[CONFIG] WEATHER_PROVIDER    :", WEATHER_PROVIDER)
    print("[CONFIG] WEATHER_CACHE_DIR   :", WEATHER_CACHE_DIR)
    print("[CONFIG] WEATHER_CACHE_FILE  :", WEATHER_CACHE_FILE)
    print("[CONFIG] WEATHER_DAILY_VARS  :", WEATHER_DAILY_VARIABLES)
    print("──────────────────────────")
    print("[CONFIG] API_HOST:PORT       :", f"{API_HOST}:{API_PORT}")
    print("[CONFIG] APP_HOST:PORT       :", f"{APP_HOST}:{APP_PORT}")
    print("──────────────────────────")


if __name__ == "__main__":
    print_config_summary()