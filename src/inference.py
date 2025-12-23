"""
Inference utilities for US Flights Delay.

- Charge deux pipelines sklearn complets (preprocessor + XGBoost) 
  depuis :
    - models/xgb_classifier_optuna_weather.joblib
    - models/xgb_regressor_optuna_weather.joblib
- Expose des helpers simples pour l'API FastAPI.
"""

from typing import Dict, Any, List

import joblib
import pandas as pd

from src.config import MODELS_DIR


# ============================================================
# Chargement des modèles (pipelines complets)
# ============================================================

def load_models() -> Dict[str, Any]:
    """
    Charge les deux pipelines entraînés.

    Retourne un dict:
        {
            "classifier": Pipeline,
            "regressor": Pipeline,
        }
    """
    classifier_path = MODELS_DIR / "xgb_classifier_optuna_weather.joblib"
    regressor_path = MODELS_DIR / "xgb_regressor_optuna_weather.joblib"

    if not classifier_path.exists():
        raise FileNotFoundError(f"[API] Classifier introuvable : {classifier_path}")
    if not regressor_path.exists():
        raise FileNotFoundError(f"[API] Regressor introuvable : {regressor_path}")

    clf = joblib.load(classifier_path)
    reg = joblib.load(regressor_path)

    print("[API] Pipelines chargés :")
    print(f"   - {classifier_path.name}")
    print(f"   - {regressor_path.name}")

    return {
        "classifier": clf,
        "regressor": reg,
    }


# ============================================================
# Utils internes
# ============================================================

def _to_dataframe(record: Dict[str, Any]) -> pd.DataFrame:
    """
    Convertit un dict (payload) en DataFrame 1 ligne.

    Hypothèse : les clés du dict correspondent aux colonnes
    utilisées à l'entraînement (train_weather.csv), à l’exception
    des colonnes cibles.
    """
    return pd.DataFrame([record])


# ============================================================
# Prédictions
# ============================================================

def predict_single(payload: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prédiction unitaire.

    payload : dict à plat (caractéristiques du mois aéroportaire)
    models  : dict retourné par load_models()
    """
    df = _to_dataframe(payload)

    clf = models["classifier"]
    reg = models["regressor"]

    proba = float(clf.predict_proba(df)[:, 1][0])
    delay = float(reg.predict(df)[0])

    risk = "HIGH" if proba >= 0.5 else "LOW"

    return {
        "high_delay_probability": proba,
        "avg_delay_minutes": delay,
        "risk_level": risk,
    }


def predict_batch(records: List[Dict[str, Any]], models: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prédictions batch.

    records : liste de payloads dict
    """
    clf = models["classifier"]
    reg = models["regressor"]

    results: List[Dict[str, Any]] = []

    for rec in records:
        df = _to_dataframe(rec)
        p = float(clf.predict_proba(df)[:, 1][0])
        d = float(reg.predict(df)[0])
        risk = "HIGH" if p >= 0.5 else "LOW"

        out = dict(rec)
        out.update(
            {
                "high_delay_probability": p,
                "avg_delay_minutes": d,
                "risk_level": risk,
            }
        )
        results.append(out)

    return {"results": results}