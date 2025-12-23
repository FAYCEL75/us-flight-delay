"""
mlflow_utils.py — Utilitaires MLflow pour le projet US-FLIGHTS-DELAY.

Objectifs :
- Centraliser la configuration MLflow.
- Garantir :
    * un seul tracking URI cohérent (local ou Docker selon la config/env)
    * un seul experiment stable : MLFLOW_EXPERIMENT_NAME
- Fournir des helpers simples :
    * setup_mlflow()
    * start_run_safe(run_name=...)
    * print_mlflow_info()

Important :
- Ne pas utiliser MLflow Model Registry (volontairement désactivé).
- Ne pas activer mlflow.autolog() dans ce projet (trop bruyant pour l’arborescence).
"""

from __future__ import annotations

import contextlib
from typing import Iterator, Optional

import mlflow

from src.config import (
    MLFLOW_TRACKING_URI_CFG,
    MLFLOW_EXPERIMENT_NAME,
)


# ============================================================
# 1 — Setup de base
# ============================================================

def setup_mlflow() -> str:
    """
    Configure MLflow (tracking URI + experiment).

    - Utilise MLFLOW_TRACKING_URI_CFG depuis config.py (qui tient compte
      de la variable d'environnement MLFLOW_TRACKING_URI si définie).
    - Crée l'experiment s'il n'existe pas encore.
    - Retourne l'ID de l'experiment.

    Returns
    -------
    experiment_id : str
    """
    # 1) Tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI_CFG)

    # 2) Experiment (get or create)
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    if experiment is None:
        experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
        print(f"[MLflow] Nouvel experiment créé : {MLFLOW_EXPERIMENT_NAME} (ID={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"[MLflow] Utilisation de l'experiment existant : {MLFLOW_EXPERIMENT_NAME} (ID={experiment_id})")

    print("──────────────────────────")
    print("[MLflow] Tracking URI :", mlflow.get_tracking_uri())
    print("[MLflow] Experiment   :", MLFLOW_EXPERIMENT_NAME, f"(ID={experiment_id})")
    print("──────────────────────────")

    return experiment_id


# ============================================================
# 2 — Helper : impression d'infos
# ============================================================

def print_mlflow_info() -> None:
    """
    Affiche un résumé compact de la config MLflow actuelle.
    Utile dans les notebooks ou pour debug.
    """
    current_uri = mlflow.get_tracking_uri()
    try:
        current_exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        exp_id = current_exp.experiment_id if current_exp is not None else "N/A"
    except Exception:
        exp_id = "N/A"

    print("──────────────────────────")
    print("[MLflow] Tracking URI :", current_uri)
    print("[MLflow] Experiment   :", MLFLOW_EXPERIMENT_NAME, f"(ID={exp_id})")
    print("──────────────────────────")


# ============================================================
# 3 — Context manager start_run_safe
# ============================================================

@contextlib.contextmanager
def start_run_safe(run_name: Optional[str] = None) -> Iterator[mlflow.ActiveRun]:
    """
    Context manager sécurisé pour lancer un run MLflow.

    Usage :
    -------
    from src.mlflow_utils import setup_mlflow, start_run_safe

    setup_mlflow()
    with start_run_safe(run_name="my_run"):
        mlflow.log_param("foo", 123)
        mlflow.log_metric("bar", 0.99)

    Comportement :
    --------------
    - S'assure qu'un experiment existe (via setup_mlflow() que tu dois appeler avant).
    - Ferme toujours le run, même en cas d'exception.
    """
    # Si aucun experiment n'est encore défini, on appelle setup_mlflow() par sécurité
    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        setup_mlflow()
        exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    # Démarrage du run
    active_run = mlflow.start_run(run_name=run_name)
    try:
        yield active_run
    finally:
        mlflow.end_run()


# ============================================================
# 4 — Exécution directe (debug)
# ============================================================

if __name__ == "__main__":
    print("[INFO] Test mlflow_utils.setup_mlflow()")
    exp_id = setup_mlflow()
    print_mlflow_info()

    print("[INFO] Test start_run_safe()")
    with start_run_safe(run_name="test_run_mlflow_utils"):
        mlflow.log_param("test_param", 1)
        mlflow.log_metric("test_metric", 0.123)

    print("[INFO] mlflow_utils.py OK ✅")