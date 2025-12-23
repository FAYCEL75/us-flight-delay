"""
train_optuna.py — VERSION ULTRA PREMIUM

Objectifs :
- Entraîner deux modèles XGBoost avec Optuna :
    1) Classification : high_delay_risk
    2) Régression : avg_delay_per_flight
- Utiliser les données enrichies météo :
    - data/processed/train_weather.csv
    - data/processed/val_weather.csv
    - data/processed/test_weather.csv
- Utiliser le préprocesseur central de features.py (StandardScaler + OneHotEncoder)
- Logger proprement dans MLflow :
    - 1 run pour le classifieur
    - 1 run pour le régressseur
- Sauvegarder les modèles :
    - MLflow : artifacts/model
    - joblib : models/xgb_classifier_optuna_weather.joblib
              models/xgb_regressor_optuna_weather.joblib
- Uploader les modèles vers S3 si aws.enabled = true

Aucune colonne inventée :
- On utilise uniquement les colonnes définies dans features.get_feature_columns()
  + cibles définies dans config.py.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

import joblib
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from xgboost import XGBClassifier, XGBRegressor

import optuna

import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError

from src.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    MLFLOW_TRACKING_URI_CFG,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_CLASSIFIER_RUN_NAME,
    MLFLOW_REGRESSOR_RUN_NAME,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    OPTUNA_N_TRIALS_CLASSIFIER,
    OPTUNA_N_TRIALS_REGRESSOR,
    OPTUNA_TIMEOUT_SECONDS_CLASSIFIER,
    OPTUNA_TIMEOUT_SECONDS_REGRESSOR,
    AWS_ENABLED,
    AWS_S3_BUCKET,
    AWS_S3_MODELS_PREFIX,
)
from src.features import build_preprocessor, get_feature_columns


# ============================================================
# 1 — Préparation des chemins et de MLflow
# ============================================================

MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("[INFO] MODELS_DIR =", MODELS_DIR)

# Configuration MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI_CFG)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

print("[INFO] MLflow tracking URI :", mlflow.get_tracking_uri())
print("[INFO] MLflow experiment   :", MLFLOW_EXPERIMENT_NAME)


# ============================================================
# 2 — Chargement des datasets enrichis météo
# ============================================================

def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge les datasets train/val/test enrichis avec les features météo.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    train_path = DATA_PROCESSED_DIR / "train_weather.csv"
    val_path = DATA_PROCESSED_DIR / "val_weather.csv"
    test_path = DATA_PROCESSED_DIR / "test_weather.csv"

    for p, name in [(train_path, "train_weather.csv"),
                    (val_path, "val_weather.csv"),
                    (test_path, "test_weather.csv")]:
        if not p.is_file():
            raise FileNotFoundError(
                f"❌ Fichier {name} introuvable dans {DATA_PROCESSED_DIR}. "
                "Assure-toi que 04_meteo_integration a été exécuté."
            )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print("[INFO] train_weather :", train_df.shape)
    print("[INFO] val_weather   :", val_df.shape)
    print("[INFO] test_weather  :", test_df.shape)

    return train_df, val_df, test_df


# ============================================================
# 3 — Préparation X / y
# ============================================================

def prepare_xy(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Sépare features et cibles en s'appuyant sur :
    - get_feature_columns() pour X
    - TARGET_CLASSIFICATION / TARGET_REGRESSION pour y

    Returns
    -------
    X, y_clf, y_reg
    """
    feature_cols = get_feature_columns()

    print("[DEBUG] Colonnes attendues par get_feature_columns() :", feature_cols)
    print("[DEBUG] Colonnes présentes dans df :", list(df.columns))

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"❌ Colonnes de features manquantes dans le dataset : {missing_features}"
        )

    if TARGET_CLASSIFICATION not in df.columns:
        raise ValueError(f"❌ Colonne cible classification manquante : {TARGET_CLASSIFICATION}")

    if TARGET_REGRESSION not in df.columns:
        raise ValueError(f"❌ Colonne cible regression manquante : {TARGET_REGRESSION}")

    X = df[feature_cols].copy()
    y_clf = df[TARGET_CLASSIFICATION].astype(int)
    y_reg = df[TARGET_REGRESSION].astype(float)

    return X, y_clf, y_reg


# ============================================================
# 3b — Utilitaire RMSE compatible toutes versions sklearn
# ============================================================

def compute_rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Calcule la RMSE de manière compatible avec toutes les versions de sklearn
    (sans utiliser l’argument squared).
    """
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


# ============================================================
# 4 — Upload S3 des modèles (optionnel)
# ============================================================

def upload_model_to_s3(local_path: Path, model_name: str) -> None:
    """
    Upload le modèle (fichier .joblib) vers S3, si AWS_ENABLED = True.

    - S3 bucket : AWS_S3_BUCKET
    - Prefix    : AWS_S3_MODELS_PREFIX

    Le chemin S3 final sera : s3://bucket/prefix/model_name
    """
    if not AWS_ENABLED:
        print("[INFO] AWS_ENABLED = False → pas d'upload S3.")
        return

    if not AWS_S3_BUCKET:
        print("[WARN] AWS_ENABLED = True mais AWS_S3_BUCKET est vide → pas d'upload S3.")
        return

    if not AWS_S3_MODELS_PREFIX:
        prefix = ""
    else:
        prefix = AWS_S3_MODELS_PREFIX.rstrip("/")

    s3_key = f"{prefix}/{model_name}" if prefix else model_name

    try:
        s3 = boto3.client("s3")
        s3.upload_file(str(local_path), AWS_S3_BUCKET, s3_key)
        print(f"[INFO] Modèle uploadé sur S3 → s3://{AWS_S3_BUCKET}/{s3_key}")
    except (BotoCoreError, NoCredentialsError, ClientError) as e:
        print(f"[WARN] Échec upload S3 pour {local_path.name} : {e}")


# ============================================================
# 5 — Optuna : espaces de recherche
# ============================================================

def suggest_xgb_params_classifier(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Espace de recherche pour XGBClassifier.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }
    return params


def suggest_xgb_params_regressor(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Espace de recherche pour XGBRegressor.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }
    return params


# ============================================================
# 6 — Optuna : tuning classification
# ============================================================

def tune_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Any]:
    """
    Optimisation Optuna pour la classification (AUC sur le set de validation).
    """

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params_classifier(trial)

        preprocessor = build_preprocessor()
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
            random_state=42,
            **params,
        )

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipe.fit(X_train, y_train)

        y_proba = pipe.predict_proba(X_val)[:, 1]

        try:
            auc = roc_auc_score(y_val, y_proba)
        except ValueError:
            # si une seule classe dans y_val
            auc = 0.5

        # Optuna maximise → on retourne AUC
        return auc

    study = optuna.create_study(
        direction="maximize",
        study_name="xgb_classifier_optuna_weather",
    )

    print(
        f"[OPTUNA-CLF] n_trials={OPTUNA_N_TRIALS_CLASSIFIER}, "
        f"timeout={OPTUNA_TIMEOUT_SECONDS_CLASSIFIER}"
    )

    study.optimize(
        objective,
        n_trials=OPTUNA_N_TRIALS_CLASSIFIER,
        timeout=OPTUNA_TIMEOUT_SECONDS_CLASSIFIER,
        show_progress_bar=False,
    )

    print("[OPTUNA-CLF] Best AUC :", study.best_value)
    print("[OPTUNA-CLF] Best params :", study.best_params)

    return study.best_params


# ============================================================
# 7 — Optuna : tuning régression
# ============================================================

def tune_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Any]:
    """
    Optimisation Optuna pour la régression (RMSE sur le set de validation).
    """

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params_regressor(trial)

        preprocessor = build_preprocessor()
        model = XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
            **params,
        )

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_val)

        # ⚠️ IMPORTANT : pas de squared=False pour compat sklearn
        rmse = compute_rmse(y_val, y_pred)

        # Optuna MINIMISE → donc on retourne RMSE
        return rmse

    study = optuna.create_study(
        direction="minimize",
        study_name="xgb_regressor_optuna_weather",
    )

    print(
        f"[OPTUNA-REG] n_trials={OPTUNA_N_TRIALS_REGRESSOR}, "
        f"timeout={OPTUNA_TIMEOUT_SECONDS_REGRESSOR}"
    )

    study.optimize(
        objective,
        n_trials=OPTUNA_N_TRIALS_REGRESSOR,
        timeout=OPTUNA_TIMEOUT_SECONDS_REGRESSOR,
        show_progress_bar=False,
    )

    print("[OPTUNA-REG] Best RMSE :", study.best_value)
    print("[OPTUNA-REG] Best params :", study.best_params)

    return study.best_params


# ============================================================
# 8 — Entraînement final + MLflow + sauvegarde (Classifier)
# ============================================================

def train_final_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_params: Dict[str, Any],
) -> None:
    """
    Entraîne le classifieur final sur train+val, évalue sur test, log dans MLflow,
    sauvegarde le modèle en joblib + upload S3.
    """
    print("[CLF] Entraînement final avec les meilleurs hyperparamètres Optuna…")

    preprocessor = build_preprocessor()
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42,
        **best_params,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    pipe.fit(X_train_full, y_train_full)

    # Évaluation sur test
    y_proba_test = pipe.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= 0.5).astype(int)

    try:
        auc_test = roc_auc_score(y_test, y_proba_test)
    except ValueError:
        auc_test = 0.5

    acc_test = accuracy_score(y_test, y_pred_test)
    rec_test = recall_score(y_test, y_pred_test, pos_label=1)

    print("[CLF] Test AUC     :", auc_test)
    print("[CLF] Test Accuracy:", acc_test)
    print("[CLF] Test Recall  :", rec_test)

    model_path = MODELS_DIR / "xgb_classifier_optuna_weather.joblib"

    with mlflow.start_run(run_name=MLFLOW_CLASSIFIER_RUN_NAME):
        # Log hyperparams
        mlflow.log_params(best_params)

        # Log metrics
        mlflow.log_metric("test_auc", auc_test)
        mlflow.log_metric("test_accuracy", acc_test)
        mlflow.log_metric("test_recall", rec_test)

        # Log modèle dans MLflow
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=None,
        )

        # Sauvegarde joblib
        joblib.dump(pipe, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="joblib")

        print(f"[CLF] Modèle sauvegardé localement → {model_path}")

    # Upload S3
    upload_model_to_s3(model_path, model_name=model_path.name)


# ============================================================
# 9 — Entraînement final + MLflow + sauvegarde (Regressor)
# ============================================================

def train_final_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_params: Dict[str, Any],
) -> None:
    """
    Entraîne le régressseur final sur train+val, évalue sur test, log dans MLflow,
    sauvegarde le modèle en joblib + upload S3.
    """
    print("[REG] Entraînement final avec les meilleurs hyperparamètres Optuna…")

    preprocessor = build_preprocessor()
    model = XGBRegressor(
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
        **best_params,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    pipe.fit(X_train_full, y_train_full)

    # Évaluation sur test
    y_pred_test = pipe.predict(X_test)

    rmse_test = compute_rmse(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    print("[REG] Test RMSE :", rmse_test)
    print("[REG] Test MAE  :", mae_test)
    print("[REG] Test R2   :", r2_test)

    model_path = MODELS_DIR / "xgb_regressor_optuna_weather.joblib"

    with mlflow.start_run(run_name=MLFLOW_REGRESSOR_RUN_NAME):
        # Log hyperparams
        mlflow.log_params(best_params)

        # Log metrics
        mlflow.log_metric("test_rmse", rmse_test)
        mlflow.log_metric("test_mae", mae_test)
        mlflow.log_metric("test_r2", r2_test)

        # Log modèle dans MLflow
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=None,
        )

        # Sauvegarde joblib
        joblib.dump(pipe, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="joblib")

        print(f"[REG] Modèle sauvegardé localement → {model_path}")

    # Upload S3
    upload_model_to_s3(model_path, model_name=model_path.name)


# ============================================================
# 10 — main()
# ============================================================

def main() -> None:
    print("──────────────────────────")
    print("[MAIN] Chargement des datasets météo…")
    train_df, val_df, test_df = load_datasets()

    print("[MAIN] Préparation X / y…")
    X_train, y_clf_train, y_reg_train = prepare_xy(train_df)
    X_val, y_clf_val, y_reg_val = prepare_xy(val_df)
    X_test, y_clf_test, y_reg_test = prepare_xy(test_df)

    print("──────────────────────────")
    print("[MAIN] Optuna — classification (high_delay_risk)…")
    best_params_clf = tune_classifier(X_train, y_clf_train, X_val, y_clf_val)

    print("──────────────────────────")
    print("[MAIN] Optuna — régression (avg_delay_per_flight)…")
    best_params_reg = tune_regressor(X_train, y_reg_train, X_val, y_reg_val)

    print("──────────────────────────")
    print("[MAIN] Entraînement final du classifieur…")
    train_final_classifier(
        X_train, y_clf_train, X_val, y_clf_val, X_test, y_clf_test, best_params_clf
    )

    print("──────────────────────────")
    print("[MAIN] Entraînement final du régressseur…")
    train_final_regressor(
        X_train, y_reg_train, X_val, y_reg_val, X_test, y_reg_test, best_params_reg
    )

    print("──────────────────────────")
    print("[MAIN] Pipeline Optuna + XGBoost terminée ✅")


if __name__ == "__main__":
    main()