Link to github [project](https://github.com/FAYCEL75/us-flight-delay)

# US Flights Delay Prediction

## Objectif du projet

Ce projet vise à prédire les **retards aériens aux États-Unis** à partir de données historiques agrégées, enrichies par des données météo publiques.

Il propose :

* une **classification** du risque de retard élevé,
* une **régression** du retard moyen par vol,
* une **API FastAPI**,
* une **application Streamlit interactive**,
* un **suivi d’expériences avec MLflow**,
* une **explicabilité complète avec SHAP**.

---

## Données

### Source

* Dataset officiel agrégé par **aéroport / compagnie / mois**
* Données météo issues de **Open-Meteo Archive API**

### Variables principales

* Trafic aérien (arr_flights, arr_del15, etc.)
* Causes de retard (weather, carrier, NAS, security…)
* Météo agrégée mensuelle
* Variables temporelles

---

## EDA (Exploration des données)

Les analyses exploratoires ont mis en évidence :

* une distribution très asymétrique des retards,
* un rôle amplificateur de la météo,
* une forte dépendance saisonnière et géographique,
* des interactions complexes entre congestion et météo.

Ces observations ont guidé :

* le choix des features,
* la séparation classification / régression,
* l’utilisation de modèles non linéaires.

---

## Modélisation

### Tâches

* **Classification** : `high_delay_risk`
* **Régression** : `avg_delay_per_flight`

### Modèles

* XGBoost Classifier
* XGBoost Regressor

### Optimisation

* Optuna (AUC pour la classification, RMSE pour la régression)

---

## Explicabilité

Les modèles sont expliqués via **SHAP** :

* summary plots
* bar plots (top features)
* force plots pour des prédictions individuelles

Les sorties sont intégrées dans l’application.

---

## Suivi des expériences

* **MLflow** avec backend SQLite
* Tracking des runs
* Comparaison des modèles
* Chargement reproductible des modèles finaux

---

## API FastAPI

### Endpoints

* `/health` : statut de l’API
* `/predict` : prédiction classification + régression

L’API charge les modèles de manière paresseuse et valide les entrées via Pydantic.

---

## Application Streamlit

Fonctionnalités :

* formulaire de prédiction complet,
* carte interactive des aéroports,
* affichage du risque sous forme visuelle,
* fallback automatique sur modèles locaux,
* historique des prédictions exportable.

---

## Architecture

          ┌────────────────────┐
          │  Données Trafic US │
          └─────────┬──────────┘
                    │
          ┌─────────▼──────────┐
          │  Feature Engineering│
          └─────────┬──────────┘
                    │
        ┌───────────▼───────────┐
        │  Modèles XGBoost       │
        │  (Optuna optimisés)    │
        └───────┬────────┬──────┘
                │        │
        ┌───────▼───┐ ┌──▼────────┐
        │ MLflow     │ │ SHAP       │
        └───────┬───┘ └────────────┘
                │
        ┌───────▼────────┐
        │ FastAPI /predict│
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │ Streamlit App   │
        └────────────────┘

---

## Limites

* Données agrégées mensuelles
* Pas de prédiction temps réel
* Pas de monitoring en production

---

## Améliorations futures

* Données horaires ou journalières
* Monitoring des dérives
* Déploiement cloud complet
* A/B testing des modèles

---

## Auteur

Projet réalisé dans le cadre d’un parcours Data Science & MLOps
par **Faycel**.
