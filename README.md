# GitHub Star Recommender 🌟

Ce projet est un système de recommandation de dépôts GitHub basé sur le filtrage collaboratif. Il utilise l'API GitHub pour collecter des données, MongoDB pour le stockage, et **plusieurs algorithmes de recommandation** (dont SVD via scikit-learn) pour générer des prédictions.

L'objectif est de suggérer des dépôts pertinents à un utilisateur en analysant les similitudes avec les historiques de "stars" d'utilisateurs experts ("Power Users").
Le projet intègre également **MLflow** pour le suivi des expériences et la gestion des modèles.

## 📋 Table des matières

- [Architecture du Projet](#architecture-du-projet)
- [Logique de Collecte & Données](#logique-de-collecte--données)
- [Modélisation & MLflow](#modélisation--mlflow)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [API Reference](#api-reference)
- [Recherche et Expérimentation](#recherche-et-expérimentation)

## <a id="architecture-du-projet"></a>📂 Architecture du Projet

Le projet suit une structure modulaire, séparant l'API, le code d'entraînement, et l'application web.

```text
├── .github/workflows/                 # CI/CD pipelines
├── config/                            # Configuration globale
├── research/                          # Notebooks pour l'analyse et l'expérimentation
├── src/
│   ├── api/                           # Code de l'API (FastAPI)
│   ├── GitHubAPIRecommendationOfRepos/ # Package principal
│   │   ├── components/                # Modules logiques (Ingestion, Transformation)
│   │   ├── train/                     # Pipelines d'entraînement et définition des modèles
│   │   ├── model/                     # Modèles sauvegardés
│   │   ├── entity/                    # Entités de données
│   │   ├── utils/                     # Utilitaires
│   │   └── ...
│   └── mlflow_store_reco/             # Base de données et artifacts MLflow
├── templates/                         # Fichiers HTML pour l'interface Web
├── static/                            # Fichiers statiques (CSS, JS)
├── app.py                             # Application Web (Flask)
├── docker-compose.yml                 # Orchestration Docker
├── Dockerfile                         # Image Docker
├── openapi.yaml                       # Spécification OpenAPI
├── params.yaml                        # Hyperparamètres
└── requirements.txt                   # Dépendances Python
```

## <a id="logique-de-collecte--données"></a>🔍 Logique de Collecte & Données

### Stratégie "Power Users"

Pour garantir la pertinence des recommandations et réduire le bruit dans la matrice, nous appliquons un filtre lors de l'extraction des données (ETL).

Nous ne conservons un utilisateur dans notre base MongoDB que si :

- Il a donné une "star" à au moins **100 dépôts**.
- **Chacun** de ces 100 dépôts possède au moins **5 000 stars** au total.

**Pourquoi ?** Ce filtre vise à isoler les utilisateurs expérimentés qui effectuent une curation de haute qualité sur des projets technologiques majeurs, permettant à l'algorithme d'apprendre des motifs robustes.

### Schéma MongoDB
Les données sont stockées dans une collection (par exemple `users`).
**Structure réelle d'un document :**

```json
{
  "_id": "vanpelt",               // Identifiant utilisateur (String)
  "date_ajout": "2025-11-17...",  // Date d'extraction
  "repos": [                      // Liste simple de Strings ("owner/repo")
    "obra/superpowers",
    "jdx/mise",
    "dop251/goja",
    "grafana/k6"
    // ... (100 éléments filtrés)
  ]
}
```

## <a id="modélisation--mlflow"></a>🧠 Modélisation & MLflow

### Algorithmes
Le système supporte plusieurs approches de recommandation :
1.  **SVD (Singular Value Decomposition)** : Factorisation matricielle pour extraire les caractéristiques latentes.
2.  **(Autres modèles extensibles)** : L'architecture permet d'ajouter facilement de nouveaux modèles.

### MLflow Integration
**MLflow** est utilisé pour :
- Tracker les expériences d'entraînement.
- Enregistrer les métriques (précision, temps d'entraînement).
- Sauvegarder et versionner les modèles.

Une interface graphique MLflow est intégrée et lancée automatiquement avec l'application.

## <a id="installation"></a>🛠 Installation

### Option 1 : Docker (Recommandé)

Docker Compose permet de lancer tout l'écosystème en une commande.

```bash
docker-compose up --build
```
Cela lancera :
- 🌐 L'interface Web sur `http://localhost:5000`
- 🔬 L'interface MLflow sur `http://localhost:5001`
- 🔌 L'API sur `http://localhost:8000`

### Option 2 : Installation Locale

#### Prérequis
- Python 3.11+
- MongoDB (Instance locale ou Atlas)

#### Étapes

1. Clonet et installer les dépendances :
    ```bash
    git clone https://github.com/LucasBarbet/GitHub-API-Recommendation-of-repos.git
    cd GitHub-API-Recommendation-of-repos
    pip install -r requirements.txt
    ```

2. Lancer l'application :
    ```bash
    python app.py
    ```
    Le script lancera automatiquement le serveur Web et le serveur MLflow en arrière-plan.

## <a id="configuration"></a>⚙️ Configuration

### Variables d'environnement
Créez un fichier `.env` à la racine :

```bash
GITHUB_TOKEN="votre_token_ici"
MONGO_URI="mongodb://localhost:27017/"
```

### Paramètres d'entraînement
Le fichier `params.yaml` contrôle les hyperparamètres :

```yaml
svd_model:
  n_components: 50
  n_iter: 5
```

## <a id="utilisation"></a>▶️ Utilisation

### 1. Interface Web
Accédez à `http://localhost:5000`.

1.  **Dashboard** : Vue d'ensemble.
2.  **Recommandation** :
    - Entrez un nom d'utilisateur GitHub.
    - **Choisissez le modèle** (ex: `svd_model`) via le menu déroulant.
    - Réglez le nombre de recommandations (K).
    - Lancez la prédiction.
3.  **MLflow** : Cliquez sur le bouton "MLFlow" ou allez sur `http://localhost:5001` pour explorer les modèles entraînés.

### 2. API (FastAPI)
L'API expose les endpoints principaux. Documentation Swagger : `http://localhost:8000/docs`.

### 3. Entraînement
Pour réentraîner les modèles et les logger dans MLflow :

```bash
python src/GitHubAPIRecommendationOfRepos/train/main.py
```

## <a id="recherche-et-expérimentation"></a>🔬 Recherche et Expérimentation

Le dossier `research/` contient des notebooks Jupyter pour l'analyse de données et le prototypage :

- `trails.ipynb` : Tests préliminaires et exploration.
- `data.ipynb` : Analyse approfondie des données collectées.
- `visualisation_data.ipynb` : Visualisation des distributions (stars, utilisateurs, etc.).
- `transfert_BDD.ipynb` : Scripts pour la migration ou la manipulation de données en base.
