# GitHub Star Recommender 🌟

Ce projet est un système de recommandation de dépôts GitHub basé sur le filtrage collaboratif. Il utilise l'API GitHub pour collecter des données, MongoDB pour le stockage, et une **SVD (Singular Value Decomposition)** implémentée via **scikit-learn** pour générer des prédictions.

L'objectif est de suggérer des dépôts pertinents à un utilisateur en analysant les similitudes avec les historiques de "stars" d'utilisateurs experts ("Power Users").

## 📋 Table des matières

- [Architecture du Projet](#architecture-du-projet)
- [Logique de Collecte & Données](#logique-de-collecte--données)
- [Modélisation (SVD)](#modélisation-svd)
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
│   │   ├── main.py                    # Point d'entrée de l'API
│   │   ├── routes.py                  # Définition des routes
│   │   └── ...
│   └── GitHubAPIRecommendationOfRepos/ # Package principal
│       ├── components/                # Modules logiques (Ingestion, Transformation)
│       ├── train/                     # Pipelines d'entraînement (SVD)
│       ├── entity/                    # Entités de données
│       └── utils/                     # Utilitaires
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

Pour garantir la pertinence des recommandations et réduire le bruit dans la matrice, nous appliquons un filtre très strict lors de l'extraction des données (ETL).

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

## <a id="modélisation-svd"></a>🧠 Modélisation (SVD)

Le moteur de recommandation repose sur une approche de factorisation matricielle.

$$M \approx U \Sigma V^T$$

Nous utilisons TruncatedSVD de la bibliothèque **scikit-learn**.

- **Construction de la Matrice :** Transformation des données MongoDB en une "Sparse Matrix" (Utilisateurs $\times$ Dépôts).
- **Réduction de dimension :** L'algorithme compresse cette matrice pour extraire les caractéristiques latentes (goûts cachés des utilisateurs).
- **Prédiction :** Le produit scalaire des matrices réduites permet de prédire le score d'intérêt d'un utilisateur pour un dépôt non encore visité.

## <a id="installation"></a>🛠 Installation

### Option 1 : Docker (Recommandé)

Docker Compose permet de lancer l'API, l'application Web et (si configuré) une base de données locale.

```bash
docker-compose up --build
```
Cela lancera :
- L'API sur `http://localhost:8000`
- L'interface Web sur `http://localhost:5000`

### Option 2 : Installation Locale

#### Prérequis
- Python 3.8+
- MongoDB (Instance locale ou Atlas)
- Compte GitHub (pour le Token API)

#### Étapes

1. Cloner le dépôt :
    ```bash
    git clone https://github.com/LucasBarbet/GitHub-API-Recommendation-of-repos.git
    cd GitHub-API-Recommendation-of-repos
    ```

2. Créer un environnement virtuel et installer les dépendances :
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    
    pip install -r requirements.txt
    ```

## <a id="configuration"></a>⚙️ Configuration

### Variables d'environnement
Créez un fichier `.env` ou exportez vos variables pour la connexion à la base de données et l'API GitHub.

```bash
export GITHUB_TOKEN="votre_token_ici"
export MONGO_URI="mongodb://localhost:27017/"
```

### Paramètres du modèle
Modifiez `params.yaml` pour ajuster les hyperparamètres de la SVD (ex: nombre de composants).

```yaml
svd_model:
  n_components: 50
  n_iter: 5
  random_state: 42
```

## <a id="utilisation"></a>▶️ Utilisation

### 1. Interface Web (Flask)

Pour utiliser l'interface graphique conviviale :

```bash
python app.py
```
Accédez à `http://localhost:5000` pour rechercher des utilisateurs, voir leurs favoris et obtenir des recommandations.

### 2. API (FastAPI)

Pour lancer le backend API séparément :

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```
La documentation interactive (Swagger UI) est disponible sur `http://localhost:8000/docs`.

### 3. Entraînement du Modèle

Pour lancer le pipeline d'entraînement manuellement :

```bash
python src/GitHubAPIRecommendationOfRepos/train/main.py
```
Note : Assurez-vous d'avoir les données nécessaires dans votre base MongoDB ou vos fichiers locaux.

## <a id="api-reference"></a>📡 API Reference

L'API expose plusieurs endpoints pour interagir avec le système de recommandation :

- `GET /api/health` : Vérifie l'état du service.
- `GET /api/users/{username}` : Récupère les informations et les dépôts d'un utilisateur.
- `POST /api/users` : Ajoute un nouvel utilisateur à la base.
- `POST /api/users/{username}/repos` : Ajoute un dépôt aux favoris d'un utilisateur.
- `POST /api/predict` : Génère des recommandations pour un utilisateur donné.

Consultez le fichier `openapi.yaml` ou accédez à `/docs` une fois l'API lancée pour plus de détails.

## <a id="recherche-et-expérimentation"></a>🔬 Recherche et Expérimentation

Le dossier `research/` contient des notebooks Jupyter pour l'analyse de données et le prototypage :

- `trails.ipynb` : Tests préliminaires et exploration.
- `data.ipynb` : Analyse approfondie des données collectées.
- `visualisation_data.ipynb` : Visualisation des distributions (stars, utilisateurs, etc.).
- `transfert_BDD.ipynb` : Scripts pour la migration ou la manipulation de données en base.
