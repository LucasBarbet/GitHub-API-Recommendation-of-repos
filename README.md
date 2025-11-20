# GitHub Star Recommender 🌟

Ce projet est un système de recommandation de dépôts GitHub basé sur le filtrage collaboratif. Il utilise l'API GitHub pour collecter des données, MongoDB pour le stockage, et une **SVD (Singular Value Decomposition)** implémentée via **scikit-learn** pour générer des prédictions.

L'objectif est de suggérer des dépôts pertinents à un utilisateur en analysant les similitudes avec les historiques de "stars" d'utilisateurs experts ("Power Users").

## 📋 Table des matières

- [Architecture du Projet](#-architecture-du-projet)
- [Logique de Collecte & Données](#-logique-de-collecte--donn%C3%A9es)
- [Modélisation (SVD)](#-mod%C3%A9lisation-svd)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Utilisation](#-utilisation)

## 📂 Architecture du Projet

Le projet suit une structure modulaire standard pour les pipelines de Machine Learning, séparant la configuration, le code source (src) et les expérimentations (research).

```text
├── .github/workflows/ # CI/CD pipelines  
├── config/ # Configuration globale (config.yaml)  
├── research/ # Notebooks pour l'analyse exploratoire (trails.ipynb)  
├── src/  
│ └── githubRecommender/ # Package principal  
│ ├── components/ # Modules logiques (Data Ingestion, Transformation, Model Trainer)  
│ ├── config/ # Gestionnaires de configuration  
│ ├── entity/ # Data Classes et entités  
│ ├── pipeline/ # Orchestration des étapes (Train, Predict)  
│ └── utils/ # Fonctions utilitaires communes  
├── templates/ # Fichiers HTML pour l'interface Web (index.html)  
├── app.py # Application Web (Flask/Streamlit)  
├── main.py # Point d'entrée pour l'exécution du pipeline  
├── params.yaml # Hyperparamètres du modèle SVD  
├── schema.yaml # Schéma des données  
├── Dockerfile # Conteneurisation de l'application  
└── requirements.txt # Dépendances Python
```

## 🔍 Logique de Collecte & Données

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

## 🧠 Modélisation (SVD)

Le moteur de recommandation repose sur une approche de factorisation matricielle.

\$\$M \\approx U \\Sigma V^T\$\$

Nous utilisons TruncatedSVD de la bibliothèque **scikit-learn**.

- **Construction de la Matrice :** Transformation des données MongoDB en une "Sparse Matrix" (Utilisateurs \$\\times\$ Dépôts).
- **Réduction de dimension :** L'algorithme compresse cette matrice pour extraire les caractéristiques latentes (goûts cachés des utilisateurs).
- **Prédiction :** Le produit scalaire des matrices réduites permet de prédire le score d'intérêt d'un utilisateur pour un dépôt non encore visité.

## 🛠 Installation

### Prérequis

- Python 3.8+
- MongoDB (Instance locale ou Atlas)
- Compte GitHub (pour le Token API)

### Étapes

- Cloner le dépôt :  
    Bash  
    git clone <https://github.com/LucasBarbet/GitHub-API-Recommendation-of-repos.git>  
    cd GitHub-API-Recommendation-of-repos  

- Créer un environnement virtuel et installer les dépendances :  
    Bash  
    python -m venv venv  
    \# Windows  
    venv\\Scripts\\activate  
    \# Linux/Mac  
    source venv/bin/activate  
    <br/>pip install -r requirements.txt  

## ⚙️ Configuration

- Variables d'environnement :  
    Créez un fichier .env ou exportez vos variables pour la connexion à la base de données et l'API GitHub.  
    Bash  
    export GITHUB_TOKEN="votre_token_ici"  
    export MONGO_URI="mongodb://localhost:27017/"  

- Paramètres du modèle :  
    Modifiez params.yaml pour ajuster les hyperparamètres de la SVD (ex: nombre de composants).  
    YAML  
    svd_model:  
    n_components: 50  
    n_iter: 5  
    random_state: 42  

## ▶️ Utilisation

### 1\. Exécuter le Pipeline (ETL + Entraînement)

Pour lancer la collecte des données, le traitement et l'entraînement du modèle via le point d'entrée principal :

Bash

python main.py  

_Cela déclenchera les pipelines définis dans src/.../pipeline/._

### 2\. Lancer l'Application Web

Pour utiliser l'interface graphique et visualiser les recommandations :

Bash

python app.py  

L'application sera accessible sur <http://localhost:5000> (ou le port défini).

### 3\. Expérimentation

Les notebooks dans le dossier research/ (ex: trails.ipynb) peuvent être utilisés pour tester de nouvelles hypothèses ou visualiser la distribution des stars avant de modifier le code de production.
