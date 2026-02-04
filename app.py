from flask import Flask, render_template, request
from src.GitHubAPIRecommendationOfRepos.components.prediction import PredictionPipeline
from src.GitHubAPIRecommendationOfRepos.utils.db_connector import get_database
from src.GitHubAPIRecommendationOfRepos.constants import MONGO_COLLECTION_NAME

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

import sys

@app.route('/prepare_prediction', methods=['POST'])
def prepare_prediction():
    username = request.form.get('username', '').strip()
    sys.stderr.write(f"DEBUG: prepare_prediction called with username='{username}'\n")
    
    db = get_database()
    # Recherche par "_id" (le nom d'utilisateur est stocké dans _id)
    user_data = db[MONGO_COLLECTION_NAME].find_one({"_id": username})
    sys.stderr.write(f"DEBUG: user_data found: {user_data}\n")

    if not user_data:
        sys.stderr.write("DEBUG: User not found, returning index.html with error\n")
        return render_template('index.html', error="Utilisateur introuvable !", username=username)

    current_repos = user_data.get('repos', [])
    return render_template('recommendation_setup.html', username=username, current_repos=current_repos)

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Récupérer l'entrée utilisateur
    username = request.form.get('username')
    try:
        top_k = int(request.form.get('k', 5))
    except ValueError:
        top_k = 5
    
    # 2. Récupérer les infos de cet utilisateur dans MONGO
    db = get_database()
    # Recherche par "_id"
    user_data = db[MONGO_COLLECTION_NAME].find_one({"_id": username})

    if not user_data:
        return render_template('index.html', error="Utilisateur introuvable dans la base !")

    current_repos = user_data.get('repos', [])

    # 3. Faire la prédiction via le composant
    pipeline = PredictionPipeline()
    try:
        recommendations = pipeline.predict(username, current_repos, top_k=top_k)
    except TypeError: # Fallback if predict doesn't support top_k yet
        recommendations = pipeline.predict(username, current_repos)
        recommendations = recommendations[:top_k] # Slice result if needed

    # 4. Afficher les résultats
    return render_template('results.html', username=username, recommendations=recommendations)

@app.route('/add_user', methods=['POST'])
def add_user():
    username = request.form.get('username')
    if not username:
         return render_template('index.html', error="Veuillez entrer un nom d'utilisateur.")
    
    db = get_database()
    collection = db[MONGO_COLLECTION_NAME]
    
    # Vérification par "_id"
    if collection.find_one({"_id": username}):
        return render_template('index.html', error=f"L'utilisateur {username} existe déjà !")
        
    # Insertion avec "_id" = username
    collection.insert_one({"_id": username, "repos": []})
    return render_template('index.html', success=f"Utilisateur {username} ajouté avec succès !")

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)