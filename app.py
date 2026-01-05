from flask import Flask, render_template, request
from components.prediction import PredictionPipeline
from utils.db_connector import get_database
from constants import MONGO_COLLECTION_NAME

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Récupérer l'entrée utilisateur
    username = request.form.get('username')
    
    # 2. Récupérer les infos de cet utilisateur dans MONGO
    db = get_database()
    user_data = db[MONGO_COLLECTION_NAME].find_one({"username": username}) # ou "_id": username selon votre modif

    if not user_data:
        return render_template('index.html', error="Utilisateur introuvable dans la base !")

    current_repos = user_data.get('repos', [])

    # 3. Faire la prédiction via le composant
    pipeline = PredictionPipeline()
    recommendations = pipeline.predict(username, current_repos)

    # 4. Afficher les résultats
    return render_template('results.html', username=username, recommendations=recommendations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)