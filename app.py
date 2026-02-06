import os
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

# Backend API URL (default to internal docker alias if not set)
API_URL = os.environ.get("API_URL", "http://api:8000")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prepare_prediction', methods=['POST'])
def prepare_prediction():
    username = request.form.get('username', '').strip()
    
    # Call API to get user details
    try:
        response = requests.get(f"{API_URL}/api/users/{username}")
        if response.status_code == 200:
            user_data = response.json()
            current_repos = user_data.get('repos', [])
            return render_template('recommendation_setup.html', username=username, current_repos=current_repos)
        elif response.status_code == 404:
             return render_template('index.html', error="Utilisateur introuvable !", username=username)
        else:
            return render_template('index.html', error=f"Erreur DB: {response.text}", username=username)
    except requests.exceptions.RequestException as e:
        return render_template('index.html', error=f"Erreur de connexion API: {e}", username=username)

@app.route('/predict', methods=['POST'])
def predict():
    username = request.form.get('username')
    try:
        top_k = int(request.form.get('k', 5))
    except ValueError:
        top_k = 5
    
    # Call API to predict
    payload = {"user": username, "k": top_k}
    try:
        response = requests.post(f"{API_URL}/api/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            recommendations = result.get('recommendations', [])
            return render_template('results.html', username=username, recommendations=recommendations)
        elif response.status_code == 404:
            return render_template('index.html', error="Utilisateur introuvable pour la prédiction !")
        else:
             return render_template('index.html', error=f"Erreur de prédiction: {response.text}")
    except requests.exceptions.RequestException as e:
        return render_template('index.html', error=f"Erreur de connexion API: {e}")

@app.route('/add_user', methods=['POST'])
def add_user():
    username = request.form.get('username')
    if not username:
         return render_template('index.html', error="Veuillez entrer un nom d'utilisateur.")
    
    # Call API to add user
    payload = {"username": username}
    try:
        response = requests.post(f"{API_URL}/api/users", json=payload)
        if response.status_code == 201:
            return render_template('index.html', success=f"Utilisateur {username} ajouté avec succès !")
        elif response.status_code == 409:
            return render_template('index.html', error=f"L'utilisateur {username} existe déjà !")
        else:
            return render_template('index.html', error=f"Erreur d'ajout: {response.text}")
    except requests.exceptions.RequestException as e:
        return render_template('index.html', error=f"Erreur de connexion API: {e}")

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)