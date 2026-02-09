import os
import requests
from flask import Flask, render_template, request, jsonify

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
        elif response.status_code == 404:
             return render_template('index.html', error="Utilisateur introuvable !", username=username)
        else:
            return render_template('index.html', error=f"Erreur DB: {response.text}", username=username)
    except requests.exceptions.RequestException as e:
        return render_template('index.html', error=f"Erreur de connexion API: {e}", username=username)

    # Fetch available models from API
    try:
        response = requests.get(f"{API_URL}/api/models")
        if response.status_code == 200:
            models = response.json().get("models", [])
        else:
            models = ["svd_model"]
    except:
        models = ["svd_model"]

    return render_template('recommendation_setup.html', username=username, current_repos=current_repos, models=models)

@app.route('/add_favorite', methods=['POST'])
def add_favorite():
    try:
        data = request.get_json()
        username = data.get('username')
        repo_name = data.get('repo_name')
        
        if not username or not repo_name:
            return jsonify({"error": "Missing data"}), 400

        # Call API to add repo
        response = requests.post(f"{API_URL}/api/users/{username}/repos", json={"repo_name": repo_name})
        
        if response.status_code == 200:
            return jsonify({"success": True})
        else:
            return jsonify({"error": response.text}), response.status_code
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def recommend():
    username = request.form.get('username')
    k = request.form.get('k', 5)
    model_name = request.form.get('model_name', 'svd_model')
    
    payload = {
        "user": username,
        "k": int(k),
        "model_name": model_name
    }
    
    try:
        response = requests.post(f"{API_URL}/api/predict", json=payload)
        response.raise_for_status()
        data = response.json()
        recommendations = data.get('recommendations', [])
        
        return render_template('results.html', username=username, recommendations=recommendations, model_name=model_name)
    except Exception as e:
        return f"Error connecting to API: {e}", 500

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