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
    import subprocess
    import socket
    import threading
    import platform
    import sys

    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def start_mlflow_ui():
        port = 5001
        if not is_port_in_use(port):
            print(f"Starting MLflow UI on port {port}...")
            
            # Use absolute path for DB
            current_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(current_dir, "src", "mlflow_store_reco", "mlflow.db")
            backend_store_uri = f"sqlite:///{db_path}"
            
            # Use absolute path for Artifact Root
            artifact_root = os.path.join(current_dir, "src", "mlflow_store_reco")
            
            print(f"MLflow DB Path: {backend_store_uri}")
            
            # AUTOMATICALLY UPGRADE DB
            try:
                print("Upgrading MLflow database schema...")
                upgrade_cmd = ["mlflow", "db", "upgrade", backend_store_uri]
                subprocess.run(upgrade_cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
                print("MLflow database upgrade successful.")
            except subprocess.CalledProcessError as e:
                print(f"Error upgrading MLflow database: {e}")
            except FileNotFoundError:
                print("mlflow command not found. Skipping DB upgrade.")

            cmd = [
                "mlflow", "ui",
                "--backend-store-uri", backend_store_uri,
                "--host", "0.0.0.0",
                "--port", str(port)
            ]
            
            # Run in background but redirect output to stdout/stderr so we can see it in docker logs
            process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True
            )
            print(f"MLflow UI started with PID: {process.pid}")
        else:
            print(f"MLflow UI port {port} is already in use.")

    # Start MLflow in a separate thread to avoid blocking (though Popen is non-blocking, good to wrap setup)
    start_mlflow_ui()
    
    app.run(host="0.0.0.0", port=5000, debug=True)