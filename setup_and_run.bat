@echo off
echo Check for Conda...
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Conda not found in PATH. Please install Anaconda or Miniconda.
    pause
    exit /b 1
)

echo Creating/Updating Conda environment 'recommender_env' (Python 3.11)...
call conda env create -f environment.yml -y || call conda env update -f environment.yml --prune -y

echo Activating environment...
call conda activate recommender_env

echo ---------------------------------------------------
echo Launching Services locally...
echo 1. Backend API (FastAPI) on port 8000
echo 2. Frontend Web App (Flask) on port 5000
echo ---------------------------------------------------

:: Set PYTHONPATH so the API can find 'src' modules
set PYTHONPATH=%CD%

:: Start API in a new window
start "Backend API" cmd /k "conda activate recommender_env && uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload"

:: Wait a moment for API to initialize
timeout /t 5 /nobreak >nul

:: Start Web App in a new window, pointing to localhost API
set API_URL=http://127.0.0.1:8000
start "Frontend Web App" cmd /k "conda activate recommender_env && python app.py"

echo.
echo Services started!
echo Access the Web App at: http://127.0.0.1:5000
echo.
pause
