# CredSafe — End-to-End Credit Card Fraud Detection

CredSafe is a full-stack machine learning project that detects credit card fraud and presents insights through an interactive dashboard. It includes data pipelines (ingestion → processing → training → evaluation), a Django web app for predictions and dashboards, MLflow tracking, and deployment guidance.

## Table of Contents
1. Project Overview
2. Tech Stack
3. Project Structure
4. Setup and Quick Start
5. Data & ML Pipeline
6. Training & Evaluation
7. Web Application (Django)
8. MLflow Tracking
9. DVC Artifacts (optional)
10. Deployment (Render / Docker)
11. Commands Cheat Sheet
12. Future Improvements

---

## 1) Project Overview
- **Goal**: Detect fraudulent transactions using supervised ML and provide an intuitive UI for predictions and analytics.
- **Highlights**:
  - Dynamic prediction form generated from feature names
  - Built-in Chart.js dashboard with class distribution and amount histogram
  - Optional Power BI and Tableau dashboards
  - Reproducible ML pipeline scripts and configuration
  - Ready-to-deploy Django app with Gunicorn/Whitenoise

## 2) Tech Stack
- **Backend**: Python 3.12, Django
- **ML**: NumPy, pandas, scikit-learn, joblib
- **Tracking**: MLflow
- **Orchestration**: DVC (optional), YAML configs
- **Frontend**: Bootstrap 5, Chart.js, Font Awesome
- **Serving**: Gunicorn (prod), Whitenoise for static files
- **Deployment**: Render (recommended) or Docker

## 3) Project Structure
```
Credit-card/
├─ Backend/
│  ├─ creditrisk/            # Django project (settings, urls, wsgi)
│  └─ predictor/             # Django app (views, urls, forms)
├─ src/                      # ML scripts (data & model code)
├─ pipelines/                # Optional pipeline wrapper
├─ config/                   # YAML + paths config
├─ artifacts/                # Data/model artifacts (DVC-tracked)
├─ templates/                # Django templates
├─ static/                   # CSS/Images
├─ mlflow_tracking/          # MLflow runner
├─ mlruns/                   # Local MLflow runs (if any)
├─ requirements.txt
└─ README.md
```

## 4) Setup and Quick Start
### A) Local environment
```powershell
# From repository root
Set-Location "c:\Users\Laptop\OneDrive\Desktop\prj\Credit-card"
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

### B) Run Django locally
```powershell
Set-Location "c:\Users\Laptop\OneDrive\Desktop\prj\Credit-card\Backend"
python manage.py migrate
python manage.py runserver
# App: http://127.0.0.1:8000/
```

### C) Place model artifacts (for predictions)
Put these files under `Credit-card/artifacts/preprocessed/`:
- logistic_regression_model.pkl
- scaler.pkl
- feature_names.pkl

If not present, the Predict page will still render but disable predictions or infer schema from raw CSV.

### D) Optional raw data for analytics
- Place CSV at: `Credit-card/artifacts/raw/enhanced_credit_card_fraud_dataset_2M.csv`
- Dashboard will infer target column (last column) and amount-like column for charts.

## 5) Data & ML Pipeline
- Config at `config/config.yaml` and helpers in `config/paths_config.py`.
- Main modules in `src/`:
  - **data_ingestion.py**: Load/split raw data, manage paths
  - **data_processing.py**: Clean, impute, scale, feature engineering
  - **model_training.py**: Train (e.g., Logistic Regression)
  - **model_evaluation.py**: Metrics (accuracy, ROC-AUC, etc.)
  - **hyperparameter_tuning.py**: Optional tuning
  - **logger.py**, **custom_exception.py**: Utilities

Typical flow:
1. Ingest raw CSV → split train/test
2. Process: handle missing values, scaling
3. Train model → serialize with joblib
4. Evaluate and log metrics
5. Store artifacts under `artifacts/preprocessed`

## 6) Training & Evaluation
### Run pipeline via Python scripts
```powershell
Set-Location "c:\Users\Laptop\OneDrive\Desktop\prj\Credit-card"
# Example: run a custom training script if provided
python src\model_training.py
python src\model_evaluation.py
```

### Track with MLflow (optional)
```powershell
Set-Location "c:\Users\Laptop\OneDrive\Desktop\prj\Credit-card"
mlflow ui
# Visit http://127.0.0.1:5000 to view experiments
```

## 7) Web Application (Django)
- URLs: `Backend/predictor/urls.py`
  - `/` → Home
  - `/predict/` → Predict form
  - `/dashboard/` → Built-in dashboard (Chart.js)
  - `/dashboard/powerbi/` → Power BI page
  - `/dashboard/tableau/` → Tableau page
- Views: `Backend/predictor/views.py`
  - Dynamically builds prediction form from `feature_names.pkl` or inferred schema.
  - Chart.js dashboard renders class distribution and amount histogram (quantile binned).

## 8) MLflow Tracking
- Run experiments and log metrics/artifacts.
- Local runs are stored under `mlruns/` by default.
- See `mlflow_tracking/run_mlflow.py` for example usage/integration.

## 9) DVC Artifacts (optional)
- DVC files under `.dvc/` and `artifacts/*.dvc` help version datasets, models, and metrics.
- Typical workflow:
```powershell
Set-Location "c:\Users\Laptop\OneDrive\Desktop\prj\Credit-card"
dvc pull   # fetch tracked artifacts from remote
# or dvc repro if pipeline is defined in dvc.yaml
```

## 10) Deployment
### Option A: Render (recommended)
- Root Directory: `Credit-card`
- Procfile (place in `Credit-card/`):
```
web: gunicorn creditrisk.wsgi:application --chdir Backend --bind 0.0.0.0:$PORT --log-file -
```
- Build command:
```
pip install -r requirements.txt
```
- Post-build command:
```
python Backend/manage.py collectstatic --noinput
```
- Env vars:
  - `DJANGO_SETTINGS_MODULE=creditrisk.settings`
  - `SECRET_KEY=your-secure-key`
  - `DEBUG=False`
  - `ALLOWED_HOSTS=<your-render-url>`
- Static files (in settings): add Whitenoise
```python
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"
```
- Ensure model artifacts live in the repo at `artifacts/preprocessed/`.

### Option B: Docker (portable)
Create a `Dockerfile` in `Credit-card/`:
```Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV DJANGO_SETTINGS_MODULE=creditrisk.settings
RUN python Backend/manage.py collectstatic --noinput
CMD ["gunicorn", "creditrisk.wsgi:application", "--chdir", "Backend", "--bind", "0.0.0.0:8000"]
```
Build & run:
```powershell
docker build -t credsafe ./Credit-card
docker run -p 8000:8000 credsafe
```

## 11) Commands Cheat Sheet
- Install deps: `pip install -r requirements.txt`
- Run server: `python Backend/manage.py runserver`
- DB migrate: `python Backend/manage.py migrate`
- Create superuser: `python Backend/manage.py createsuperuser`
- Collect static: `python Backend/manage.py collectstatic --noinput`
- Train model (example): `python src\model_training.py`
- Evaluate model (example): `python src\model_evaluation.py`
- MLflow UI: `mlflow ui`
- DVC pull: `dvc pull`

## 12) Future Improvements
- Add model registry and CI/CD (GitHub Actions) for automated training/deployments
- Improve feature engineering and try gradient boosting / deep learning
- Add data drift monitoring and feedback loop
- Container health checks and observability (Prometheus/Grafana)

---

### Badges (example)
- Built with: Django • scikit-learn • MLflow • Bootstrap • Chart.js
- Deploy on: Render / Docker