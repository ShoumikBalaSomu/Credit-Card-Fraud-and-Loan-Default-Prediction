# Reproducibility Guide & Environmental Specifications

To guarantee bit-exact, auditable reproducibility of all empirical metrics reported in this project, this document provides the complete hardware environment, software specifications, and execution instructions.

---

## 1. System & Hardware Specifications

Empirical benchmarking was executed natively on:
- **Operating System:** Fedora Linux 45 (Budgie Desktop Environment)
- **Kernel Version:** `Linux 6.13.12-200.fc41.x86_64` (or equivalent)
- **Architecture:** `x86_64`
- **Python Runtime:** Python `3.14.6`
- **Filesystem:** Native Btrfs/Ext4 on NVMe Solid State Drive (avoid network/cloud sync mounts such as OneDrive/FUSE for venv symlinks and multiprocessing locks).

---

## 2. Seed Control Protocol

All stochastic operations are controlled by a single master seed: **`42`**.
Configured in `configs/random_seed.yaml`:
```yaml
master_seed: 42
data_split_seed: 42
model_seed: 42
cv_seed: 42
```
Enforced via `src/utils/seeding.py`:
- `random.seed(42)`
- `np.random.seed(42)`
- `os.environ["PYTHONHASHSEED"] = "42"`
- Model initializers: `random_state=42` passed explicitly to Scikit-Learn, LightGBM, and XGBoost; `random_seed=42` passed to CatBoost.

---

## 3. Step-by-Step Execution Protocol

### Step 1: Clone Repository
```bash
git clone https://github.com/ShoumikBalaSomu/Credit-Card-Fraud-and-Loan-Default-Prediction.git
cd Credit-Card-Fraud-and-Loan-Default-Prediction
```

### Step 2: Create Isolated Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Verify Integrity with Unit Tests
Execute the offline test suite using synthetic fixtures:
```bash
pytest -v tests/
```
*Expected output: `10 passed in ~1.8s`.*

### Step 4: Execute Full End-to-End Pipeline
Run data ingestion, preprocessing, chronological splits, model training, cross-validated cost thresholding, calibration, SHAP explainability, and benchmarking:
```bash
python scripts/run_all.py
```
*All outputs, metrics, serialized preprocessors, model artifacts, and figures will be written directly into `reports/`.*

### Step 5: Launch Interactive Streamlit Dashboard
```bash
streamlit run app/app.py
```
*Accessible in your browser at `http://localhost:8501`.*

---

## 4. Primary Package Manifest

| Package | Locked Version | Role |
|---|---|---|
| `numpy` | `2.5.4` | Numerical tensor manipulations |
| `scipy` | `1.18.1` | Scientific routines and spatial distance |
| `pandas` | `3.1.1` | Tabular data ingestion and feature management |
| `scikit-learn`| `1.9.1` | Preprocessing pipelines, metrics, Logistic Regression, Random Forest |
| `imbalanced-learn`| `0.14.2` | Resampling and SMOTE implementations |
| `xgboost` | `3.4.1` | Extreme gradient boosting trees |
| `lightgbm` | `4.7.0` | Light gradient boosting trees |
| `catboost` | `1.2.10` | Ordered categorical boosting trees |
| `shap` | `0.54.1` | Game-theoretic TreeSHAP feature attributions |
| `matplotlib` | `3.11.2` | Headless figure rendering (`Agg` backend) |
| `seaborn` | `0.13.2` | Statistical data visualization |
| `streamlit` | `1.67.0` | Executive interactive web interface |
| `pytest` | `9.2.0` | Automated regression testing suite |
| `ruff` | `0.17.5` | Fast Python code linting |
| `black` | `26.10.1`| PEP 8 code formatting |
