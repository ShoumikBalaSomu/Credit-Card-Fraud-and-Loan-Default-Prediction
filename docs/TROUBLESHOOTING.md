# Troubleshooting & Engineering Post-Mortem

This document records the technical hurdles encountered during the engineering and execution of the financial risk machine learning pipeline, along with the verified diagnostic steps and resolutions.

---

## Issue 1: Virtualenv Symlink Failure on Cloud Sync FUSE Mount

### Symptoms
Attempting to create a virtual environment inside `/home/krishna/OneDrive/Project/ML-Project(G-8)` failed with:
```text
Error: [Errno 95] Operation not supported: '.../venv/bin/python3'
```
File permissions could not be changed via `chmod +x`, and git operations experienced multi-second locking latency.

### Root Cause Analysis
The directory was mounted via `bindfs` pointing to `/run/user/1000/gvfs/onedrive:...` with the `--chmod-ignore` flag. FUSE mounts backed by GVFS/OneDrive do not support POSIX symlink creation or granular permission bits required by standard Python virtual environments and multiprocessing lockfiles.

### Resolution
The entire engineering workspace, virtual environment, and execution harness were relocated to the native NVMe SSD at `/home/krishna/Projects/ML-Project-Group-08`. High-speed NVMe I/O eliminated file-locking latency, and final outputs are synchronized back to the repository target.

---

## Issue 2: Multithreaded Process Termination (Tcl_AsyncDelete / SIGABRT)

### Symptoms
During gradient boosting training with CatBoost and Random Forest, worker processes abruptly terminated with exit code 134:
```text
Tcl_AsyncDelete: async handler deleted by the wrong thread
Fatal Python error: Aborted
```

### Root Cause Analysis
When `matplotlib.pyplot` is imported on a Linux system with desktop GUI libraries installed, it defaults to interactive backends (e.g., TkAgg or Qt5Agg). When gradient boosting models spawn background C++ OpenMP threads (`thread_count=-1`), the asynchronous Tcl/Tk handler is deallocated from an auxiliary thread rather than the main GUI thread, triggering an uncatchable C-level `abort()`.

### Resolution
Enforced the non-interactive headless backend at the very top of every module, script, and notebook before importing `pyplot`:
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

---

## Issue 3: Headless Jupyter Notebook Relative Path Resolution

### Symptoms
Executing notebooks via `jupyter nbconvert --execute --inplace notebooks/*.ipynb` failed with:
```text
ModuleNotFoundError: No module named 'src'
FileNotFoundError: [Errno 2] No such file or directory: 'reports/figures/...'
```

### Root Cause Analysis
Jupyter kernels set `os.getcwd()` to the folder containing the notebook (`ML-Project-Group-08/notebooks/`). Consequently, relative imports expecting the repository root as the working directory failed to locate `src/` and output directories.

### Resolution
Injected root-detection logic into the initial setup cell of all four notebooks:
```python
import os, sys
from pathlib import Path
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
```

---

## Issue 4: Pytest Fixture Relative Path Resolution Across Environments

### Symptoms
Running `pytest` from subfolders or parent folders failed in `test_data_loading.py` and `test_no_leakage.py` with:
```text
FileNotFoundError: [Errno 2] No such file or directory: 'tests/fixtures/fraud_sample.csv'
```

### Root Cause Analysis
Test files referenced `"tests/fixtures/fraud_sample.csv"` using relative string paths rather than anchor-relative paths.

### Resolution
Updated test modules to dynamically locate fixtures relative to `__file__`:
```python
fixture_dir = Path(__file__).resolve().parent / "fixtures"
sample_path = fixture_dir / "fraud_sample.csv"
```

---

## Issue 5: Dataset Schema Mismatch in Kaggle Repositories

### Symptoms
The repository prompt suggested `kartik2112/fraud-detection` might contain loan default data alongside credit card fraud data. Inspection revealed only `fraudTrain.csv` and `fraudTest.csv`.

### Root Cause Analysis
`kartik2112/fraud-detection` is exclusively a credit card fraud transaction dataset.

### Resolution
Identified and integrated `subhamjain/loan-prediction-based-on-customer-behavior` (252,000 applicant records, 12.30% positive default rate) under CC0 license. Configured the pipeline to handle both tasks seamlessly while preserving rigorous dual-domain evaluation.
