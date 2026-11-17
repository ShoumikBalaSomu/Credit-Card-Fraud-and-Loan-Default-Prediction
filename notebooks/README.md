# Jupyter Notebooks Guide

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShoumikBalaSomu/ML-Project-Group-08/blob/main/FinRisk_ML_Group08.ipynb)

### 🌟 Unified Master Notebook
- **[`FinRisk_ML_Group08.ipynb`](FinRisk_ML_Group08.ipynb)**: Complete, self-contained end-to-end pipeline uniting all project code, datasets, 6 model architectures, cost threshold sweeps, interactive simulators, TreeSHAP global plots, 5 local regulatory case studies, and fairness drift audits in a single runnable document.

---

### Modular Phase Notebooks
The modular notebooks in this directory demonstrate each individual phase of the project:
1. `01_eda.ipynb`: Exploratory Data Analysis, class distributions, and chronological time indexing.
2. `02_baseline.ipynb`: Benchmarking 5 model families with PR-AUC, ROC-AUC, and cost metrics.
3. `03_cost_sensitive.ipynb`: Comparison of 5 cost-handling strategies, optimal thresholding, and cost savings.
4. `04_explainability.ipynb`: Global Tree SHAP feature importances and local contrastive case study evaluations.

To re-execute all notebooks headless:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```
