# Jupyter Notebooks Guide

The notebooks in this directory demonstrate each phase of the project:
1. `01_eda.ipynb`: Exploratory Data Analysis, class distributions, and chronological time indexing.
2. `02_baseline.ipynb`: Benchmarking 5 model families with PR-AUC, ROC-AUC, and cost metrics.
3. `03_cost_sensitive.ipynb`: Comparison of 5 cost-handling strategies, optimal thresholding, and cost savings.
4. `04_explainability.ipynb`: Global Tree SHAP feature importances and local contrastive case study evaluations.

To re-execute all notebooks headless:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```
