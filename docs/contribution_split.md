# Team Contribution Breakdown & Project Governance

**Course:** CSE431 - Machine Learning  
**Project Title:** Cost-Sensitive and Explainable Machine Learning for Financial Risk: Credit Card Fraud and Loan Default Prediction  
**Team Identification:** Group 08  

---

## Member Matrix & Core Responsibilities

| Member Name | Student ID / Role | Core Contributions | Work Percentage |
|---|---|---|:---:|
| **Shoumik Bala** | Group Lead / Architect | System architecture design, end-to-end pipeline integration (`scripts/run_all.py`), CatBoost/XGBoost/LightGBM hyperparameter optimization, benchmarking (`scripts/benchmark.py`), final technical report synthesis. | **35%** |
| **Md Israfil Hossen** | Data & Modeling Engineer | Data loading and chronological splitting protocols (`src/data/`), feature engineering pipelines (`src/features/`), baseline modeling (`src/models/baselines.py`), automated unit testing suite (`tests/`). | **33%** |
| **Ariyan Islam Hridoy** | Cost & XAI Specialist | Mathematical formulation of financial cost matrices (`src/evaluate/cost.py`), cost-optimal threshold tuning algorithm, TreeSHAP global and local waterfall case studies (`src/explain/`), interactive Streamlit application (`app/app.py`). | **32%** |

---

## Detailed Task Breakdown

### Shoumik Bala
- **Pipeline Architecture:** Designed the modular repository layout, configuration management (`configs/`), and unified artifact serialization protocol.
- **Tree Ensembles & Training:** Implemented scalable training routines for CatBoost, XGBoost, and LightGBM with early stopping and thread-safe headless Matplotlib configuration.
- **Inference Benchmarking:** Engineered the latency, memory footprint, and throughput profiling suite (`scripts/benchmark.py`).
- **Academic Reporting:** Lead author for the course paper (`reports/CSE431_Report.md`) and slide defense outline (`reports/slides_outline.md`).

### Md Israfil Hossen
- **Data Ingestion & Integrity:** Created dataset acquisition scripts (`src/data/download_data.py`), provenance hashing (`reports/results/data_audit.json`), and synthetic fixture generators.
- **Leakage Prevention:** Enforced strict chronological partition for 1.85M fraud records to eradicate look-ahead bias in financial time-series.
- **Feature Engineering:** Implemented vectorized Haversine distance functions, temporal cyclical transforms, and credit applicant stability metrics.
- **Quality Assurance:** Authored comprehensive pytest suites (`tests/`) covering data integrity, pipeline round-trips, and cost matrix invariants.

### Ariyan Islam Hridoy
- **Financial Cost Modeling:** Derived expected loss functions ($C(p) = c_{FN} \cdot FN + c_{FP} \cdot FP + c_{TP} \cdot TP$) and quantified $40:1$ and $10:1$ asymmetric loss matrices.
- **Threshold Optimization:** Implemented the cross-validated threshold optimization algorithm that achieved 44.01% savings on fraud and 73.41% on loan default.
- **Explainable AI (XAI):** Generated global TreeSHAP beeswarm summaries and contrastive local waterfall case studies (True Positive, False Positive, False Negative).
- **Interactive Visualization:** Developed the full-featured multi-tab Streamlit dashboard (`app/app.py`) for executive cost simulation and live sandbox testing.
