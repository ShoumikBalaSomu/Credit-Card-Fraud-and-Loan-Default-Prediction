# Financial Risk Machine Learning Streamlit Application

Interactive demonstration dashboard for **CSE431 Machine Learning Project (Group 08)**:
*"Cost-Sensitive and Explainable Machine Learning for Financial Risk: Credit Card Fraud and Loan Default Prediction"*.

---

## Features
1. **Executive Cost Simulator**:
   - Interactive threshold slider ($p \in [0.01, 0.99]$) with real-time recalculation of total cost, financial dollar savings vs. 0.50 threshold, False Negatives, False Positives, and Recall.
   - Dual-axis visual curve of Cost vs. Precision/Recall across all probability cutoffs.
2. **Model & Strategy Benchmarks**:
   - Comparative evaluation tables across all models (CatBoost, XGBoost, LightGBM, Random Forest, Logistic Regression).
   - Comparison of 5 cost-handling strategies (Standard, Class Weighting, SMOTE, Focal Loss proxy, Cost-Optimal Post-hoc Tuning).
   - Precision-Recall curves, calibration diagrams, and confusion matrices.
3. **Explainability & Trust (XAI)**:
   - Global SHAP beeswarm feature importance distribution.
   - Contrastive local instance explanations for True Positives, False Positives, and False Negatives.
4. **Live Inference Sandbox**:
   - Interactive form to input transaction parameters or loan applicant details and receive instant risk scoring and decisions.
5. **Model Governance & Cards**:
   - In-app rendering of formal Mitchell et al. (2019) Model Cards.

---

## How to Run Locally

### 1. Activate Environment
```bash
# From repository root:
source venv/bin/activate
```

### 2. Launch the Streamlit App
```bash
streamlit run app/app.py
```

The application will launch and open in your default browser at `http://localhost:8501`.
