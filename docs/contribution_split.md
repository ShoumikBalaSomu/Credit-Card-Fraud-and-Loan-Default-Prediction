# Team Collaboration & Governance

**Course:** CSE431 - Machine Learning  
**Project Title:** Cost-Sensitive and Explainable Machine Learning for Financial Risk: Credit Card Fraud and Loan Default Prediction  
**Team:** Group 08  

---

## Team Members

- **Shoumik Bala Somu** (`shoumik.bala@gmail.com`)
- **Md Israfil Hossen**
- **Ariyan Islam Hridoy**

---

## Collaborative Governance & Equal Contribution Statement

This project was developed through joint, equal collaboration among all team members. Rather than segregating work into isolated member silos, all three members contributed equally (33.3% each) across every phase of the research, implementation, evaluation, and documentation lifecycle.

### Joint Project Milestones

1. **Problem Formulation & Asymmetric Cost Modeling:**
   - Joint derivation of the Bayesian minimum risk decision rule and operational cost matrices ($40:1$ penalty for credit card fraud, $10:1$ penalty for loan default risk).
2. **Data Pipeline Engineering & Leakage Prevention:**
   - Strict chronological partitioning on 1.85M credit card transactions (2019 Train vs. 2020 Test) and stratified sampling on 252K loan applicant records.
   - Vectorized Haversine distance computations, temporal cyclical feature extraction, and credit applicant solvency metrics.
3. **Model Benchmarking & Calibration:**
   - Cross-evaluation of five model families (L2-Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost).
   - Probability calibration evaluation via Isotonic Regression, Platt Scaling, and Brier score analysis.
4. **Cost-Sensitive Threshold Optimization:**
   - Cross-validated 300-point grid search yielding optimal operational cutoffs ($p^* = 0.023$ for fraud, saving 44.01%; $p^* = 0.142$ for loan default, saving 73.41%).
5. **Explainable AI (XAI) & Governance:**
   - Game-theoretic global TreeSHAP beeswarm attributions and contrastive local instance waterfall audits for True Positives, False Positives, and False Negatives.
   - Authoring Mitchell et al. (2019) Model Cards and adverse action notice pipelines compliant with ECOA and FCRA regulations.
6. **Application Delivery & Technical Reporting:**
   - Multi-tab interactive Streamlit web application (`app/app.py`), production latency and throughput benchmarking, comprehensive course research paper (`reports/CSE431_Report.md`), and defense presentation outline (`reports/slides_outline.md`).
