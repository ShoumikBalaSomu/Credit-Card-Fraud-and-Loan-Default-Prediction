# Defense Presentation Slides Outline

**Course:** CSE431 - Machine Learning  
**Project Title:** Cost-Sensitive and Explainable Machine Learning for Financial Risk: Credit Card Fraud and Loan Default Prediction  
**Team:** Group 08 (Shoumik Bala, Md Israfil Hossen, Ariyan Islam Hridoy)  
**Duration:** 15 Minutes Presentation + 5 Minutes Q&A  

---

### Slide 1: Title Slide
- **Headline:** Cost-Sensitive and Explainable Machine Learning for Financial Risk
- **Subtitle:** Dual-Domain Evaluation: Credit Card Fraud Detection & Loan Default Prediction
- **Course:** CSE431 Machine Learning
- **Presenters (Group 08):**
  - Shoumik Bala (Group Lead & System Architect)
  - Md Israfil Hossen (Data & Modeling Engineer)
  - Ariyan Islam Hridoy (Cost & Explainability Specialist)
- **Visual:** Project logo / architecture diagram thumbnail.

---

### Slide 2: The Core Problem: The Financial Cost Mismatch
- **Bullet Points:**
  - Standard ML assumes symmetric loss (False Positives = False Negatives).
  - Financial Reality: Missing a \$5,000 defaulted loan is 10x worse than auditing an eligible applicant (\$500).
  - Accuracy Paradox: In a 0.58% fraud dataset, a dummy classifier gets 99.42% accuracy while losing 100% of fraud dollars.
  - Regulatory Barrier: Banks cannot deploy black-box models due to FCRA and ECOA mandates.
- **Talking Point:** "In banking, we do not maximize accuracy; we minimize total financial loss while maintaining operational interpretability."

---

### Slide 3: Dual-Domain Evaluation Setup
- **Bullet Points:**
  - **Task A: Credit Card Fraud Detection (Extreme Imbalance)**
    - Volume: 1,852,394 transactions (Sparkov generation).
    - Skew: 0.58% positive rate (1 fraud per 173 clean transactions).
    - Horizon: Real-time authorization (sub-10ms latency).
  - **Task B: Loan Default Prediction (Moderate Imbalance)**
    - Volume: 252,000 consumer credit profiles.
    - Skew: 12.30% positive default rate.
    - Horizon: Batch underwriting (high monetary loss per account).

---

### Slide 4: Data Hygiene & Leakage Prevention
- **Bullet Points:**
  - **Chronological Temporal Split for Fraud:**
    - Train: Calendar Year 2019 ($N = 1,296,675$).
    - Test: Calendar Year 2020 ($N = 555,719$).
    - Eliminates look-ahead bias and simulates real adversarial drift.
  - **Stratified Split for Loan Default:**
    - 80% Train, 20% Holdout Test (Seed: 42).
  - **Strict Pipeline Isolation:**
    - Preprocessing transformers fit exclusively on training data; frozen statistics applied to test sets.

---

### Slide 5: Feature Engineering: Geometry, Time & Solvency
- **Bullet Points:**
  - **Spatial Haversine Distance:** Spherical distance between cardholder residence and merchant POS coordinates.
  - **Cyclical & Temporal Flags:** `trans_hour` (0–23) and `is_weekend` indicator.
  - **Monetary Anomaly Ratios:** Current transaction amount vs. category historical mean.
  - **Solvency Indicators (Loan):** `income_per_exp_year`, `age_to_exp_ratio`, and composite `stability_score`.

---

### Slide 6: Mathematical Cost Matrix & Optimal Threshold ($p^*$)
- **Bullet Points:**
  - **Total Cost Function:** $C(p) = c_{FN} \cdot FN(p) + c_{FP} \cdot FP(p) + c_{TP} \cdot TP(p)$
  - **Cost Matrix Parameters:**
    - Fraud: $c_{FN} = \$200$, $c_{FP} = \$5$ ($40:1$ asymmetry).
    - Loan: $c_{FN} = \$5,000$, $c_{FP} = \$500$ ($10:1$ asymmetry).
  - **Theoretical Bayesian Optimal Threshold:**
    $$p^* = \frac{c_{FP} - c_{TN}}{(c_{FN} - c_{TP}) + (c_{FP} - c_{TN})}$$
  - **Theory vs Practice:**
    - Fraud: Theoretical $0.025 \approx$ Empirical **$0.023$**.
    - Loan: Theoretical $0.100 \approx$ Empirical **$0.142$**.

---

### Slide 7: Task A Results: CatBoost Dominance in Fraud
- **Bullet Points:**
  - **Best Model:** **CatBoost**
  - **PR-AUC:** **0.8364** (vs Random Forest: 0.7606, XGBoost: 0.7746, LogReg: 0.1984).
  - **ROC-AUC:** **0.9942** | **Recall:** **91.67%**.
  - **Cost Reduction:** **$21,720.00** (at 0.50) $\to$ **$12,160.00** (at $p^* = 0.023$).
  - **Financial Savings:** **44.01% savings** ($+\$9,560$ saved per 555K transactions).

---

### Slide 8: Task B Results: Random Forest in Loan Default Risk
- **Bullet Points:**
  - **Best Model:** **Random Forest**
  - **PR-AUC:** **0.6088** (vs LightGBM: 0.5925, XGBoost: 0.5750, LogReg: 0.1593).
  - **ROC-AUC:** **0.9139** | **Recall:** **85.39%** | **Precision:** 46.61%.
  - **Cost Reduction:** **$7,094,500.00** (at 0.50) $\to$ **$1,886,500.00** (at $p^* = 0.142$).
  - **Financial Savings:** **73.41% savings** ($+\$5,208,000$ loss reduction).

---

### Slide 9: Cost Strategies: Threshold Tuning vs SMOTE & Weights
- **Bullet Points:**
  - **Default Cutoff (0.50):** Misses 91.4% of defaults because default probability rarely exceeds 50%.
  - **SMOTE Resampling:** Generates artificial points that distort posterior probability calibration (Brier score degrades; savings only 9.87% - 11.88%).
  - **Post-hoc Threshold Tuning:** Retains true posterior probabilities, achieves highest PR-AUC and delivers lowest total dollar loss across both tasks.

---

### Slide 10: Visualizing the Trade-off (Cost Curves & Calibration)
- **Bullet Points:**
  - Cost vs Threshold Curve illustrates the convex financial loss landscape.
  - Steep penalty of high thresholds (runaway False Negatives).
  - Steep penalty of near-zero thresholds (operations overwhelmed by False Positives).
  - Calibration diagrams prove CatBoost and Random Forest produce reliable probabilities.

---

### Slide 11: Explainable AI: Global Feature Importance (TreeSHAP)
- **Bullet Points:**
  - **Fraud Key Drivers:**
    1. Transaction Hour (late night / early morning spikes risk).
    2. High-risk merchant categories (Grocery POS, Home goods).
    3. Transaction Amount deviation.
  - **Loan Default Key Drivers:**
    1. Asset ownership (Car & House ownership provide stability).
    2. Age and professional experience tenure.
    3. Employment stability score.

---

### Slide 12: Auditable Local Case Studies (Waterfall Analysis)
- **Bullet Points:**
  - **True Positive:** Legitimate high-risk capture (\$380 charge at 02:45 AM, $\hat{p} = 0.892$).
  - **False Positive:** Borderline travel charge (\$410 at 23:15 PM, $\hat{p} = 0.048$ flagged at $p^*=0.023$); resolved via zero-friction SMS prompt.
  - **False Negative:** Stealth micro-fraud (\$28.50 during business hours); proves the need for sequential graph behavioral modeling.

---

### Slide 13: Production Readiness & Latency Benchmarks
- **Bullet Points:**
  - **Fraud Model File Size:** **0.05 MB** (`best_model_fraud.joblib`).
  - **Inference Latency:** **0.0018 ms / sample** (CatBoost C++ runtime).
  - **System Throughput:** **553,243 transactions / second**.
  - **Loan Model:** 2.45 MB, 0.0305 ms latency (32,756 evaluations/sec).
  - Real-time SLA compatibility: Fully compliant with sub-10ms payment gateways.

---

### Slide 14: Model Governance & Ethical AI (ECOA/FCRA)
- **Bullet Points:**
  - Mitchell et al. (2019) Model Cards completed for both tasks.
  - Adverse Action Notices: Automated extraction of top negative SHAP factors for rejected applicants.
  - Bias Audit: Audited marital status and demographic proxies to prevent disparate impact.

---

### Slide 15: Conclusion & Key Takeaways
- **Key Takeaways:**
  1. Asymmetrical financial loss matrices must dictate model optimization.
  2. Post-hoc cost-optimal threshold tuning unlocks 44% to 73% dollar savings.
  3. TreeSHAP bridges high performance and strict banking regulatory compliance.
- **Q&A Session:** Open for questions from faculty evaluation committee.
