# Model Card: Random Forest for LOAN Risk Prediction

## 1. Model Overview
- **Model Architecture:** Random Forest
- **Task:** Binary Financial Risk Classification (loan)
- **Framework:** Python, scikit-learn, Gradient Boosting
- **Decision Strategy:** Cost-Sensitive Optimal Thresholding (Expected Loss Minimization)

## 2. Intended Use
- **Primary Use Case:** Real-time flagging of high-risk transactions or defaulting loan applicants.
- **Decision Criterion:** Threshold tuned to minimize operational financial loss ($FN 	imes cost_{FN} + FP 	imes cost_{FP}$) rather than naive 0.5 accuracy.
- **Out of Scope:** Automated punitive action without secondary compliance review.

## 3. Training & Validation Data
- **Dataset:** subhamjain/loan-prediction-based-on-customer-behavior
- **Validation Protocol:** Stratified holdout validation
- **Preprocessing:** Standard scaling and one-hot encoding strictly fit inside Pipeline folds.

## 4. Performance Summary
- **PR-AUC (Average Precision - Headline):** 0.6088
- **ROC-AUC:** 0.9139
- **Optimal Decision Threshold (p*):** 0.14
- **Recall at Optimal Threshold:** 0.8539
- **Precision at Optimal Threshold:** 0.4661
- **Total Expected Financial Cost:** $1,886,500.00
- **Cost Reduction vs Default (0.50) Threshold:** 73.41%
- **Brier Score:** 0.0855

## 5. Cost Matrix & Trade-offs
- **Cost Formulation:** 10:1 FN to FP penalty reflecting principal loss vs interest margin
- **FN Cost:** $5,000.00 | **FP Cost:** $500.00

## 6. Limitations & Governance
- **Subgroup Slices:** Monitored across demographic and transaction amount deciles.
- **Drift Monitoring:** Distributional shift in amount or temporal velocity requires quarterly recalibration.
- **Fairness Sanity Check:** Anonymized transactional identifiers prevent direct PII bias.
