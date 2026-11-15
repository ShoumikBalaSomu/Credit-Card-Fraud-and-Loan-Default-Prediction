# Model Card: Ensemble Blend for FRAUD Risk Prediction

## 1. Model Overview
- **Model Architecture:** Ensemble Blend
- **Task:** Binary Financial Risk Classification (fraud)
- **Framework:** Python, scikit-learn, Gradient Boosting
- **Decision Strategy:** Cost-Sensitive Optimal Thresholding (Expected Loss Minimization)

## 2. Intended Use
- **Primary Use Case:** Real-time flagging of high-risk transactions or defaulting loan applicants.
- **Decision Criterion:** Threshold tuned to minimize operational financial loss ($FN 	imes cost_{FN} + FP 	imes cost_{FP}$) rather than naive 0.5 accuracy.
- **Out of Scope:** Automated punitive action without secondary compliance review.

## 3. Training & Validation Data
- **Dataset:** kartik2112/fraud-detection
- **Validation Protocol:** Strict chronological train/test split
- **Preprocessing:** Standard scaling and one-hot encoding strictly fit inside Pipeline folds.

## 4. Performance Summary
- **PR-AUC (Average Precision - Headline):** 0.8398
- **ROC-AUC:** 0.9944
- **Optimal Decision Threshold (p*):** 0.12
- **Recall at Optimal Threshold:** 0.8333
- **Precision at Optimal Threshold:** 0.5882
- **Total Expected Financial Cost:** $14,980.00
- **Cost Reduction vs Default (0.50) Threshold:** 36.74%
- **Brier Score:** 0.0013

## 5. Cost Matrix & Trade-offs
- **Cost Formulation:** 50:1 FN to FP penalty reflecting catastrophic fraud loss vs operational friction
- **FN Cost:** $500.00 | **FP Cost:** $10.00

## 6. Limitations & Governance
- **Subgroup Slices:** Monitored across demographic and transaction amount deciles.
- **Drift Monitoring:** Distributional shift in amount or temporal velocity requires quarterly recalibration.
- **Fairness Sanity Check:** Anonymized transactional identifiers prevent direct PII bias.
