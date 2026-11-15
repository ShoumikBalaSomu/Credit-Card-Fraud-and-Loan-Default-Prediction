# Model Card: CatBoost for FRAUD Risk Prediction

## 1. Model Overview
- **Model Architecture:** CatBoost
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
- **PR-AUC (Average Precision - Headline):** 0.8364
- **ROC-AUC:** 0.9942
- **Optimal Decision Threshold (p*):** 0.02
- **Recall at Optimal Threshold:** 0.9167
- **Precision at Optimal Threshold:** 0.2299
- **Total Expected Financial Cost:** $12,160.00
- **Cost Reduction vs Default (0.50) Threshold:** 44.01%
- **Brier Score:** 0.0014

## 5. Cost Matrix & Trade-offs
- **Cost Formulation:** 50:1 FN to FP penalty reflecting catastrophic fraud loss vs operational friction
- **FN Cost:** $500.00 | **FP Cost:** $10.00

## 6. Limitations & Governance
- **Subgroup Slices:** Monitored across demographic and transaction amount deciles.
- **Drift Monitoring:** Distributional shift in amount or temporal velocity requires quarterly recalibration.
- **Fairness Sanity Check:** Anonymized transactional identifiers prevent direct PII bias.
