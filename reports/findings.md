# Empirical Findings, Operational Edge Cases & Limitations

**Course:** CSE431 - Machine Learning  
**Project:** Cost-Sensitive and Explainable Financial Risk Machine Learning  
**Team:** Group 08  

---

## 1. Summary of Empirical Findings

1. **Failure of Standard Decision Cutoffs:**
   In imbalanced financial datasets (0.58% fraud and 12.30% loan default), the industry-default threshold of $p = 0.50$ produces extreme recall deficits:
   - For credit card fraud, $p = 0.50$ leaves 25.6% of fraud undetected (\$21,720.00 loss). Shifting to $p^* = 0.023$ increases recall to 91.67% and saves 44.01% in total costs.
   - For loan default, $p = 0.50$ misses 91.4% of defaults because default probabilities rarely exceed 0.50. Shifting to $p^* = 0.142$ boosts recall from 8.60% to 85.39% and saves 73.41% (\$5.21M reduction).

2. **Supremacy of Post-Hoc Threshold Optimization Over Synthetic Resampling (SMOTE):**
   Synthetic resampling via SMOTE distorts posterior probability estimates, increasing Brier scores and degrading decision calibration. Post-hoc threshold tuning preserves authentic class likelihoods while directly minimizing the target dollar cost function.

3. **Algorithm Suitability by Domain:**
   - **CatBoost** dominated credit card fraud detection (PR-AUC 0.8364 vs LightGBM 0.7902 and XGBoost 0.7746) due to its ordered boosting mechanism on categorical merchant categories.
   - **Random Forest** achieved superior generalization on consumer loan default risk (PR-AUC 0.6088 vs LightGBM 0.5925 and XGBoost 0.5750) by averaging over high-variance applicant attributes.

---

## 2. Operational Edge Cases Identified

1. **Sub-Threshold Velocity Attacks (Stealth Fraud):**
   Low-amount transactions (\$5 - \$30) conducted during normal daylight hours bypass static threshold rules ($\hat{p} < p^*$). While individual losses are small, aggregate losses from automated botnets accumulate unless paired with rolling velocity counters (e.g., transactions per hour per card).

2. **Borderline False Alarms on High-Value Travel:**
   Legitimate high-value travel bookings occurring late at night trigger false alarms due to the sharp positive weight of `trans_hour` and `amt`. Mitigating customer friction requires automated two-factor step-up verification rather than outright hard transaction declines.

3. **Thin-File / New-to-Credit Applicants:**
   In loan default modeling, applicants with zero prior work experience or very young age fall into high-risk deciles regardless of income. Alternative data sources (utility payment history, rental verification) are necessary to assess creditworthiness equitably.

---

## 3. Methodological Limitations

1. **Stationary Cost Matrices:**
   In real-world banking, the cost of a missed fraud is transaction-specific ($c_{FN} = \text{amt}$), whereas our matrix evaluates average loss categories ($c_{FN} = \$200$). Extending to instance-dependent cost-sensitive learning (Zadrozny et al., 2003) is a promising extension.
2. **Tabular Isolation:**
   Tabular feature vectors do not model relational topologies between shared devices, IP addresses, physical merchant POS terminals, and synthetic identity networks.
3. **Macroeconomic Shocks:**
   Loan default risk is susceptible to systemic macroeconomic shifts (interest rate hikes, inflation spikes) that alter baseline default probabilities over multi-year horizons.

---

## 4. Recommendations for Production Engineering

1. **Dynamic Threshold Drift Monitoring:** Re-estimate $p^*$ on rolling 30-day validation windows to adjust for seasonal consumer spending (e.g., Black Friday, holiday shopping).
2. **Automated Adverse Action Pipeline:** Deploy the SHAP contrastive waterfall module directly into credit committee underwriting portals to generate compliant adverse reason letters in real time.
3. **Tiered Decision Flow:**
   - $\hat{p} < 0.01$: Instant frictionless approval.
   - $0.01 \le \hat{p} < 0.05$: Passive step-up authentication (SMS prompt / push notification).
   - $\hat{p} \ge 0.05$: Manual investigator review or temporary hold.
