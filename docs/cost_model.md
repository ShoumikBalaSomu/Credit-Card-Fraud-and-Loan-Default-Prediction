# Mathematical Derivation of Financial Cost Matrices & Optimal Thresholds

## 1. Limitations of Symmetric Evaluation Metrics in Banking

Standard machine learning evaluation metrics implicitly assume symmetric misclassification costs:
- **Accuracy** fails catastrophically under extreme imbalance: predicting all transactions as legitimate in a 0.58% fraud environment yields 99.42% accuracy while permitting 100% of fraud losses to escape.
- **ROC-AUC** integrates over all possible operational thresholds with uniform weight, heavily weighting unrealistic operating regimes where false alarm rates exceed 30%.
- **F1 Score** equally weights precision and recall ($\beta = 1$), ignoring that a missed \$5,000 defaulted loan incurs ten times the penalty of a \$500 underwriting friction cost.

Financial institutions operate under strict budget constraints where the economic objective function is total expected monetary loss.

---

## 2. Cost Matrix Formulation

Let $Y \in \{0, 1\}$ denote the true class label (where 1 represents the high-risk positive class: Fraud or Default), and let $\hat{Y} \in \{0, 1\}$ denote the binary decision. The cost matrix defines the financial consequence $C(\hat{y}, y)$ associated with each outcome:

| | Actual Negative ($y = 0$) | Actual Positive ($y = 1$) |
|---|---|---|
| **Predict Negative ($\hat{y} = 0$)** | $c_{TN}$ (Clean Transaction) | $c_{FN}$ (Undetected Risk / Loss) |
| **Predict Positive ($\hat{y} = 1$)** | $c_{FP}$ (False Alarm / Denied Good Credit) | $c_{TP}$ (Verification / Workout Cost) |

### 2.1 Task A: Credit Card Fraud Matrix ($40:1$ Loss Asymmetry)
- $c_{FN} = \$200.00$: Average direct financial chargeback liability absorbed by the issuing bank when an unauthorized transaction is approved.
- $c_{FP} = \$5.00$: Friction and communication cost (automated SMS alert, temporary hold, manual verification review).
- $c_{TP} = \$5.00$: Investigation overhead to confirm and reverse the fraudulent charge.
- $c_{TN} = \$0.00$: Zero-friction automated approval of legitimate consumer transaction.

### 2.2 Task B: Loan Default Matrix ($10:1$ Loss Asymmetry)
- $c_{FN} = \$5,000.00$: Net charge-off principal write-down and recovery loss on a defaulted credit facility.
- $c_{FP} = \$500.00$: Opportunity loss of foregone net interest margin plus customer acquisition amortization from turning away a creditworthy applicant.
- $c_{TP} = \$500.00$: Specialized credit committee underwriting review or proactive workout restructuring.
- $c_{TN} = \$0.00$: Timely full principal repayment and interest realization.

---

## 3. Total Operational Cost Function

Given an operating probability threshold $p \in [0, 1]$, an instance $x$ is flagged positive if and only if $\hat{P}(Y=1 \mid x) \ge p$. Across a test cohort of size $N$, total operational cost is:

$$C(p) = c_{FN} \cdot FN(p) + c_{FP} \cdot FP(p) + c_{TP} \cdot TP(p) + c_{TN} \cdot TN(p)$$

Because $c_{TN} = 0$, this simplifies to:
$$C(p) = c_{FN} \cdot FN(p) + c_{FP} \cdot FP(p) + c_{TP} \cdot TP(p)$$

---

## 4. Theoretical Derivation of the Optimal Decision Threshold

Let $\hat{p} = \hat{P}(Y=1 \mid x)$ denote a well-calibrated posterior risk probability. The expected loss of choosing action $\hat{y} = 1$ versus action $\hat{y} = 0$ is:

$$\mathbb{E}[\text{Loss} \mid \hat{y} = 1] = \hat{p} \cdot c_{TP} + (1 - \hat{p}) \cdot c_{FP}$$
$$\mathbb{E}[\text{Loss} \mid \hat{y} = 0] = \hat{p} \cdot c_{FN} + (1 - \hat{p}) \cdot c_{TN}$$

An optimal Bayesian risk minimizer selects action $\hat{y} = 1$ when:
$$\mathbb{E}[\text{Loss} \mid \hat{y} = 1] \le \mathbb{E}[\text{Loss} \mid \hat{y} = 0]$$

Substituting expressions:
$$\hat{p} \cdot c_{TP} + (1 - \hat{p}) \cdot c_{FP} \le \hat{p} \cdot c_{FN} + (1 - \hat{p}) \cdot c_{TN}$$

Grouping terms with $\hat{p}$:
$$c_{FP} - c_{TN} \le \hat{p} \cdot \left[ (c_{FN} - c_{TP}) + (c_{FP} - c_{TN}) \right]$$

Solving for $\hat{p}$ yields the theoretical Bayesian optimal decision threshold $p^*$:

$$p^* = \frac{c_{FP} - c_{TN}}{(c_{FN} - c_{TP}) + (c_{FP} - c_{TN})}$$

### 4.1 Theoretical vs. Empirical Thresholds

1. **Credit Card Fraud:**
   $$p^*_{\text{theoretical}} = \frac{5 - 0}{(200 - 5) + (5 - 0)} = \frac{5}{195 + 5} = \frac{5}{200} = 0.025$$
   Our empirical cross-validated grid search yielded $p^*_{\text{empirical}} = 0.023$. The tight correspondence ($0.023 \approx 0.025$) validates both the probability calibration quality of CatBoost and the theoretical derivation.

2. **Loan Default Prediction:**
   $$p^*_{\text{theoretical}} = \frac{500 - 0}{(5000 - 500) + (500 - 0)} = \frac{500}{4500 + 500} = \frac{500}{5000} = 0.100$$
   Our empirical cross-validated grid search yielded $p^*_{\text{empirical}} = 0.142$, accounting for empirical density concentrations along the decision boundary.

---

## 5. Economic Savings Quantification

We benchmark cost-optimal operation against the un-tuned default standard cutoff of $p = 0.50$:

$$\Delta C = C(0.50) - C(p^*)$$
$$\text{Relative Savings (\%)} = \frac{C(0.50) - C(p^*)}{C(0.50)} \times 100\%$$

### Empirical Results Summary
- **Task A (Fraud Detection):** Baseline cost of **$21,720.00** reduced to **$12,160.00** at $p^* = 0.023$, capturing **44.01% financial savings** ($+\$9,560$ preserved per 555K test transactions).
- **Task B (Loan Default):** Baseline cost of **$7,094,500.00** reduced to **$1,886,500.00** at $p^* = 0.142$, delivering **73.41% financial savings** ($+\$5,208,000$ loss reduction).
