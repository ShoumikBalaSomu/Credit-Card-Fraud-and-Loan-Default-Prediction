# Empirical Methodology: Cost-Sensitive & Explainable Financial Risk Machine Learning

## 1. Experimental Overview & Research Design

Financial risk assessment encompasses heterogeneous decision problems characterized by asymmetrical cost structures and class imbalance. This project investigates and compares two complementary domains:
1. **Task A: Credit Card Fraud Detection (Extreme Imbalance: 0.58% positive rate)**
   - High volume, low fraud incidence, real-time millisecond decision horizon.
   - Non-stationary temporal patterns and adversarial concept drift.
2. **Task B: Loan Default Prediction (Moderate Imbalance: 12.30% positive rate)**
   - High monetary exposure per decision, structural demographic and financial indicators, regulated credit underwriting requirements.

---

## 2. Dataset Splitting & Leakage Prevention Protocols

### 2.1 Chronological vs. Stratified Splitting
- **Task A (Temporal Separation):** Rather than standard random shuffling (which leaks future seasonal patterns, spending trajectories, and merchant compromise spikes into training sets), transactions are split chronologically:
  - Training Set: Calendar year 2019 ($N = 1,296,675$, Fraud rate = $0.579\%$).
  - Holdout Test Set: Calendar year 2020 ($N = 555,719$, Fraud rate = $0.386\%$).
- **Task B (Stratified Shuffle Split):** A stratified 80/20 train/test split with random seed $42$ preserves the marginal label distribution across folds without temporal leakage.

### 2.2 Strict Pipeline Encapsulation
- Standard scalers, one-hot encoders, and target statistics are computed **strictly on the training split**.
- Holdout test sets are transformed using frozen training statistics.
- Preprocessing pipelines (`preprocessor_fraud.joblib`, `preprocessor_loan.joblib`) are serialized independently to ensure reproducible inference.

---

## 3. Feature Engineering

### 3.1 Task A (Fraud Detection)
1. **Spatial Distance (`dist_to_merch_km`):** Vectorized spherical Haversine distance between cardholder residence $(\phi_1, \lambda_1)$ and merchant coordinates $(\phi_2, \lambda_2)$:
   $$d = 2 R \arcsin \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos\phi_1 \cos\phi_2 \sin^2\left(\frac{\Delta \lambda}{2}\right)}$$
2. **Temporal Features:** `trans_hour` (0–23), `is_weekend` flag (Saturday/Sunday indicator).
3. **Demographic & Monetary Indicators:** Cardholder age derived from date of birth, log-transformed amount $\ln(1 + \text{amt})$, and ratio of current transaction amount to merchant category historical mean.

### 3.2 Task B (Loan Default)
1. **Human Capital Return:** `income_per_exp_year` = $\frac{\text{Income}}{\text{Experience} + 1}$.
2. **Age-Career Maturity:** `age_to_exp_ratio` = $\frac{\text{Age}}{\text{Experience} + 1}$.
3. **Residential & Career Stability:** `stability_score` = $\frac{\text{Current Job Years} + \text{Current House Years}}{\text{Age} + 1}$ (derived from `CURRENT_JOB_YRS` and `CURRENT_HOUSE_YRS`).

---

## 4. Model Architectures & Implementations

We evaluate 5 distinct algorithms spanning linear baselines, bagging, and gradient boosting:
1. **L2-Regularized Logistic Regression:** Scaled numerical inputs with one-hot categorical features providing a parametric benchmark.
2. **Random Forest:** Bagging ensemble of 100 deep decision trees using bootstrap aggregation.
3. **XGBoost (Extreme Gradient Boosting):** Exact greedy tree split algorithm with depth regularization and histogram building.
4. **LightGBM:** Fast leaf-wise tree growth with gradient-based one-side sampling (GOSS).
5. **CatBoost:** Ordered boosting with symmetric oblivious decision trees natively handling high-cardinality categorical variables.

---

## 5. Cost-Sensitive Strategies

We benchmark five class-handling strategies:
1. **Strategy 1: Unweighted Default (Baseline):** Standard loss optimization with fixed $p = 0.50$ classification cutoff.
2. **Strategy 2: Balanced Class Weighting:** Heuristic reweighting assigning inversely proportional weights to class frequencies:
   $$w_1 = \frac{N}{2 \cdot N_1}, \quad w_0 = \frac{N}{2 \cdot N_0}$$
3. **Strategy 3: Resampling (SMOTE):** Synthetic Minority Over-sampling Technique applied to the training fold to synthetically balance class representations.
4. **Strategy 4: Focal Loss Proxy:** Asymmetric sample reweighting focusing training on hard examples while down-weighting well-classified negatives.
5. **Strategy 5: Cost-Optimal Post-hoc Threshold Tuning:** Retaining true unweighted posterior probabilities and directly optimizing the operational decision threshold $p^*$ to minimize total dollar cost:
   $$p^* = \arg\min_{p \in (0, 1)} C(p)$$

---

## 6. Model Explainability Framework

Model transparency is enforced via Shapley Additive Explanations (SHAP) based on cooperative game theory:
$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$$
- **Global Interpretability:** TreeSHAP calculates exact attribution values across test samples, producing beeswarm distributions that highlight monotonic and non-linear risk drivers.
- **Local Explanations & Contrastive Case Studies:** Instance-level waterfall decompositions illuminate decision trajectories for True Positives, False Positives, and False Negatives, fulfilling adverse action notice requirements under the Fair Credit Reporting Act (FCRA).
