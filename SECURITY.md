# Security Policy, Compliance & Threat Model
## CSE431: Cost-Sensitive & Explainable Financial Risk Machine Learning

---

## 1. Executive Summary & Regulatory Framework

This repository implements machine learning systems designed for deployment within highly regulated financial technology and card-issuing banking environments. As financial risk models handle sensitive transaction records, they are subject to strict regulatory frameworks:

1. **PCI-DSS v4.0 (Payment Card Industry Data Security Standard)**
   - *Requirement 3: Protect Stored Account Data.*
   - *Requirement 3.4:* Primary Account Numbers (PANs) must be rendered unreadable wherever they are stored (using strong cryptography, hashing, or tokenization).
2. **GDPR (General Data Protection Regulation - EU 2016/679)**
   - *Article 5(1)(c) (Data Minimisation):* Personal data must be adequate, relevant, and limited to what is strictly necessary in relation to the purposes for which they are processed.
   - *Article 25 (Data Protection by Design and by Default):* Integration of privacy safeguards directly into feature engineering pipelines.
3. **Model Governance & SR 11-7 / OCC 2011-12 Principles**
   - Rigorous prevention of temporal lookahead bias, target leakage, and out-of-distribution adversarial degradation.

---

## 2. PII Sanitization & Data Minimization Protocol

Raw transaction datasets (e.g. Kaggle `fraudTrain.csv` / `fraudTest.csv`) contain direct personal identifiers and raw card numbers. The ingestion pipeline executes the following mandatory sanitization steps prior to feature generation or model training:

| Column Name | Category | Risk / Compliance Violation | Action Taken & Replacement |
| :--- | :--- | :--- | :--- |
| `cc_num` | Financial Account Data | **PCI-DSS Req 3.4**: Unencrypted PAN (16 digits) | One-way HMAC-SHA256 tokenization (`tok_<hash>[:16]`). Raw PAN dropped immediately. |
| `first`, `last` | Direct PII | **GDPR Art 4(1)**: Direct personal identity; memorization hazard | Permanently dropped. Zero predictive generalizability. |
| `street` | Direct PII | **GDPR Art 4(1)**: Exact physical home address | Permanently dropped. |
| `dob` | Direct PII | **GDPR Art 4(1)**: Exact date of birth | Transformed into relative `age_years` at transaction time: `(trans_date - dob).dt.days / 365.25`. Raw `dob` dropped. |
| `trans_num` | Identifier / UUID | Spurious tree splits; non-generalizable random token | Permanently dropped. |
| `Unnamed: 0` | Index Artifact | Unintended monotonic ordering leak | Permanently dropped. |
| `lat`, `long` | Geolocation PII | Precise residential coordinates | Retained only for relational calculation: `distance_km = haversine(cust_lat, cust_lon, merch_lat, merch_lon)`. |

### Programmatic PII Compliance Assertion
In both the standalone audit suite (`src/security_audit.py`) and the unified Jupyter notebook (`CSE431_CostSensitive_Explainable_Financial_Risk_ML.ipynb`), the following runtime assertion executes before training:

```python
assert_pci_gdpr_compliance(feature_columns)
```

If any forbidden column or cleartext PAN string format is detected in the input schema, execution halts immediately with a `CRITICAL` compliance violation.

---

## 3. Temporal Leakage & Integrity Invariants

Financial fraud detection models face severe temporal non-stationarity and concept drift. Evaluating models using random k-fold cross-validation or unconstrained train/test splits introduces catastrophic **lookahead leakage** (future fraudulent patterns predicting past transactions).

### Chronological Partitioning Rules
1. **Training & Validation Partitioning (`fraudTrain.csv`)**:
   - Total rows: 1,296,675.
   - Time span: `2019-01-01 00:00:18` to `2020-06-21 12:13:37`.
   - **Chronological 80/20 Cutoff**: `2020-03-06 07:16:43`.
   - Train Split: Rows 0 through 1,037,339 (`2019-01-01` to `2020-03-06`).
   - Validation Split: Rows 1,037,340 through 1,296,674 (`2020-03-06` to `2020-06-21`).
2. **Out-of-Time Test Set (`fraudTest.csv`)**:
   - Total rows: 555,719.
   - Time span: `2020-06-21 12:14:25` to `2020-12-31 23:59:34`.
   - **Zero Temporal Overlap**: The test set begins 48 seconds after the training/validation dataset terminates. It is held completely untouched until final benchmark evaluation.
3. **Transformation & Encoding Discipline**:
   - All standardizers, ordinal encoders, categorical frequency maps, and target encodings are fitted **exclusively on the training split**.
   - Transformers execute `.transform()` on validation and out-of-time test sets.
4. **Historical Rolling Window Discipline**:
   - Entity-level features (e.g. 24-hour transaction velocity, rolling spend sums) strictly enforce $t_{event} < t_{current}$, prohibiting current transaction inclusion in historical statistics.

---

## 4. Track B Out-of-Time Target Contamination Guards

Track B formulates a 4-class cardholder risk profile classification problem. To ensure empirical validity in production deployment:
- **Feature Observation Window ($W_{feat}$)**: strictly spans $[T_0, T_{split}]$.
- **Target Observation Window ($W_{target}$)**: strictly spans $[T_{split}, T_{end}]$.
- **Mathematical Invariant**:
  $$\max(t \in W_{feat}) < \min(t \in W_{target})$$
- Target metrics derived from future fraud frequency or chargeback claims cannot be accessed by feature engineering functions under any circumstance.

---

## 5. Adversarial Threat Model & Robustness Suite

Financial risk ML systems operate in an active adversarial environment where criminal syndicates adapt techniques to bypass algorithmic detection. The built-in `AdversarialRobustnessSuite` validates model stability against four core attack vectors:

### Vector 1: Micro-Splitting (Structuring / Smurfing)
- **Threat**: A fraudster possessing stolen card credentials splits an intended $800 transfer into four $200 transactions spaced 60 seconds apart to evade single-transaction dollar thresholds.
- **Defense & Invariant**: Rolling velocity features (e.g., transaction count in last 5 minutes) escalate exponentially, triggering cost-sensitive alert thresholds regardless of fragmented dollar amounts.

### Vector 2: Spatial Coordinate Perturbation
- **Threat**: Fraudsters route transactions through proxy merchant terminals or spoof GPS signals to mimic domestic/local activity.
- **Evaluation**: Coordinates (`merch_lat`, `merch_long`) are perturbed with Gaussian jitter $\mathcal{N}(0, \sigma^2)$ where $\sigma = 0.02^\circ$ (~2.2 km).
- **Invariant**: The model's risk score predictions must maintain rank stability (Spearman $\rho > 0.90$) and mean absolute score drift must not exceed 0.05.

### Vector 3: Diurnal Timestamp Shifts & Cyclical Boundaries
- **Threat**: Exploitation of timezone differences or non-continuous cyclical hour representations (e.g. midnight discontinuity between hour 23 and hour 0).
- **Defense**: Time-of-day features are engineered using continuous cyclical harmonics:
  $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
- Discontinuity at 23:59 $\to$ 00:00 is mathematically zero.

### Vector 4: Data Pipeline Stress & Edge-Case NaNs
- **Threat**: Malformed JSON feeds, upstream network timeouts yielding null fields, zero amounts, negative refunds, or unseen merchant string tokens.
- **Defense**: Native LightGBM NaN handling combined with robust categorical imputation ensures zero pipeline crashes. Predictions remain bounded in $[0, 1]$.

---

## 6. Track A Cost-Loss Matrix Resilience

Financial decisioning uses the asymmetric cost matrix:

$$\text{Financial Loss} = TP \times (-\$5) + FP \times (-\$15) + \sum_{i \in FN} -(\text{amt}_i + \$25) + TN \times \$0$$

Under adversarial perturbations and stress testing, the dollar-loss threshold optimizer guarantees that the cost-optimal decision threshold $\tau^*$ delivers statistically significant net dollar savings over the naive "Approve All" baseline and the default uncalibrated 0.5 threshold.

---

## 7. Reporting a Vulnerability

If you discover a potential security flaw, PII leakage vector, or algorithmic exploit in this codebase:
1. Do **not** open a public issue.
2. Submit a confidential security report via encrypted email to the maintainers or repository administrators.
3. Include detailed steps to reproduce, sample payload or transaction vector, and expected vs observed behavior.
4. Maintainers commit to acknowledging reports within 48 hours and providing a remediation timeline within 5 business days.
