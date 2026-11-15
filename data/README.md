# Dataset Documentation & Provenance

This directory manages the raw and processed data assets for the dual-task financial risk evaluation:
1. **Task A: Credit Card Fraud Detection** (Extreme Imbalance: 0.58% positive)
2. **Task B: Loan Default Prediction** (Moderate Imbalance: 12.30% positive)

---

## 1. Task A: Credit Card Fraud Dataset

- **Source:** Kaggle `kartik2112/fraud-detection`
- **Original Generator:** Sparkov Data Generation tool by Brandon Harris
- **License:** CC0: Public Domain
- **Files & Provenance:**
  - `fraudTrain.csv` (Train): 1,296,675 transactions spanning Jan 1, 2019 to Dec 31, 2019 (7,506 fraud, 0.579%).
  - `fraudTest.csv` (Test): 555,719 transactions spanning Jan 1, 2020 to Dec 31, 2020 (2,145 fraud, 0.386%).
- **Split Protocol:** Strict **chronological temporal separation**. Training on 2019 data and testing on 2020 data guarantees zero look-ahead bias and simulates real-world deployment against concept drift.
- **Key Features:**
  - `trans_date_trans_time`, `amt`, `category`, `lat`, `long`, `merch_lat`, `merch_long`, `dob`, `city_pop`.
  - Engineered: `dist_to_merch_km` (Haversine formula), `trans_hour`, `age`, `log_amt`, `amt_to_cat_mean_ratio`.

---

## 2. Task B: Loan Default Risk Dataset

- **Source:** Kaggle `subhamjain/loan-prediction-based-on-customer-behavior`
- **License:** CC0: Public Domain
- **Files & Provenance:**
  - `Training Data.csv`: 252,000 applicant records (30,996 defaults, 12.30% positive rate).
- **Split Protocol:** Stratified Shuffle Split (80% Train, 20% Holdout Test with random seed `42`).
- **Target Variable:** `Risk_Flag` (1 = Loan Defaulted, 0 = Loan Repaid).
- **Key Features:**
  - `Income`, `Age`, `Experience`, `Married/Single`, `House_Ownership`, `Car_Ownership`, `Profession`, `CITY`, `STATE`, `CURRENT_JOB_YRS`, `CURRENT_HOUSE_YRS`.
  - Engineered: `income_per_exp_year`, `age_to_exp_ratio`, `stability_score`.

---

## 3. How to Download & Regenerate Datasets

To download datasets automatically via `kagglehub`:
```bash
python scripts/run_all.py --download-only
```
Or via Kaggle CLI:
```bash
kaggle datasets download -d kartik2112/fraud-detection -p data/raw/
kaggle datasets download -d subhamjain/loan-prediction-based-on-customer-behavior -p data/raw/
unzip -q data/raw/fraud-detection.zip -d data/raw/
unzip -q data/raw/loan-prediction-based-on-customer-behavior.zip -d data/raw/
```

*Note: In automated testing and CI environments, lightweight synthetic fixtures in `tests/fixtures/` allow all tests and validation suites to execute offline in < 2 seconds without external network dependencies.*
