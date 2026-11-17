import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_fraud_raw(df: pd.DataFrame):
    """Mirror of clean_fraud_raw function in master notebook."""
    df_clean = df.copy()
    initial_rows = len(df_clean)

    # 1. Deduplication
    dups_count = int(df_clean.duplicated().sum())
    if dups_count > 0:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # 2. String Hygiene
    str_cols = [
        c
        for c in df_clean.columns
        if df_clean[c].dtype == "object" or df_clean[c].dtype == "string"
    ]
    for c in str_cols:
        df_clean[c] = df_clean[c].astype(str).str.strip().str.lower()

    # 3. Non-negativity constraint on monetary amounts & Winsorization
    if "amt" in df_clean.columns:
        df_clean["amt"] = pd.to_numeric(df_clean["amt"], errors="coerce")
        null_amt = int(df_clean["amt"].isna().sum())
        if null_amt > 0:
            imp = SimpleImputer(strategy="median")
            df_clean[["amt"]] = imp.fit_transform(df_clean[["amt"]])
        df_clean["amt"] = np.maximum(0.0, df_clean["amt"])
        amt_p999 = float(df_clean["amt"].quantile(0.999))
        outliers_amt = int((df_clean["amt"] > amt_p999).sum())
        df_clean["amt"] = np.clip(df_clean["amt"], a_min=0.0, a_max=amt_p999)
    else:
        null_amt, outliers_amt = 0, 0

    # 4. Spatial Coordinate Boundary Validation
    coord_cols = ["lat", "long", "merch_lat", "merch_long"]
    null_coords = 0
    for col in coord_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            null_coords += int(df_clean[col].isna().sum())
            if "lat" in col:
                df_clean[col] = np.clip(df_clean[col], -90.0, 90.0)
            elif "long" in col:
                df_clean[col] = np.clip(df_clean[col], -180.0, 180.0)

    if null_coords > 0:
        coord_imputer = SimpleImputer(strategy="median")
        df_clean[coord_cols] = coord_imputer.fit_transform(df_clean[coord_cols])

    audit = {
        "Initial Rows": initial_rows,
        "Duplicates Removed": dups_count,
        "Null Monetary Imputed": null_amt,
        "Extreme Amount Outliers Capped": outliers_amt,
        "Null Coordinates Imputed": null_coords,
        "Final Clean Rows": len(df_clean),
    }
    return df_clean, audit


def clean_loan_raw(df: pd.DataFrame):
    """Mirror of clean_loan_raw function in master notebook."""
    df_clean = df.copy()
    initial_rows = len(df_clean)

    # 1. Deduplication
    dups_count = int(df_clean.duplicated().sum())
    if dups_count > 0:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # 2. String Normalization
    for col in ["Married/Single", "House_Ownership", "Car_Ownership"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()

    # 3. Numerical Imputation & Constraint Enforcement
    num_cols = ["Income", "Age", "Experience", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS"]
    existing_num = [c for c in num_cols if c in df_clean.columns]
    for c in existing_num:
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")

    null_num = int(df_clean[existing_num].isna().sum().sum())
    if null_num > 0:
        imp = SimpleImputer(strategy="median")
        df_clean[existing_num] = imp.fit_transform(df_clean[existing_num])

    # 4. Outlier Winsorization on Income (99th percentile)
    if "Income" in df_clean.columns:
        inc_p99 = float(df_clean["Income"].quantile(0.99))
        outliers_inc = int((df_clean["Income"] > inc_p99).sum())
        df_clean["Income"] = np.clip(df_clean["Income"], a_min=10000.0, a_max=inc_p99)
    else:
        outliers_inc = 0

    # 5. Logical Consistency: Job years cannot exceed Age - 16
    if "CURRENT_JOB_YRS" in df_clean.columns and "Age" in df_clean.columns:
        max_possible_job_yrs = np.maximum(0, df_clean["Age"] - 16)
        inconsistent_jobs = int((df_clean["CURRENT_JOB_YRS"] > max_possible_job_yrs).sum())
        df_clean["CURRENT_JOB_YRS"] = np.minimum(df_clean["CURRENT_JOB_YRS"], max_possible_job_yrs)
    else:
        inconsistent_jobs = 0

    audit = {
        "Initial Rows": initial_rows,
        "Duplicates Removed": dups_count,
        "Null Numeric Imputed": null_num,
        "Income Outliers Capped": outliers_inc,
        "Inconsistent Job Years Adjusted": inconsistent_jobs,
        "Final Clean Rows": len(df_clean),
    }
    return df_clean, audit


def test_clean_fraud_raw_deduplication_and_imputation():
    raw_data = pd.DataFrame(
        {
            "amt": [10.0, -5.0, np.nan, 999999.0, 10.0],
            "category": [" grocery_pos ", "misc_net", np.nan, "shopping_pos", " grocery_pos "],
            "lat": [40.0, -120.0, np.nan, 35.0, 40.0],
            "long": [-74.0, 200.0, np.nan, -80.0, -74.0],
            "merch_lat": [40.1, 41.0, np.nan, 35.1, 40.1],
            "merch_long": [-74.1, -75.0, np.nan, -80.1, -74.1],
        }
    )

    cleaned, audit = clean_fraud_raw(raw_data)
    # Row 4 is duplicate of Row 0
    assert audit["Duplicates Removed"] == 1
    assert len(cleaned) == 4
    # No NaNs remain
    assert cleaned["amt"].isna().sum() == 0
    assert cleaned["lat"].isna().sum() == 0
    # Negative amount sanitized
    assert (cleaned["amt"] >= 0).all()
    # Coordinates bounded
    assert (cleaned["lat"] >= -90.0).all() and (cleaned["lat"] <= 90.0).all()
    assert (cleaned["long"] >= -180.0).all() and (cleaned["long"] <= 180.0).all()
    # String whitespace stripped and lowercased
    assert cleaned["category"].iloc[0] == "grocery_pos"


def test_clean_loan_raw_constraints_and_winsorization():
    raw_data = pd.DataFrame(
        {
            "Income": [50000.0, -100.0, np.nan, 50000000.0, 50000.0],
            "Age": [25, 30, 45, 50, 25],
            "Experience": [3, 5, np.nan, 15, 3],
            "Married/Single": [" Single ", "married", "single", "married", " Single "],
            "House_Ownership": ["RENTED", "owned", "rented", "norent_noown", "RENTED"],
            "Car_Ownership": ["no", "YES", "no", "yes", "no"],
            "CURRENT_JOB_YRS": [20, 2, 5, 10, 20],  # 20 > (25 - 16 = 9) -> inconsistent!
            "CURRENT_HOUSE_YRS": [12, 10, 14, 11, 12],
        }
    )

    cleaned, audit = clean_loan_raw(raw_data)
    assert audit["Duplicates Removed"] == 1
    assert len(cleaned) == 4
    assert cleaned["Income"].isna().sum() == 0
    assert (cleaned["Income"] >= 10000.0).all()
    # Inconsistent job years adjusted
    assert cleaned.loc[0, "CURRENT_JOB_YRS"] <= (cleaned.loc[0, "Age"] - 16)
    # Categoricals normalized
    assert cleaned["Married/Single"].iloc[0] == "single"
    assert cleaned["House_Ownership"].iloc[0] == "rented"
    assert cleaned["Car_Ownership"].iloc[1] == "yes"


def test_sklearn_pipeline_integration():
    """Verifies that Scikit-Learn Pipeline + ColumnTransformer + Estimators train and predict seamlessly."""
    X = pd.DataFrame(
        {
            "num1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            "num2": [10.0, 20.0, 30.0, np.nan, 50.0, 60.0, 70.0, 80.0],
            "cat1": ["a", "b", "a", np.nan, "b", "a", "b", "a"],
        }
    )
    y = np.array([0, 0, 0, 1, 1, 0, 1, 1])

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, ["num1", "num2"]),
            ("cat", cat_pipeline, ["cat1"]),
        ]
    )

    full_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )

    full_pipeline.fit(X, y)
    preds = full_pipeline.predict(X)
    probas = full_pipeline.predict_proba(X)

    assert len(preds) == len(y)
    assert probas.shape == (8, 2)
    assert np.allclose(probas.sum(axis=1), 1.0)
