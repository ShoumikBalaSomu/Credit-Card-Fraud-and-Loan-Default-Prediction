import numpy as np
import pandas as pd


def preprocess_fraud_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal, spatial, and financial features without future leakage."""
    df = df.copy()
    if "trans_date_trans_time" in df.columns:
        dt = pd.to_datetime(df["trans_date_trans_time"])
        df["trans_hour"] = dt.dt.hour
        # Pandas 4.0 compatibility
        df["trans_dayofweek"] = (
            dt.dt.day_of_week if hasattr(dt.dt, "day_of_week") else dt.dt.dayofweek
        )
        df["trans_month"] = dt.dt.month
        df["is_weekend"] = (df["trans_dayofweek"] >= 5).astype(int)
    else:
        df["trans_hour"] = 12
        df["trans_dayofweek"] = 2
        df["trans_month"] = 6
        df["is_weekend"] = 0

    if "dob" in df.columns:
        dob_dt = pd.to_datetime(df["dob"], errors="coerce")
        trans_yr = dt.dt.year if "trans_date_trans_time" in df.columns else 2020
        df["age"] = trans_yr - dob_dt.dt.year
        df["age"] = df["age"].fillna(40.0)
    else:
        df["age"] = 40.0

    # Distance between cardholder location and merchant location (Haversine approx in km)
    if all(col in df.columns for col in ["lat", "long", "merch_lat", "merch_long"]):
        lat1, lon1 = np.radians(df["lat"]), np.radians(df["long"])
        lat2, lon2 = np.radians(df["merch_lat"]), np.radians(df["merch_long"])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        df["dist_to_merch_km"] = 6371.0 * c
    else:
        df["dist_to_merch_km"] = 15.0

    # Amount features
    if "amt" in df.columns:
        df["amt_log"] = np.log1p(np.maximum(0, df["amt"]))
    else:
        df["amt"] = 50.0
        df["amt_log"] = np.log1p(50.0)

    if "category" not in df.columns:
        df["category"] = "misc_net"
    if "gender" not in df.columns:
        df["gender"] = "M"
    if "city_pop" not in df.columns:
        df["city_pop"] = 10000

    return df


def preprocess_loan_df(df: pd.DataFrame) -> pd.DataFrame:
    """Process demographic and credit history features for loan applicants."""
    df = df.copy()
    if "Income" in df.columns:
        df["income_log"] = np.log1p(np.maximum(0, df["Income"]))
    else:
        df["Income"] = 500000
        df["income_log"] = np.log1p(500000)

    if "Experience" in df.columns and "Age" in df.columns:
        df["exp_to_age_ratio"] = df["Experience"] / (df["Age"] + 1e-5)
    else:
        df["exp_to_age_ratio"] = 0.2

    if "CURRENT_JOB_YRS" in df.columns and "CURRENT_HOUSE_YRS" in df.columns:
        df["job_to_house_ratio"] = df["CURRENT_JOB_YRS"] / (df["CURRENT_HOUSE_YRS"] + 1e-5)
    else:
        df["job_to_house_ratio"] = 0.5

    return df
