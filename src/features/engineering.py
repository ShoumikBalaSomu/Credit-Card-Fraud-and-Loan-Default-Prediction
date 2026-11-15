from typing import List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_fraud_feature_pipeline() -> Tuple[ColumnTransformer, List[str]]:
    numeric_features = [
        "amt",
        "amt_log",
        "city_pop",
        "dist_to_merch_km",
        "trans_hour",
        "trans_dayofweek",
        "is_weekend",
        "age",
    ]
    categorical_features = ["category", "gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    all_feature_cols = numeric_features + categorical_features
    return preprocessor, all_feature_cols


def build_loan_feature_pipeline() -> Tuple[ColumnTransformer, List[str]]:
    numeric_features = [
        "Income",
        "income_log",
        "Age",
        "Experience",
        "CURRENT_JOB_YRS",
        "CURRENT_HOUSE_YRS",
        "exp_to_age_ratio",
        "job_to_house_ratio",
    ]
    categorical_features = ["Married/Single", "House_Ownership", "Car_Ownership"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    all_feature_cols = numeric_features + categorical_features
    return preprocessor, all_feature_cols
