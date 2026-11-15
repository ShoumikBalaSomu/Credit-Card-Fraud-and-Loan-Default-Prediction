import numpy as np
import pandas as pd


def make_synthetic_fraud_data(
    n_samples: int = 10000, fraud_ratio: float = 0.005, random_state: int = 42
) -> pd.DataFrame:
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legit amounts log-normal, fraud amounts higher on average
    amt_legit = np.random.lognormal(mean=3.5, sigma=1.0, size=n_legit)
    amt_fraud = np.random.lognormal(mean=5.5, sigma=1.2, size=n_fraud)

    categories = [
        "misc_net",
        "grocery_pos",
        "entertainment",
        "gas_transport",
        "shopping_net",
        "shopping_pos",
    ]
    cat_legit = np.random.choice(categories, size=n_legit)
    cat_fraud = np.random.choice(categories, size=n_fraud, p=[0.35, 0.05, 0.1, 0.1, 0.3, 0.1])

    dates = pd.date_range(start="2019-01-01", periods=n_samples, freq="10s")

    df_legit = pd.DataFrame(
        {
            "trans_date_trans_time": dates[:n_legit],
            "amt": amt_legit,
            "category": cat_legit,
            "city_pop": np.random.randint(500, 1000000, size=n_legit),
            "lat": np.random.uniform(30.0, 48.0, size=n_legit),
            "long": np.random.uniform(-120.0, -70.0, size=n_legit),
            "is_fraud": 0,
        }
    )

    df_fraud = pd.DataFrame(
        {
            "trans_date_trans_time": dates[n_legit:],
            "amt": amt_fraud,
            "category": cat_fraud,
            "city_pop": np.random.randint(500, 1000000, size=n_fraud),
            "lat": np.random.uniform(30.0, 48.0, size=n_fraud),
            "long": np.random.uniform(-120.0, -70.0, size=n_fraud),
            "is_fraud": 1,
        }
    )

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    return df


def make_synthetic_loan_data(
    n_samples: int = 10000, default_ratio: float = 0.12, random_state: int = 42
) -> pd.DataFrame:
    np.random.seed(random_state)
    income = np.random.randint(10000, 10000000, size=n_samples)
    age = np.random.randint(21, 75, size=n_samples)
    experience = np.random.randint(0, 25, size=n_samples)
    married = np.random.choice(["single", "married"], size=n_samples)
    house_own = np.random.choice(
        ["rented", "owned", "norent_noown"], size=n_samples, p=[0.85, 0.12, 0.03]
    )
    car_own = np.random.choice(["no", "yes"], size=n_samples, p=[0.7, 0.3])
    current_job_yrs = np.random.randint(0, 15, size=n_samples)
    current_house_yrs = np.random.randint(10, 15, size=n_samples)

    # Latent risk score
    z = (
        -1.5
        - 0.0000001 * income
        + 0.02 * (60 - age)
        - 0.05 * experience
        + 0.3 * (married == "single")
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    # Adjust to target default_ratio
    threshold = np.quantile(prob, 1.0 - default_ratio)
    risk_flag = (prob >= threshold).astype(int)

    df = pd.DataFrame(
        {
            "Id": np.arange(1, n_samples + 1),
            "Income": income,
            "Age": age,
            "Experience": experience,
            "Married/Single": married,
            "House_Ownership": house_own,
            "Car_Ownership": car_own,
            "CURRENT_JOB_YRS": current_job_yrs,
            "CURRENT_HOUSE_YRS": current_house_yrs,
            "Risk_Flag": risk_flag,
        }
    )
    return df
