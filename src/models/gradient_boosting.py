import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


def get_xgboost(
    scale_pos_weight: float = 1.0, random_state: int = 42, **kwargs
) -> xgb.XGBClassifier:
    params = dict(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    params.update(kwargs)
    return xgb.XGBClassifier(**params)


def get_lightgbm(
    scale_pos_weight: float = 1.0, random_state: int = 42, **kwargs
) -> lgb.LGBMClassifier:
    params = dict(
        n_estimators=120,
        num_leaves=31,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    params.update(kwargs)
    return lgb.LGBMClassifier(**params)


def get_catboost(
    scale_pos_weight: float = 1.0, random_state: int = 42, **kwargs
) -> CatBoostClassifier:
    params = dict(
        iterations=120,
        depth=6,
        learning_rate=0.08,
        auto_class_weights="Balanced" if scale_pos_weight > 1.0 else None,
        random_seed=random_state,
        verbose=0,
    )
    params.update(kwargs)
    return CatBoostClassifier(**params)
