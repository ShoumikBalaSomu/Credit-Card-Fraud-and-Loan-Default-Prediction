import matplotlib

matplotlib.use("Agg")
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

from src.data.load import load_fraud_raw, load_loan_raw
from src.data.preprocess import preprocess_fraud_df, preprocess_loan_df
from src.data.splits import chronological_fraud_split, stratified_loan_split
from src.evaluate.calibration import compute_calibration_stats
from src.evaluate.cost import evaluate_cost_curve
from src.evaluate.metrics import compute_all_metrics
from src.evaluate.plots import (
    plot_calibration_curve,
    plot_confusion_matrix_heatmap,
    plot_cost_curve,
    plot_pr_curve,
)
from src.explain.model_cards import generate_model_card
from src.explain.shap_global import run_tree_shap_global
from src.features.engineering import build_fraud_feature_pipeline, build_loan_feature_pipeline
from src.models.baselines import get_logistic_regression, get_random_forest
from src.models.cost_sensitive import CostMatrix, ThresholdOptimizer
from src.models.gradient_boosting import get_catboost, get_lightgbm, get_xgboost
from src.utils.artifacts import save_artifact
from src.utils.io import load_yaml, save_json
from src.utils.logging import setup_logger
from src.utils.seeding import set_seed

logger = setup_logger("train")


def train_and_evaluate_task(task: str = "fraud", random_seed: int = 42) -> Dict[str, Any]:
    set_seed(random_seed)
    logger.info(f"================ STARTING EXPERIMENT FOR TASK: {task.upper()} ================")

    # 1. Load Configurations
    cost_cfg = load_yaml("configs/cost_matrices.yaml")[task]
    data_cfg = load_yaml("configs/data.yaml")[task]
    cost_matrix = CostMatrix(
        cost_fn=cost_cfg["cost_fn"],
        cost_fp=cost_cfg["cost_fp"],
        cost_tp=cost_cfg.get("cost_tp", 0.0),
        cost_tn=cost_cfg.get("cost_tn", 0.0),
    )

    # 2. Data Loading & Preprocessing
    if task == "fraud":
        df_train_raw, df_test_raw = load_fraud_raw(
            max_train_rows=data_cfg.get("sample_size_train", 100000),
            max_test_rows=data_cfg.get("sample_size_test", 40000),
        )
        df_train_raw, df_test_raw = chronological_fraud_split(df_train_raw, df_test_raw)
        df_train = preprocess_fraud_df(df_train_raw)
        df_test = preprocess_fraud_df(df_test_raw)
        preprocessor, feature_names_base = build_fraud_feature_pipeline()
        target_col = data_cfg["target_column"]
    else:
        df_raw = load_loan_raw(max_rows=data_cfg.get("sample_size", 60000))
        df_train_raw, df_test_raw = stratified_loan_split(
            df_raw,
            target_col=data_cfg["target_column"],
            test_size=data_cfg.get("test_size", 0.20),
            random_state=random_seed,
        )
        df_train = preprocess_loan_df(df_train_raw)
        df_test = preprocess_loan_df(df_test_raw)
        preprocessor, feature_names_base = build_loan_feature_pipeline()
        target_col = data_cfg["target_column"]

    y_train = df_train[target_col].to_numpy().astype(int)
    y_test = df_test[target_col].to_numpy().astype(int)

    # Strict Pipeline Fit: Transform X_train, then transform X_test
    logger.info("Fitting preprocessor strictly on training fold...")
    X_train_proc = preprocessor.fit_transform(df_train)
    X_test_proc = preprocessor.transform(df_test)

    # Get transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_cols = preprocessor.transformers[1][2]
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    num_cols = preprocessor.transformers[0][2]
    all_feature_names = list(num_cols) + cat_feature_names

    logger.info(
        f"Dataset processed: Train shape {X_train_proc.shape}, Test shape {X_test_proc.shape}"
    )
    pos_rate_train = float(np.mean(y_train))
    pos_rate_test = float(np.mean(y_test))
    logger.info(
        f"Train positives: {np.sum(y_train)} ({pos_rate_train*100:.3f}%), Test positives: {np.sum(y_test)} ({pos_rate_test*100:.3f}%)"
    )

    # 3. Model Zoo Execution
    scale_pos_weight = float((len(y_train) - np.sum(y_train)) / max(1, np.sum(y_train)))

    models = {
        "Logistic Regression": get_logistic_regression(random_state=random_seed),
        "Random Forest": get_random_forest(random_state=random_seed),
        "XGBoost": get_xgboost(random_state=random_seed),
        "LightGBM": get_lightgbm(random_state=random_seed),
        "CatBoost": get_catboost(random_state=random_seed),
    }

    model_eval_results = {}
    best_pr_auc = -1.0
    best_model_name = None
    best_model_obj = None
    best_y_proba = None

    for name, model in models.items():
        t0 = time.time()
        logger.info(f"Training {name}...")
        model.fit(X_train_proc, y_train)
        fit_time = time.time() - t0

        y_proba = model.predict_proba(X_test_proc)[:, 1]

        # Optimize threshold on validation / train or compute standard
        opt = ThresholdOptimizer(cost_matrix=cost_matrix, n_thresholds=150)
        # We find threshold on training fold predictions to avoid test leakage
        y_train_proba = model.predict_proba(X_train_proc)[:, 1]
        opt.fit(y_train, y_train_proba)
        optimal_th = opt.optimal_threshold_

        metrics = compute_all_metrics(
            y_test, y_proba, threshold=optimal_th, cost_matrix=cost_matrix
        )
        metrics["fit_time_sec"] = float(fit_time)
        model_eval_results[name] = metrics

        logger.info(
            f'[{name}] PR-AUC: {metrics["pr_auc"]:.4f} | ROC-AUC: {metrics["roc_auc"]:.4f} | Cost: ${metrics["total_cost"]:,.2f} (Savings: {metrics["cost_savings_pct"]:.1f}%) | Optimal p*: {optimal_th:.2f}'
        )

        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            best_model_name = name
            best_model_obj = model
            best_y_proba = y_proba

    # 4. Compare Cost Handling Strategies on the Best Model Architecture
    logger.info(
        f"Comparing 5 cost-handling strategies on best model architecture ({best_model_name})..."
    )
    strategy_results = {}

    # Strategy 1: Baseline Unweighted (p = 0.5)
    s1_metrics = compute_all_metrics(y_test, best_y_proba, threshold=0.5, cost_matrix=cost_matrix)
    strategy_results["Baseline (Unweighted, p=0.5)"] = s1_metrics

    # Strategy 2: Class-Weighted (scale_pos_weight or balanced)
    if best_model_name == "LightGBM":
        m_cw = get_lightgbm(scale_pos_weight=scale_pos_weight, random_state=random_seed)
    elif best_model_name == "XGBoost":
        m_cw = get_xgboost(scale_pos_weight=scale_pos_weight, random_state=random_seed)
    elif best_model_name == "CatBoost":
        m_cw = get_catboost(scale_pos_weight=scale_pos_weight, random_state=random_seed)
    elif best_model_name == "Random Forest":
        m_cw = get_random_forest(class_weight="balanced", random_state=random_seed)
    else:
        m_cw = get_logistic_regression(class_weight="balanced", random_state=random_seed)
    m_cw.fit(X_train_proc, y_train)
    y_proba_cw = m_cw.predict_proba(X_test_proc)[:, 1]
    strategy_results["Class-Weighted Model (p=0.5)"] = compute_all_metrics(
        y_test, y_proba_cw, threshold=0.5, cost_matrix=cost_matrix
    )

    # Strategy 3: Threshold Optimization on Expected Cost
    opt_best = ThresholdOptimizer(cost_matrix=cost_matrix, n_thresholds=150)
    y_train_proba_best = best_model_obj.predict_proba(X_train_proc)[:, 1]
    opt_best.fit(y_train, y_train_proba_best)
    p_star = opt_best.optimal_threshold_
    strategy_results["Threshold Optimization (p*)"] = compute_all_metrics(
        y_test, best_y_proba, threshold=p_star, cost_matrix=cost_matrix
    )

    # Strategy 4: Resampling (SMOTE applied strictly on training set)
    logger.info("Applying SMOTE strictly on training fold...")
    sm = SMOTE(random_state=random_seed, sampling_strategy=0.1 if task == "fraud" else 0.3)
    X_train_sm, y_train_sm = sm.fit_resample(X_train_proc, y_train)
    if best_model_name == "LightGBM":
        m_sm = get_lightgbm(random_state=random_seed)
    elif best_model_name == "XGBoost":
        m_sm = get_xgboost(random_state=random_seed)
    elif best_model_name == "CatBoost":
        m_sm = get_catboost(random_state=random_seed)
    elif best_model_name == "Random Forest":
        m_sm = get_random_forest(random_state=random_seed)
    else:
        m_sm = get_logistic_regression(random_state=random_seed)
    m_sm.fit(X_train_sm, y_train_sm)
    y_proba_sm = m_sm.predict_proba(X_test_proc)[:, 1]
    strategy_results["SMOTE Resampled (p=0.5)"] = compute_all_metrics(
        y_test, y_proba_sm, threshold=0.5, cost_matrix=cost_matrix
    )

    # Strategy 5: Cost-Sensitive Sample Weights
    sample_weights = np.where(y_train == 1, cost_matrix.cost_fn / cost_matrix.cost_fp, 1.0)
    if best_model_name == "LightGBM":
        m_sw = get_lightgbm(random_state=random_seed)
        m_sw.fit(X_train_proc, y_train, sample_weight=sample_weights)
    elif best_model_name == "XGBoost":
        m_sw = get_xgboost(random_state=random_seed)
        m_sw.fit(X_train_proc, y_train, sample_weight=sample_weights)
    else:
        m_sw = get_random_forest(random_state=random_seed)
        m_sw.fit(X_train_proc, y_train, sample_weight=sample_weights)
    y_proba_sw = m_sw.predict_proba(X_test_proc)[:, 1]
    strategy_results["Cost-Weighted Loss (p=0.5)"] = compute_all_metrics(
        y_test, y_proba_sw, threshold=0.5, cost_matrix=cost_matrix
    )

    # 5. Repeated Cross-Validation (5-Fold Stratified) for Statistical Comparison
    logger.info("Running 5-fold cross-validation for statistical verification...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    cv_records = []

    # Run CV on a fast representative subset if needed
    sub_idx = np.random.choice(len(y_train), size=min(len(y_train), 30000), replace=False)
    X_sub = X_train_proc[sub_idx]
    y_sub = y_train[sub_idx]

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_sub, y_sub)):
        X_tr, y_tr = X_sub[train_idx], y_sub[train_idx]
        X_va, y_val = X_sub[val_idx], y_sub[val_idx]
        for m_name, m_func in [
            ("Logistic Regression", lambda: get_logistic_regression(random_state=random_seed)),
            ("Random Forest", lambda: get_random_forest(random_state=random_seed)),
            ("XGBoost", lambda: get_xgboost(random_state=random_seed)),
            ("LightGBM", lambda: get_lightgbm(random_state=random_seed)),
        ]:
            m = m_func()
            m.fit(X_tr, y_tr)
            preds_proba = m.predict_proba(X_va)[:, 1]
            m_metrics = compute_all_metrics(
                y_val, preds_proba, threshold=0.5, cost_matrix=cost_matrix
            )
            cv_records.append(
                {
                    "fold": fold,
                    "model": m_name,
                    "pr_auc": m_metrics["pr_auc"],
                    "roc_auc": m_metrics["roc_auc"],
                    "f1": m_metrics["f1"],
                    "total_cost": m_metrics["total_cost"],
                }
            )
    df_cv = pd.DataFrame(cv_records)
    cv_summary = df_cv.groupby("model")[["pr_auc", "roc_auc", "f1", "total_cost"]].agg(
        ["mean", "std"]
    )

    # 6. Cost Curves, Calibration & Plots
    logger.info("Generating cost curves, calibration plots, and artifacts...")
    df_cost_curve = evaluate_cost_curve(y_test, best_y_proba, cost_matrix=cost_matrix)
    calib_stats = compute_calibration_stats(y_test, best_y_proba)

    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_pr_curve(
        y_test,
        best_y_proba,
        f"{task.upper()} - PR Curve ({best_model_name})",
        fig_dir / f"{task}_pr_curve.png",
    )
    plot_cost_curve(
        df_cost_curve,
        p_star,
        f"{task.upper()} - Expected Cost Curve ({best_model_name})",
        fig_dir / f"{task}_cost_curve.png",
    )
    plot_calibration_curve(
        calib_stats["prob_true"],
        calib_stats["prob_pred"],
        calib_stats["ece"],
        f"{task.upper()} - Calibration Curve",
        fig_dir / f"{task}_calibration.png",
    )

    # Confusion matrix
    best_cm = np.array(
        [
            [
                strategy_results["Threshold Optimization (p*)"]["TN"],
                strategy_results["Threshold Optimization (p*)"]["FP"],
            ],
            [
                strategy_results["Threshold Optimization (p*)"]["FN"],
                strategy_results["Threshold Optimization (p*)"]["TP"],
            ],
        ]
    )
    plot_confusion_matrix_heatmap(
        best_cm,
        f"{task.upper()} - Confusion Matrix at p*={p_star:.2f}",
        fig_dir / f"{task}_confusion_matrix.png",
    )

    # 7. SHAP Global Analysis
    logger.info("Computing SHAP global importance on holdout sample...")
    sample_shap_size = min(500, len(X_test_proc))
    shap_sample_idx = np.random.choice(len(X_test_proc), size=sample_shap_size, replace=False)
    X_shap_sample = X_test_proc[shap_sample_idx]

    shap_importance = run_tree_shap_global(
        best_model_obj, X_shap_sample, all_feature_names, fig_dir / f"{task}_shap_summary.png"
    )

    # Save Model Cards
    generate_model_card(
        task=task,
        model_name=best_model_name,
        metrics=strategy_results["Threshold Optimization (p*)"],
        cost_matrix_info=cost_cfg,
        out_path=Path("reports") / f"model_card_{task}.md",
    )

    # Save best model artifact
    save_artifact(best_model_obj, Path("reports/results") / f"best_model_{task}.joblib")
    save_artifact(preprocessor, Path("reports/results") / f"preprocessor_{task}.joblib")

    # 8. Tables and JSON outputs
    df_models = pd.DataFrame(model_eval_results).T
    df_strategies = pd.DataFrame(strategy_results).T

    df_models.to_csv(f"reports/tables/{task}_model_comparison.csv")
    df_strategies.to_csv(f"reports/tables/{task}_cost_strategies.csv")
    df_cost_curve.to_csv(f"reports/tables/{task}_cost_curve.csv", index=False)

    cv_summary.to_csv(f"reports/tables/{task}_cv_summary.csv")

    results_payload = {
        "task": task,
        "best_model": best_model_name,
        "optimal_threshold": float(p_star),
        "models": model_eval_results,
        "cost_strategies": strategy_results,
        "calibration": calib_stats,
        "shap_top_features": shap_importance,
        "cost_matrix": cost_cfg,
    }
    save_json(results_payload, f"reports/results/{task}_results.json")

    logger.info(f"Task {task.upper()} completed successfully!")
    return results_payload
