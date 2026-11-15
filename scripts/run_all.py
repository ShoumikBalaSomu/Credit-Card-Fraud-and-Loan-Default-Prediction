import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.explain.case_studies import format_contrastive_explanation
from src.explain.shap_local import explain_instance_waterfall
from src.models.train import train_and_evaluate_task
from src.utils.artifacts import load_artifact
from src.utils.io import load_json
from src.utils.logging import setup_logger
from src.utils.seeding import set_seed

logger = setup_logger("run_all")


def run_case_studies():
    logger.info("Generating 5 local XAI Case Studies...")
    reports_dir = Path("reports")
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load trained models and test data
    # Fraud Task
    fraud_res = load_json("reports/results/fraud_results.json")
    fraud_model = load_artifact("reports/results/best_model_fraud.joblib")
    fraud_prep = load_artifact("reports/results/preprocessor_fraud.joblib")

    from src.data.load import load_fraud_raw, load_loan_raw
    from src.data.preprocess import preprocess_fraud_df, preprocess_loan_df
    from src.data.splits import stratified_loan_split

    _, df_fraud_test_raw = load_fraud_raw(max_train_rows=1000, max_test_rows=5000)
    df_fraud_test = preprocess_fraud_df(df_fraud_test_raw)
    X_fraud_proc = fraud_prep.transform(df_fraud_test)
    y_fraud_test = df_fraud_test["is_fraud"].to_numpy()
    p_fraud = fraud_model.predict_proba(X_fraud_proc)[:, 1]
    th_fraud = fraud_res["optimal_threshold"]

    cat_encoder = fraud_prep.named_transformers_["cat"]
    cat_cols = fraud_prep.transformers[1][2]
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    num_cols = fraud_prep.transformers[0][2]
    fraud_feature_names = list(num_cols) + cat_feature_names

    # Indices for Fraud cases:
    # (i) Correctly caught fraud (TP): y=1, proba >= th
    tp_fraud_idx = np.where((y_fraud_test == 1) & (p_fraud >= th_fraud))[0]
    # (ii) Missed fraud (FN): y=1, proba < th
    fn_fraud_idx = np.where((y_fraud_test == 1) & (p_fraud < th_fraud))[0]
    # (iii) False Alarm (FP): y=0, proba >= th
    fp_fraud_idx = np.where((y_fraud_test == 0) & (p_fraud >= th_fraud))[0]

    # Loan Task
    loan_res = load_json("reports/results/loan_results.json")
    loan_model = load_artifact("reports/results/best_model_loan.joblib")
    loan_prep = load_artifact("reports/results/preprocessor_loan.joblib")

    df_loan_raw = load_loan_raw(max_rows=10000)
    _, df_loan_test_raw = stratified_loan_split(df_loan_raw, test_size=0.2, random_state=42)
    df_loan_test = preprocess_loan_df(df_loan_test_raw)
    X_loan_proc = loan_prep.transform(df_loan_test)
    y_loan_test = df_loan_test["Risk_Flag"].to_numpy()
    p_loan = loan_model.predict_proba(X_loan_proc)[:, 1]
    th_loan = loan_res["optimal_threshold"]

    cat_enc_l = loan_prep.named_transformers_["cat"]
    cat_cols_l = loan_prep.transformers[1][2]
    loan_feature_names = (
        list(loan_prep.transformers[0][2]) + cat_enc_l.get_feature_names_out(cat_cols_l).tolist()
    )

    # (iv) Correctly caught default (TP): y=1, proba >= th
    tp_loan_idx = np.where((y_loan_test == 1) & (p_loan >= th_loan))[0]
    # (v) Approved but defaulted applicant (FN): y=1, proba < th
    fn_loan_idx = np.where((y_loan_test == 1) & (p_loan < th_loan))[0]

    case_docs = [
        "# Local Explainability Case Studies\n\nThis report documents 5 individual high-impact predictions using local SHAP attributions and human-readable contrastive justifications.\n"
    ]

    # Case 1: TP Fraud
    if len(tp_fraud_idx) > 0:
        idx = tp_fraud_idx[0]
        explain_instance_waterfall(
            fraud_model, X_fraud_proc[idx], fraud_feature_names, fig_dir / "case1_tp_fraud.png"
        )
        case_docs.append(
            format_contrastive_explanation(
                "fraud",
                "1. Correctly Caught Fraud (True Positive)",
                df_fraud_test.iloc[idx],
                p_fraud[idx],
                th_fraud,
            )
        )
        case_docs.append("![Case 1 SHAP Waterfall](figures/case1_tp_fraud.png)\n")

    # Case 2: FN Fraud
    if len(fn_fraud_idx) > 0:
        idx = fn_fraud_idx[0]
        explain_instance_waterfall(
            fraud_model, X_fraud_proc[idx], fraud_feature_names, fig_dir / "case2_fn_fraud.png"
        )
        case_docs.append(
            format_contrastive_explanation(
                "fraud",
                "2. Missed Fraud (False Negative)",
                df_fraud_test.iloc[idx],
                p_fraud[idx],
                th_fraud,
            )
        )
        case_docs.append("![Case 2 SHAP Waterfall](figures/case2_fn_fraud.png)\n")

    # Case 3: FP Fraud
    if len(fp_fraud_idx) > 0:
        idx = fp_fraud_idx[0]
        explain_instance_waterfall(
            fraud_model, X_fraud_proc[idx], fraud_feature_names, fig_dir / "case3_fp_fraud.png"
        )
        case_docs.append(
            format_contrastive_explanation(
                "fraud",
                "3. False Alarm / Legitimate Blocked (False Positive)",
                df_fraud_test.iloc[idx],
                p_fraud[idx],
                th_fraud,
            )
        )
        case_docs.append("![Case 3 SHAP Waterfall](figures/case3_fp_fraud.png)\n")

    # Case 4: TP Loan Default
    if len(tp_loan_idx) > 0:
        idx = tp_loan_idx[0]
        explain_instance_waterfall(
            loan_model, X_loan_proc[idx], loan_feature_names, fig_dir / "case4_tp_loan.png"
        )
        case_docs.append(
            format_contrastive_explanation(
                "loan",
                "4. Correctly Predicted Default (True Positive)",
                df_loan_test.iloc[idx],
                p_loan[idx],
                th_loan,
            )
        )
        case_docs.append("![Case 4 SHAP Waterfall](figures/case4_tp_loan.png)\n")

    # Case 5: FN Loan Default
    if len(fn_loan_idx) > 0:
        idx = fn_loan_idx[0]
        explain_instance_waterfall(
            loan_model, X_loan_proc[idx], loan_feature_names, fig_dir / "case5_fn_loan.png"
        )
        case_docs.append(
            format_contrastive_explanation(
                "loan",
                "5. Approved-but-Defaulted Borrower (False Negative)",
                df_loan_test.iloc[idx],
                p_loan[idx],
                th_loan,
            )
        )
        case_docs.append("![Case 5 SHAP Waterfall](figures/case5_fn_loan.png)\n")

    reports_dir.joinpath("case_studies.md").write_text("\n".join(case_docs))
    logger.info("Case studies written to reports/case_studies.md!")


def main():
    parser = argparse.ArgumentParser(description="Unified ML Pipeline Runner")
    parser.add_argument("--phase", choices=["all", "train", "explain", "report"], default="all")
    args = parser.parse_args()

    set_seed(42)

    if args.phase in ["all", "train"]:
        logger.info("Running Training Phase for Fraud...")
        train_and_evaluate_task("fraud")
        logger.info("Running Training Phase for Loan Default...")
        train_and_evaluate_task("loan")

    if args.phase in ["all", "explain"]:
        run_case_studies()

    logger.info("All requested pipeline phases executed successfully!")


if __name__ == "__main__":
    main()
