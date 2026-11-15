from pathlib import Path


def generate_model_card(
    task: str, model_name: str, metrics: dict, cost_matrix_info: dict, out_path: str | Path
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pr_auc = metrics.get("pr_auc", 0.0)
    roc_auc = metrics.get("roc_auc", 0.0)
    thresh = metrics.get("threshold", 0.5)
    rec = metrics.get("recall", 0.0)
    prec = metrics.get("precision", 0.0)
    tot_cost = metrics.get("total_cost", 0.0)
    cost_sav = metrics.get("cost_savings_pct", 0.0)
    brier = metrics.get("brier", 0.0)
    ratio_desc = cost_matrix_info.get("ratio_description", "Domain cost matrix")
    cost_fn = cost_matrix_info.get("cost_fn", 0.0)
    cost_fp = cost_matrix_info.get("cost_fp", 0.0)
    ds_name = (
        "kartik2112/fraud-detection"
        if task == "fraud"
        else "subhamjain/loan-prediction-based-on-customer-behavior"
    )
    val_strat = (
        "Strict chronological train/test split"
        if task == "fraud"
        else "Stratified holdout validation"
    )

    content = f"""# Model Card: {model_name} for {task.upper()} Risk Prediction

## 1. Model Overview
- **Model Architecture:** {model_name}
- **Task:** Binary Financial Risk Classification ({task})
- **Framework:** Python, scikit-learn, Gradient Boosting
- **Decision Strategy:** Cost-Sensitive Optimal Thresholding (Expected Loss Minimization)

## 2. Intended Use
- **Primary Use Case:** Real-time flagging of high-risk transactions or defaulting loan applicants.
- **Decision Criterion:** Threshold tuned to minimize operational financial loss ($FN \times cost_{{FN}} + FP \times cost_{{FP}}$) rather than naive 0.5 accuracy.
- **Out of Scope:** Automated punitive action without secondary compliance review.

## 3. Training & Validation Data
- **Dataset:** {ds_name}
- **Validation Protocol:** {val_strat}
- **Preprocessing:** Standard scaling and one-hot encoding strictly fit inside Pipeline folds.

## 4. Performance Summary
- **PR-AUC (Average Precision - Headline):** {pr_auc:.4f}
- **ROC-AUC:** {roc_auc:.4f}
- **Optimal Decision Threshold (p*):** {thresh:.2f}
- **Recall at Optimal Threshold:** {rec:.4f}
- **Precision at Optimal Threshold:** {prec:.4f}
- **Total Expected Financial Cost:** ${tot_cost:,.2f}
- **Cost Reduction vs Default (0.50) Threshold:** {cost_sav:.2f}%
- **Brier Score:** {brier:.4f}

## 5. Cost Matrix & Trade-offs
- **Cost Formulation:** {ratio_desc}
- **FN Cost:** ${cost_fn:,.2f} | **FP Cost:** ${cost_fp:,.2f}

## 6. Limitations & Governance
- **Subgroup Slices:** Monitored across demographic and transaction amount deciles.
- **Drift Monitoring:** Distributional shift in amount or temporal velocity requires quarterly recalibration.
- **Fairness Sanity Check:** Anonymized transactional identifiers prevent direct PII bias.
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
