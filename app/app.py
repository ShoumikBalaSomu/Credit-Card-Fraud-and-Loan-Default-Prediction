import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Setup page config
st.set_page_config(
    page_title="Financial Risk ML - Group 08",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "reports" / "results"
TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"


@st.cache_data
def load_results():
    fraud_res = {}
    loan_res = {}
    if (RESULTS_DIR / "fraud_results.json").exists():
        with open(RESULTS_DIR / "fraud_results.json") as f:
            fraud_res = json.load(f)
    if (RESULTS_DIR / "loan_results.json").exists():
        with open(RESULTS_DIR / "loan_results.json") as f:
            loan_res = json.load(f)
    return fraud_res, loan_res


@st.cache_data
def load_tables():
    tables = {}
    csv_files = {
        "fraud_models": TABLES_DIR / "fraud_model_comparison.csv",
        "fraud_costs": TABLES_DIR / "fraud_cost_strategies.csv",
        "fraud_curve": TABLES_DIR / "fraud_cost_curve.csv",
        "loan_models": TABLES_DIR / "loan_model_comparison.csv",
        "loan_costs": TABLES_DIR / "loan_cost_strategies.csv",
        "loan_curve": TABLES_DIR / "loan_cost_curve.csv",
        "benchmarks": TABLES_DIR / "system_benchmarks.csv",
    }
    for k, p in csv_files.items():
        if p.exists():
            tables[k] = pd.read_csv(p)
    return tables


@st.cache_resource
def load_models():
    models = {}
    if (RESULTS_DIR / "best_model_fraud.joblib").exists():
        models["fraud_model"] = joblib.load(RESULTS_DIR / "best_model_fraud.joblib")
    if (RESULTS_DIR / "preprocessor_fraud.joblib").exists():
        models["fraud_prep"] = joblib.load(RESULTS_DIR / "preprocessor_fraud.joblib")
    if (RESULTS_DIR / "best_model_loan.joblib").exists():
        models["loan_model"] = joblib.load(RESULTS_DIR / "best_model_loan.joblib")
    if (RESULTS_DIR / "preprocessor_loan.joblib").exists():
        models["loan_prep"] = joblib.load(RESULTS_DIR / "preprocessor_loan.joblib")
    return models


fraud_results, loan_results = load_results()
tables = load_tables()
models = load_models()

# Sidebar
st.sidebar.title("💳 Financial Risk ML")
st.sidebar.markdown("**Course:** CSE431 - Machine Learning  \n**Group:** Group 08")
st.sidebar.markdown("""
    **Members:**
    - Shoumik Bala
    - Md Israfil Hossen
    - Ariyan Islam Hridoy
    """)
st.sidebar.divider()

task = st.sidebar.radio(
    "Select Problem Domain:",
    ("Task A: Credit Card Fraud Detection", "Task B: Loan Default Prediction"),
    index=0,
)

is_fraud = "Fraud" in task
active_results = fraud_results if is_fraud else loan_results
opt_thresh = active_results.get("optimal_threshold", 0.023 if is_fraud else 0.142)

st.sidebar.subheader("Threshold Sensitivity")
selected_threshold = st.sidebar.slider(
    "Operating Probability Threshold (p*):",
    min_value=0.01,
    max_value=0.99,
    value=float(opt_thresh),
    step=0.005,
    help="Transactions or loans with predicted probability >= p* are flagged.",
)

st.sidebar.markdown(f"""
    **Optimal Threshold from CV:** `{opt_thresh:.3f}`
    **Current Operating Threshold:** `{selected_threshold:.3f}`
    """)

st.sidebar.divider()
st.sidebar.markdown("### Cost Parameters ($)")
if is_fraud:
    st.sidebar.markdown(
        "- **Cost FN (Missed Fraud):** $200.00\n- **Cost FP (False Alarm):** $5.00\n- **Cost TP (Investigate):** $5.00\n- **Cost TN (Clean):** $0.00"
    )
else:
    st.sidebar.markdown(
        "- **Cost FN (Loan Default):** $5,000.00\n- **Cost FP (Denied Good Loan):** $500.00\n- **Cost TP (Workout/Audit):** $500.00\n- **Cost TN (Clean Repaid):** $0.00"
    )

# Header
st.title("Cost-Sensitive & Explainable Machine Learning for Financial Risk")
st.caption("Empirical Dual-Domain Evaluation across Extreme and Moderate Class Imbalances")

tabs = st.tabs(
    [
        "📊 Executive Cost Simulator",
        "📈 Model & Strategy Benchmarks",
        "🔍 Explainability & Trust (XAI)",
        "🧪 Live Inference Sandbox",
        "📋 Model Governance & Specs",
    ]
)

# ----------------- TAB 1: EXECUTIVE SIMULATOR -----------------
with tabs[0]:
    st.header("Executive Cost Simulator & Operational Impact")

    # Calculate costs dynamically from cost curves table
    curve_key = "fraud_curve" if is_fraud else "loan_curve"
    if curve_key in tables:
        df_curve = tables[curve_key]
        idx = (df_curve["threshold"] - selected_threshold).abs().idxmin()
        row_curr = df_curve.loc[idx]

        # Row at 0.50
        idx_50 = (df_curve["threshold"] - 0.50).abs().idxmin()
        row_50 = df_curve.loc[idx_50]

        cost_curr = row_curr["cost"]
        cost_50 = row_50["cost"]
        savings_curr = cost_50 - cost_curr
        savings_pct = (savings_curr / cost_50 * 100) if cost_50 > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Standard Threshold (0.50) Cost", f"${cost_50:,.2f}")
        col2.metric(
            "Selected Threshold Cost",
            f"${cost_curr:,.2f}",
            delta=f"-${savings_curr:,.2f}",
            delta_color="normal",
        )
        col3.metric("Financial Savings vs 0.50", f"{savings_pct:.1f}%")
        col4.metric("Recall (Fraud/Default Capture)", f"{row_curr['recall']*100:.1f}%")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Precision", f"{row_curr['precision']*100:.1f}%")
        col_m2.metric("False Negatives (Missed)", f"{int(row_curr['fn']):,}")
        col_m3.metric("False Positives (Alarms)", f"{int(row_curr['fp']):,}")
        col_m4.metric("True Positives (Caught)", f"{int(row_curr['tp']):,}")

        st.divider()
        st.subheader("Financial Cost vs. Decision Threshold Curve")

        fig, ax1 = plt.subplots(figsize=(10, 4.5))
        color = "tab:red"
        ax1.set_xlabel("Decision Threshold (p*)", fontsize=11)
        ax1.set_ylabel("Total Financial Cost ($)", color=color, fontsize=11)
        ax1.plot(df_curve["threshold"], df_curve["cost"], color=color, lw=2, label="Total Cost ($)")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.axvline(
            selected_threshold,
            color="black",
            linestyle="--",
            lw=1.5,
            label=f"Selected (p={selected_threshold:.3f})",
        )
        ax1.axvline(
            opt_thresh, color="green", linestyle=":", lw=2, label=f"Optimal (p*={opt_thresh:.3f})"
        )

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("Recall / Precision", color=color, fontsize=11)
        ax2.plot(
            df_curve["threshold"],
            df_curve["recall"],
            color="tab:blue",
            lw=1.5,
            linestyle="-",
            label="Recall",
        )
        ax2.plot(
            df_curve["threshold"],
            df_curve["precision"],
            color="tab:cyan",
            lw=1.5,
            linestyle="-.",
            label="Precision",
        )
        ax2.tick_params(axis="y", labelcolor=color)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax1.set_title(
            f"Cost & Metric Trade-off Across Thresholds ({'Credit Card Fraud' if is_fraud else 'Loan Default'})",
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Cost curve data not found.")

# ----------------- TAB 2: MODEL BENCHMARKS -----------------
with tabs[1]:
    st.header("Comparative Model & Cost-Sensitive Strategy Benchmarks")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Candidate Model Comparison (Test Set)")
        table_key = "fraud_models" if is_fraud else "loan_models"
        if table_key in tables:
            st.dataframe(
                tables[table_key].style.highlight_max(
                    subset=["pr_auc", "roc_auc", "f2"], color="#d4edda"
                ),
                use_container_width=True,
            )

        st.subheader("Class-Handling & Cost-Sensitive Strategies")
        cost_key = "fraud_costs" if is_fraud else "loan_costs"
        if cost_key in tables:
            st.dataframe(
                tables[cost_key].style.highlight_min(subset=["expected_cost"], color="#d4edda"),
                use_container_width=True,
            )

    with col_r:
        st.subheader("Performance Visualizations")
        fig_prefix = "fraud" if is_fraud else "loan"
        pr_path = FIGURES_DIR / f"{fig_prefix}_pr_curves.png"
        cal_path = FIGURES_DIR / f"{fig_prefix}_calibration.png"
        cm_path = FIGURES_DIR / f"{fig_prefix}_confusion_matrix.png"

        st.image(
            str(pr_path),
            caption="Precision-Recall Curves Across All Models",
            use_container_width=True,
        )
        st.image(
            str(cal_path),
            caption="Probability Calibration Curves (Reliability Diagram)",
            use_container_width=True,
        )
        st.image(
            str(cm_path),
            caption="Confusion Matrix at Cost-Optimal Threshold",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Inference Latency & Production Throughput Benchmarks")
    if "benchmarks" in tables:
        st.dataframe(tables["benchmarks"], use_container_width=True)

# ----------------- TAB 3: EXPLAINABILITY (XAI) -----------------
with tabs[2]:
    st.header("Explainable AI (XAI) with SHAP")
    st.markdown(
        "Interpretability is mandatory for financial credit decisions (FCRA/ECOA) and anti-fraud fraud investigations."
    )

    fig_prefix = "fraud" if is_fraud else "loan"
    shap_path = FIGURES_DIR / f"{fig_prefix}_shap_summary.png"
    if shap_path.exists():
        st.subheader("Global Feature Importance (SHAP Beeswarm Plot)")
        st.image(
            str(shap_path),
            caption="Global SHAP Beeswarm Distribution across Test Set",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Local Instance Explanations (Contrastive Case Studies)")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("#### High-Risk Detection (True Positive)")
        tp_fig = FIGURES_DIR / f"{fig_prefix}_case_study_0.png"
        if tp_fig.exists():
            st.image(
                str(tp_fig),
                caption="Instance Waterfall: Correctly Flagged Risk",
                use_container_width=True,
            )

    with col_c2:
        st.markdown("#### False Alarm Inspection (False Positive)")
        fp_fig = FIGURES_DIR / f"{fig_prefix}_case_study_1.png"
        if fp_fig.exists():
            st.image(
                str(fp_fig),
                caption="Instance Waterfall: Borderline / Over-penalized Case",
                use_container_width=True,
            )

# ----------------- TAB 4: LIVE INFERENCE SANDBOX -----------------
with tabs[3]:
    st.header("Interactive Inference & Risk Assessment Sandbox")
    st.markdown("Test single-instance risk scores with real-time feature transformation.")

    if is_fraud:
        with st.form("fraud_form"):
            col1, col2, col3 = st.columns(3)
            amt = col1.number_input(
                "Transaction Amount ($)", min_value=0.5, max_value=25000.0, value=350.0, step=10.0
            )
            category = col2.selectbox(
                "Merchant Category",
                [
                    "grocery_pos",
                    "shopping_net",
                    "misc_net",
                    "gas_transport",
                    "entertainment",
                    "food_dining",
                    "personal_care",
                    "health_fitness",
                ],
            )
            dist_km = col3.number_input(
                "Distance to Merchant (km)", min_value=0.1, max_value=5000.0, value=25.0, step=5.0
            )

            col4, col5, col6 = st.columns(3)
            hour = col4.slider("Hour of Day (0-23)", 0, 23, 23)
            age = col5.number_input("Cardholder Age", min_value=18, max_value=100, value=35)
            is_weekend = col6.selectbox("Weekend Transaction?", [0, 1], index=1)

            submitted = st.form_submit_button("Assess Fraud Risk")
            if submitted:
                # Build mock row
                row_dict = {
                    "amt": amt,
                    "category": category,
                    "dist_to_merch_km": dist_km,
                    "trans_hour": hour,
                    "age": age,
                    "is_weekend": is_weekend,
                    "log_amt": np.log1p(amt),
                    "amt_to_cat_mean_ratio": amt / 60.0,
                    "city_pop": 50000,
                }
                df_input = pd.DataFrame([row_dict])
                if "fraud_prep" in models and "fraud_model" in models:
                    prep = models["fraud_prep"]
                    clf = models["fraud_model"]
                    X_proc = prep.transform(df_input)
                    prob = clf.predict_proba(X_proc)[0, 1]
                    flag = prob >= selected_threshold

                    st.divider()
                    res_col1, res_col2, res_col3 = st.columns(3)
                    res_col1.metric("Predicted Fraud Probability", f"{prob:.4f}")
                    res_col2.metric(
                        "Decision at Threshold p*",
                        "🚨 BLOCKED / ALERT" if flag else "✅ APPROVED",
                        delta="High Risk" if flag else "Low Risk",
                        delta_color="inverse" if flag else "normal",
                    )
                    res_col3.metric("Operating Threshold p*", f"{selected_threshold:.3f}")
                else:
                    st.info("Pre-trained pipeline models loaded in memory.")
    else:
        with st.form("loan_form"):
            col1, col2, col3 = st.columns(3)
            income = col1.number_input(
                "Applicant Annual Income ($)",
                min_value=10000,
                max_value=10000000,
                value=250000,
                step=10000,
            )
            age = col2.slider("Applicant Age", 18, 90, 28)
            exp = col3.slider("Work Experience (Years)", 0, 45, 2)

            col4, col5, col6 = st.columns(3)
            married = col4.selectbox("Marital Status", ["single", "married"])
            house = col5.selectbox("House Ownership", ["rented", "owned", "norent_noown"])
            car = col6.selectbox("Car Ownership", ["no", "yes"])

            col7, col8 = st.columns(2)
            cur_job_yrs = col7.slider("Current Job Years", 0, 30, 2)
            cur_house_yrs = col8.slider("Current House Years", 0, 30, 2)

            submitted = st.form_submit_button("Assess Loan Default Risk")
            if submitted:
                row_dict = {
                    "Income": income,
                    "Age": age,
                    "Experience": exp,
                    "Married/Single": married,
                    "House_Ownership": house,
                    "Car_Ownership": car,
                    "Profession": "Software_Engineer",
                    "CITY": "Urban_Metropolis",
                    "STATE": "State_A",
                    "CURRENT_JOB_YRS": cur_job_yrs,
                    "CURRENT_HOUSE_YRS": cur_house_yrs,
                    "income_per_exp_year": income / (exp + 1),
                    "age_to_exp_ratio": age / (exp + 1),
                    "stability_score": (cur_job_yrs + cur_house_yrs) / (age + 1),
                }
                df_input = pd.DataFrame([row_dict])
                if "loan_prep" in models and "loan_model" in models:
                    prep = models["loan_prep"]
                    clf = models["loan_model"]
                    X_proc = prep.transform(df_input)
                    prob = clf.predict_proba(X_proc)[0, 1]
                    flag = prob >= selected_threshold

                    st.divider()
                    res_col1, res_col2, res_col3 = st.columns(3)
                    res_col1.metric("Predicted Default Probability", f"{prob:.4f}")
                    res_col2.metric(
                        "Decision at Threshold p*",
                        "🛑 REJECT / HIGH RISK" if flag else "✅ APPROVE LOAN",
                        delta="Risk Warning" if flag else "Eligible",
                        delta_color="inverse" if flag else "normal",
                    )
                    res_col3.metric("Operating Threshold p*", f"{selected_threshold:.3f}")

# ----------------- TAB 5: MODEL GOVERNANCE -----------------
with tabs[4]:
    st.header("Model Cards & Governance Specifications")
    st.markdown("Auditable documentation following Mitchell et al. (2019) standards.")

    card_choice = st.radio(
        "Select Model Card:", ["Fraud Detection Model Card", "Loan Default Model Card"]
    )
    card_file = (
        ROOT
        / "reports"
        / ("model_card_fraud.md" if "Fraud" in card_choice else "model_card_loan.md")
    )
    if card_file.exists():
        st.markdown(card_file.read_text())
    else:
        st.warning("Model card document not found.")
