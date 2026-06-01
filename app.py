"""
Loan Default Prediction - Streamlit app.

Tab 1 — Predict: enter applicant details, get default-risk probability.
Tab 2 — Model Performance: ROC curves, confusion matrix, comparison table.

Run:  streamlit run app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st
from PIL import Image

MODEL_PATH   = os.path.join("models", "credit_risk_model.joblib")
META_PATH    = os.path.join("models", "feature_metadata.joblib")
ROC_PATH     = os.path.join("reports", "roc_curves.png")
CM_PATH      = os.path.join("reports", "confusion_matrix.png")
METRICS_PATH = os.path.join("reports", "model_comparison.csv")

st.set_page_config(page_title="CreditGuard – Loan Default Predictor",
                   page_icon="💳", layout="wide")

LABELS = {
    "duration":               "Loan duration (months)",
    "amount":                 "Credit amount (DM)",
    "installment_rate":       "Installment rate (% of income)",
    "present_residence":      "Years at current residence",
    "age":                    "Age (years)",
    "number_credits":         "Existing credits at this bank",
    "people_liable":          "Dependents",
    "status":                 "Checking account status",
    "credit_history":         "Credit history",
    "purpose":                "Loan purpose",
    "savings":                "Savings account",
    "employment_duration":    "Employment duration",
    "personal_status_sex":    "Personal status & sex",
    "other_debtors":          "Other debtors / guarantors",
    "property":               "Property owned",
    "other_installment_plans":"Other installment plans",
    "housing":                "Housing",
    "job":                    "Job type",
    "telephone":              "Registered telephone",
    "foreign_worker":         "Foreign worker",
}


@st.cache_resource
def load_artifacts():
    def _train():
        import train
        with st.spinner("Training model (first run only, ~15s)…"):
            train.main()

    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        _train()
    try:
        return joblib.load(MODEL_PATH), joblib.load(META_PATH)
    except Exception:
        _train()
        return joblib.load(MODEL_PATH), joblib.load(META_PATH)


def predict_tab(model, meta):
    st.subheader("Applicant details")
    st.caption("Fill in the applicant's information and click **Predict**.")

    inputs = {}
    num_cols = st.columns(4)
    for i, c in enumerate(meta["numeric"]):
        lo, hi, med = meta["num_ranges"][c]
        with num_cols[i % 4]:
            inputs[c] = st.number_input(LABELS.get(c, c),
                                        min_value=lo, max_value=hi, value=med)

    st.divider()
    cat_cols = st.columns(3)
    for i, c in enumerate(meta["categorical"]):
        with cat_cols[i % 3]:
            inputs[c] = st.selectbox(LABELS.get(c, c), meta["cat_options"][c])

    st.divider()
    if st.button("🔍 Predict default risk", type="primary", use_container_width=True):
        row  = pd.DataFrame([inputs])[meta["numeric"] + meta["categorical"]]
        prob = float(model.predict_proba(row)[0, 1])

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Default probability", f"{prob * 100:.1f}%")
            st.progress(min(prob, 1.0))
        with col2:
            if prob >= 0.6:
                st.error("🔴 **High risk** — likely to default.\n\n"
                         "Recommend rejecting or requiring collateral / a guarantor.")
            elif prob >= 0.35:
                st.warning("🟡 **Moderate risk** — review manually before approving.")
            else:
                st.success("🟢 **Low risk** — good candidate for approval.")


def performance_tab(meta):
    st.subheader("Model comparison")
    st.caption(f"Best model by ROC-AUC: **{meta['best_model']}** — "
               "trained on 800 samples, evaluated on 200 held-out loans.")

    if os.path.exists(METRICS_PATH):
        df = pd.read_csv(METRICS_PATH).round(3)
        df = df.rename(columns={"ROC_AUC": "ROC-AUC", "CV_ROC_AUC": "CV ROC-AUC"})
        st.dataframe(df.style.highlight_max(
            subset=["ROC-AUC"], color="#d4edda"), use_container_width=True)
    else:
        st.info("Run `python train.py` to generate metrics.")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ROC Curves")
        if os.path.exists(ROC_PATH):
            st.image(Image.open(ROC_PATH), use_container_width=True)
        else:
            st.warning("ROC curve not found. Run `python train.py` first.")

    with col2:
        st.markdown("#### Confusion Matrix")
        if os.path.exists(CM_PATH):
            st.image(Image.open(CM_PATH), use_container_width=True)
        else:
            st.warning("Confusion matrix not found. Run `python train.py` first.")

    st.divider()
    with st.expander("What do these charts mean?"):
        st.markdown("""
**ROC Curve** — plots True Positive Rate (catching real defaulters) against False
Positive Rate (wrongly flagging good applicants) at every threshold.
A model hugging the top-left corner is best; AUC = 1.0 is perfect.

**Confusion Matrix** — shows actual vs predicted outcomes on the test set:
- **True Negative** (top-left): good applicants correctly approved ✅
- **False Positive** (top-right): good applicants wrongly flagged ⚠️
- **False Negative** (bottom-left): real defaulters missed ❌ ← most costly
- **True Positive** (bottom-right): defaulters correctly caught ✅

We optimise for **Recall** (catching defaulters) because a missed defaulter
costs the bank more than a wrongly rejected good applicant.
""")


def main():
    model, meta = load_artifacts()

    st.title("💳 CreditGuard — Loan Default Predictor")
    st.caption(
        f"Predicts credit default risk using a **{meta['best_model']}** model "
        "trained on the UCI German Credit dataset (1,000 loans)."
    )

    tab1, tab2 = st.tabs(["🔍 Predict", "📊 Model Performance"])
    with tab1:
        predict_tab(model, meta)
    with tab2:
        performance_tab(meta)


if __name__ == "__main__":
    main()
