"""
Loan Default Prediction - Streamlit app.

Loads the trained pipeline and lets a user enter applicant details to get a
default-risk probability.

Run:  streamlit run app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join("models", "credit_risk_model.joblib")
META_PATH = os.path.join("models", "feature_metadata.joblib")

st.set_page_config(page_title="Loan Default Predictor", page_icon="💳", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model, meta


# Friendlier labels for the raw column names
LABELS = {
    "duration": "Loan duration (months)",
    "amount": "Credit amount (DM)",
    "installment_rate": "Installment rate (% of income)",
    "present_residence": "Years at current residence",
    "age": "Age (years)",
    "number_credits": "Existing credits at this bank",
    "people_liable": "Dependents",
    "status": "Checking account status",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "savings": "Savings account",
    "employment_duration": "Employment duration",
    "personal_status_sex": "Personal status & sex",
    "other_debtors": "Other debtors / guarantors",
    "property": "Property owned",
    "other_installment_plans": "Other installment plans",
    "housing": "Housing",
    "job": "Job type",
    "telephone": "Registered telephone",
    "foreign_worker": "Foreign worker",
}


def main():
    model, meta = load_artifacts()

    st.title("💳 Loan Default Predictor")
    st.caption(
        f"Predicts the probability that a loan applicant will **default**, "
        f"using a {meta['best_model']} model trained on the German Credit dataset."
    )

    st.subheader("Applicant details")
    inputs = {}

    # Numeric inputs
    cols = st.columns(2)
    for i, c in enumerate(meta["numeric"]):
        lo, hi, med = meta["num_ranges"][c]
        with cols[i % 2]:
            inputs[c] = st.number_input(LABELS.get(c, c), min_value=lo,
                                        max_value=hi, value=med)

    # Categorical inputs
    cat_cols = st.columns(2)
    for i, c in enumerate(meta["categorical"]):
        with cat_cols[i % 2]:
            inputs[c] = st.selectbox(LABELS.get(c, c), meta["cat_options"][c])

    if st.button("Predict default risk", type="primary"):
        row = pd.DataFrame([inputs])[meta["numeric"] + meta["categorical"]]
        prob = float(model.predict_proba(row)[0, 1])

        st.subheader("Result")
        st.metric("Default probability", f"{prob * 100:.1f}%")
        st.progress(min(prob, 1.0))

        if prob >= 0.6:
            st.error("High risk — likely to default. Recommend rejecting or "
                     "requiring collateral / a guarantor.")
        elif prob >= 0.35:
            st.warning("Moderate risk — review manually before approving.")
        else:
            st.success("Low risk — good candidate for approval.")

    with st.expander("About this model"):
        st.write(
            "Trained on the UCI German Credit dataset (1,000 loans). Several "
            "classifiers were compared (Logistic Regression, KNN, Decision Tree, "
            "Random Forest, Gradient Boosting); the best by ROC-AUC is served here. "
            "See `reports/model_comparison.csv` for the full comparison."
        )


if __name__ == "__main__":
    main()
