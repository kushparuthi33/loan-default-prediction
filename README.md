# Loan Default Prediction

Predicting whether a loan applicant will **default** using the UCI **German Credit**
dataset (1,000 historical loans). The project covers the full data-science workflow —
EDA, preprocessing, model comparison, evaluation — and ships an interactive
**Streamlit** app that scores new applicants.

> Built as a supervised-learning project covering regression/classification,
> regularization, KNN, decision trees, random forests, and gradient boosting.

---

## Problem

When a bank issues a loan, approving a borrower who later defaults causes a direct
financial loss. The goal is to flag high-risk applicants up front so the lender can
reject, ask for collateral, or review manually. This is a binary classification task:
`default = 1` (bad credit) vs `0` (repaid).

## Dataset

- **UCI German Credit** — 1,000 loans, 20 features (7 numeric, 13 categorical).
- Features include checking-account status, credit history, loan purpose, credit
  amount, duration, savings, employment length, age, housing, and more.
- **~30% default rate** — a moderately imbalanced problem, so the models are tuned for
  recall and ROC-AUC, not just accuracy.

## Approach

1. **EDA** — target balance, numeric distributions, and default rates across categories.
2. **Preprocessing** — `StandardScaler` for numeric + `OneHotEncoder` for categorical,
   wrapped in a scikit-learn `Pipeline` to prevent data leakage.
3. **Modeling** — trained and compared five classifiers with balanced class weights.
4. **Evaluation** — ROC-AUC, confusion matrix, and feature importance, with 5-fold CV.
5. **Deployment** — best pipeline served through a Streamlit app.

## Results

Models ranked by ROC-AUC on a held-out 20% test set (5-fold CV AUC also shown):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.750 | 0.558 | 0.800 | 0.658 | **0.806** | 0.785 |
| Random Forest | 0.755 | 0.590 | 0.600 | 0.595 | 0.805 | 0.795 |
| Gradient Boosting | 0.790 | 0.680 | 0.567 | 0.618 | 0.792 | 0.771 |
| KNN | 0.755 | 0.667 | 0.367 | 0.473 | 0.747 | 0.740 |
| Decision Tree | 0.570 | 0.377 | 0.667 | 0.482 | 0.599 | 0.689 |

**Logistic Regression** gave the best ROC-AUC (0.81) while catching **80% of actual
defaulters** (recall). The strongest predictors were checking-account status, credit
history, loan duration, savings, and credit amount — all consistent with financial
intuition.

## Project structure

```
loan-default-prediction/
├── data/
│   └── german_credit.csv          # dataset (1,000 rows)
├── notebooks/
│   └── loan_default_analysis.ipynb # EDA + modeling walkthrough
├── reports/
│   ├── model_comparison.csv        # metrics for every model
│   ├── roc_curves.png
│   └── confusion_matrix.png
├── models/                         # created by train.py
│   ├── credit_risk_model.joblib
│   └── feature_metadata.joblib
├── train.py                        # train, compare, export best model
├── app.py                          # Streamlit prediction app
├── requirements.txt
└── README.md
```

## How to run

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. train models and export the best pipeline
python train.py

# 3. launch the interactive predictor
streamlit run app.py
```

Or open `notebooks/loan_default_analysis.ipynb` to step through the full analysis.

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Streamlit

## Future work

- Threshold tuning for a target precision/recall trade-off
- Hyperparameter search (GridSearchCV)
- Adding XGBoost and SHAP-based explanations
