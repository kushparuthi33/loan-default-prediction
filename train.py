"""
Loan Default Prediction - model training and comparison.

Trains several classifiers on the German Credit dataset to predict whether a
loan applicant will DEFAULT (bad credit risk), compares them, and saves the
best pipeline plus evaluation reports.

Run:  python train.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # render plots without a display
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay,
)

DATA_PATH = os.path.join("data", "german_credit.csv")
MODEL_DIR = "models"
REPORT_DIR = "reports"
RANDOM_STATE = 42

NUMERIC = ["duration", "amount", "installment_rate", "present_residence",
           "age", "number_credits", "people_liable"]
CATEGORICAL = ["status", "credit_history", "purpose", "savings",
               "employment_duration", "personal_status_sex", "other_debtors",
               "property", "other_installment_plans", "housing", "job",
               "telephone", "foreign_worker"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    # Reframe target: in this dataset credit_risk = 1 (good) / 0 (bad).
    # We predict DEFAULT, so default = 1 when credit is bad.
    df["default"] = (df["credit_risk"] == 0).astype(int)
    df = df.drop(columns=["credit_risk"])
    return df


def build_pipeline(model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", pre), ("clf", model)])


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = load_data()
    X = df[NUMERIC + CATEGORICAL]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                                random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8,
                                                class_weight="balanced",
                                                random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    rows = []
    fitted = {}
    roc_data = {}
    for name, model in models.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=5,
                                 scoring="roc_auc").mean()
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC_AUC": roc_auc_score(y_test, proba),
            "CV_ROC_AUC": cv_auc,
        })
        fitted[name] = pipe
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_test, proba))

    results = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    results.to_csv(os.path.join(REPORT_DIR, "model_comparison.csv"), index=False)
    print("\n=== Model comparison (sorted by ROC-AUC) ===")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    best_name = results.iloc[0]["Model"]
    best_pipe = fitted[best_name]
    print(f"\nBest model: {best_name}")

    # --- Save best pipeline + metadata for the app ---
    joblib.dump(best_pipe, os.path.join(MODEL_DIR, "credit_risk_model.joblib"))
    meta = {
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "cat_options": {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL},
        "num_ranges": {c: (int(df[c].min()), int(df[c].max()), int(df[c].median()))
                       for c in NUMERIC},
        "best_model": best_name,
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "feature_metadata.joblib"))

    # --- Plots ---
    # ROC curves
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Loan Default Models"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(REPORT_DIR, "roc_curves.png"), dpi=120)
    plt.close()

    # Confusion matrix for best model
    cm = confusion_matrix(y_test, best_pipe.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=["Repaid", "Default"]).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout(); plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()

    print("\nSaved: models/credit_risk_model.joblib, models/feature_metadata.joblib")
    print("Saved: reports/model_comparison.csv, reports/roc_curves.png, reports/confusion_matrix.png")


if __name__ == "__main__":
    main()
