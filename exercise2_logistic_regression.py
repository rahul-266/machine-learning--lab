# ============================================================
# MACHINE LEARNING ENGINEERING LAB
# Exercise 2 - Logistic Regression
# Python 3.12 + Scikit-Learn
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "bank.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "exercise2"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXERCISE 2
# LOGISTIC REGRESSION
# ============================================================

print("=" * 70)
print("EXERCISE 2 - LOGISTIC REGRESSION")
print("Bank Marketing Direct Conversion")
print("=" * 70)


# ============================================================
# STEP 1 - LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Place bank.csv inside the data folder."
    )

df = pd.read_csv(
    DATA_FILE,
    sep=";"
)

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# STEP 2 - TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(
    df["y"].value_counts()
)

print("\nTarget percentage:")

print(
    (df["y"].value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# STEP 3 - SELECT FEATURES
# ============================================================

feature_cols = [
    "age",
    "job",
    "balance",
    "duration",
    "campaign",
    "poutcome"
]

target_col = "y"


X = df[feature_cols].copy()

y = df[target_col].map(
    {
        "no": 0,
        "yes": 1
    }
)


# ============================================================
# STEP 4 - VALIDATE TARGET
# ============================================================

if y.isnull().any():
    raise ValueError(
        "Unexpected values found in target column y."
    )

print("\nTarget mapping:")
print("no  -> 0")
print("yes -> 1")


# ============================================================
# STEP 5 - TRAIN TEST SPLIT
# 75/25 STRATIFIED
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

print(
    f"Training rows : {len(X_train)}"
)

print(
    f"Testing rows  : {len(X_test)}"
)

print(
    f"Train positive rate: {y_train.mean():.4f}"
)

print(
    f"Test positive rate : {y_test.mean():.4f}"
)


# ============================================================
# STEP 6 - PREPROCESSING
# ============================================================

numeric_cols = [
    "age",
    "balance",
    "duration",
    "campaign"
]

categorical_cols = [
    "job",
    "poutcome"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            RobustScaler(),
            numeric_cols
        ),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_cols
        )
    ]
)


# ============================================================
# STEP 7 - LOGISTIC REGRESSION
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42
            )
        )
    ]
)


# ============================================================
# STEP 8 - TRAIN
# ============================================================

print("\nTraining Logistic Regression...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# STEP 9 - PREDICT PROBABILITIES
# ============================================================

y_prob = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# STEP 10 - DEFAULT THRESHOLD 0.50
# ============================================================

y_pred_50 = (
    y_prob >= 0.50
).astype(int)

recall_50 = recall_score(
    y_test,
    y_pred_50
)

f1_50 = f1_score(
    y_test,
    y_pred_50
)

print("\n" + "=" * 70)
print("DEFAULT THRESHOLD = 0.50")
print("=" * 70)

print(
    f"Recall  : {recall_50:.4f}"
)

print(
    f"F1 Score: {f1_50:.4f}"
)


# ============================================================
# STEP 11 - REQUIRED THRESHOLD 0.35
# ============================================================

threshold = 0.35

y_pred = (
    y_prob >= threshold
).astype(int)


# ============================================================
# STEP 12 - EVALUATION
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

avg_precision = average_precision_score(
    y_test,
    y_prob
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("THRESHOLD = 0.35")
print("=" * 70)

print(
    f"ROC-AUC           : {roc_auc:.4f}"
)

print(
    f"Average Precision : {avg_precision:.4f}"
)

print(
    f"Recall            : {recall:.4f}"
)

print(
    f"F1 Score          : {f1:.4f}"
)


# ============================================================
# STEP 13 - CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

print(f"\nTrue Negative : {tn}")
print(f"False Positive: {fp}")
print(f"False Negative: {fn}")
print(f"True Positive : {tp}")


cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_0",
        "Actual_1"
    ],
    columns=[
        "Predicted_0",
        "Predicted_1"
    ]
)

cm_df.to_csv(
    OUTPUT_DIR / "confusion_matrix.csv"
)


# ============================================================
# STEP 14 - CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Deposit",
            "Subscribed"
        ],
        zero_division=0
    )
)


# ============================================================
# STEP 15 - ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Bank Marketing"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curve.png",
    dpi=150
)

plt.close()


# ============================================================
# STEP 16 - PRECISION RECALL CURVE
# ============================================================

precision, recall_values, _ = (
    precision_recall_curve(
        y_test,
        y_prob
    )
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    recall_values,
    precision
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve - Bank Marketing"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "precision_recall_curve.png",
    dpi=150
)

plt.close()


# ============================================================
# STEP 17 - SAVE PREDICTIONS
# ============================================================

prediction_output = X_test.copy()

prediction_output["Actual"] = (
    y_test.values
)

prediction_output["Probability"] = (
    y_prob
)

prediction_output["Prediction"] = (
    y_pred
)

prediction_output["Threshold"] = (
    threshold
)

prediction_output.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)


# ============================================================
# STEP 18 - ACCEPTANCE CHECK
# ============================================================

print("\n" + "=" * 70)
print("EXERCISE 2 ACCEPTANCE CHECK")
print("=" * 70)

auc_pass = roc_auc >= 0.78
recall_pass = recall >= 0.70

print(
    f"ROC-AUC >= 0.78 : "
    f"{'PASS' if auc_pass else 'CHECK'}"
)

print(
    f"Recall >= 0.70   : "
    f"{'PASS' if recall_pass else 'CHECK'}"
)

print(
    f"Threshold        : {threshold}"
)


# ============================================================
# OUTPUTS
# ============================================================

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(
        f" - {file.name}"
    )


print("\nExercise 2 completed.")
