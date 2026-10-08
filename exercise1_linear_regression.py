# ============================================================
# MACHINE LEARNING ENGINEERING LAB
# Exercise 1 - Multiple Linear Regression
# Python 3.12 + Scikit-Learn
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "insurance.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# EXERCISE 1
# MULTIPLE LINEAR REGRESSION
# ============================================================

print("=" * 70)
print("EXERCISE 1 - MULTIPLE LINEAR REGRESSION")
print("Insurance Cost Prediction")
print("=" * 70)


# ============================================================
# STEP 1 - LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Place insurance.csv inside the data folder."
    )

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())


# ============================================================
# STEP 2 - CHECK TARGET SKEWNESS
# ============================================================

skewness = df["charges"].skew()

print("\n" + "=" * 70)
print("TARGET SKEWNESS")
print("=" * 70)

print(f"Charges skewness: {skewness:.4f}")

if skewness > 1.5:
    print("Result: Charges has strong positive skewness.")
else:
    print("Result: Positive skewness is lower than 1.5.")


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(df["charges"], bins=30)

plt.title("Distribution of Medical Charges")
plt.xlabel("Charges")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "charges_distribution.png",
    dpi=150
)

plt.close()


# ============================================================
# STEP 3 - FEATURES AND TARGET
# ============================================================

X = df.drop("charges", axis=1)
y = df["charges"]


num_cols = [
    "age",
    "bmi",
    "children"
]

cat_cols = [
    "sex",
    "smoker",
    "region"
]


# ============================================================
# STEP 4 - PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            num_cols
        ),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            cat_cols
        )
    ]
)


# ============================================================
# STEP 5 - TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

print(f"Training rows : {len(X_train)}")
print(f"Testing rows  : {len(X_test)}")


# ============================================================
# STEP 6 - LINEAR REGRESSION PIPELINE
# ============================================================

model_pipeline = Pipeline(
    steps=[
        (
            "prep",
            preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


# ============================================================
# STEP 7 - TRAIN MODEL
# ============================================================

model_pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# STEP 8 - PREDICTIONS
# ============================================================

y_pred = model_pipeline.predict(X_test)


# ============================================================
# STEP 9 - MODEL EVALUATION
# ============================================================

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(f"MSE  : {mse:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# STEP 10 - COEFFICIENTS AND INTERCEPT
# ============================================================

regressor = model_pipeline.named_steps["regressor"]
prep = model_pipeline.named_steps["prep"]

feature_names = prep.get_feature_names_out()

coefficients = pd.DataFrame(
    {
        "Feature": feature_names,
        "Coefficient": regressor.coef_
    }
)

coefficients["AbsoluteCoefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "AbsoluteCoefficient",
    ascending=False
)

print("\n" + "=" * 70)
print("MODEL COEFFICIENTS")
print("=" * 70)

print(coefficients[
    ["Feature", "Coefficient"]
].to_string(index=False))

print(f"\nIntercept: {regressor.intercept_:,.2f}")


# ============================================================
# STEP 11 - SMOKER COEFFICIENT
# ============================================================

smoker_rows = coefficients[
    coefficients["Feature"].str.contains(
        "smoker",
        case=False,
        regex=False
    )
]

print("\n" + "=" * 70)
print("SMOKER EFFECT")
print("=" * 70)

print(smoker_rows[
    ["Feature", "Coefficient"]
].to_string(index=False))

if not smoker_rows.empty:
    smoker_coefficient = smoker_rows.iloc[0]["Coefficient"]

    print(
        f"\nSmoker coefficient: ${smoker_coefficient:,.2f}"
    )

    if smoker_coefficient > 23000:
        print(
            "Acceptance: PASS - smoker coefficient "
            "is greater than $23,000."
        )
    else:
        print(
            "Acceptance: CHECK - smoker coefficient "
            "is not greater than $23,000."
        )


# ============================================================
# STEP 12 - ACTUAL VS PREDICTED
# ============================================================

prediction_df = pd.DataFrame(
    {
        "ActualCharges": y_test.values,
        "PredictedCharges": y_pred
    }
)

prediction_df["Residual"] = (
    prediction_df["ActualCharges"]
    - prediction_df["PredictedCharges"]
)

prediction_df.to_csv(
    OUTPUT_DIR / "exercise1_predictions.csv",
    index=False
)

print(
    f"\nPrediction file saved: "
    f"{OUTPUT_DIR / 'exercise1_predictions.csv'}"
)


plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

min_value = min(
    y_test.min(),
    y_pred.min()
)

max_value = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value]
)

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Medical Charges")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=150
)

plt.close()


# ============================================================
# STEP 13 - RESIDUAL DIAGNOSTIC
# ============================================================

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "residuals_vs_fitted.png",
    dpi=150
)

plt.close()


# ============================================================
# STEP 14 - LOG1P TARGET BENCHMARK
# ============================================================

print("\n" + "=" * 70)
print("LOG1P TARGET BENCHMARK")
print("=" * 70)


log_model = Pipeline(
    steps=[
        (
            "prep",
            ColumnTransformer(
                transformers=[
                    (
                        "num",
                        StandardScaler(),
                        num_cols
                    ),
                    (
                        "cat",
                        OneHotEncoder(
                            drop="first",
                            handle_unknown="ignore"
                        ),
                        cat_cols
                    )
                ]
            )
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)

y_train_log = np.log1p(y_train)

log_model.fit(
    X_train,
    y_train_log
)

log_pred = log_model.predict(X_test)

log_pred_original_scale = np.expm1(
    log_pred
)

log_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        log_pred_original_scale
    )
)

log_r2 = r2_score(
    y_test,
    log_pred_original_scale
)

print(f"Log1p RMSE : ${log_rmse:,.2f}")
print(f"Log1p R²   : {log_r2:.4f}")


# ============================================================
# STEP 15 - FINAL ACCEPTANCE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL ACCEPTANCE CHECK")
print("=" * 70)

r2_pass = r2 >= 0.75
rmse_pass = rmse < 4800

print(
    f"R² >= 0.75      : "
    f"{'PASS' if r2_pass else 'CHECK'}"
)

print(
    f"RMSE < $4,800   : "
    f"{'PASS' if rmse_pass else 'CHECK'}"
)

print(
    f"Residual plot    : "
    f"{'CREATED'}"
)

print(
    f"Prediction CSV   : "
    f"{'CREATED'}"
)

print("\nExercise 1 completed.")


# ============================================================
# OUTPUT FILES
# ============================================================

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(f" - {file.name}")

# ============================================================
# EXERCISE 2
# LOGISTIC REGRESSION - BANK MARKETING
# ============================================================

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
from sklearn.preprocessing import RobustScaler


print("\n\n")
print("=" * 70)
print("EXERCISE 2 - LOGISTIC REGRESSION")
print("Bank Marketing Direct Conversion")
print("=" * 70)


# ============================================================
# STEP 1 - LOAD DATA
# ============================================================

BANK_FILE = BASE_DIR / "data" / "bank.csv"

if not BANK_FILE.exists():
    raise FileNotFoundError(
        f"Bank dataset not found: {BANK_FILE}"
    )

bank_df = pd.read_csv(
    BANK_FILE,
    sep=";"
)

print("\nDataset shape:")
print(bank_df.shape)

print("\nFirst 5 rows:")
print(bank_df.head())

print("\nMissing values:")
print(bank_df.isnull().sum())


# ============================================================
# STEP 2 - TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(bank_df["y"].value_counts())

print("\nTarget percentage:")
print(
    bank_df["y"].value_counts(
        normalize=True
    ) * 100
)


# ============================================================
# STEP 3 - SELECT GUIDE FEATURES
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

X_bank = bank_df[feature_cols].copy()

y_bank = (
    bank_df[target_col]
    .map({"no": 0, "yes": 1})
)


# ============================================================
# STEP 4 - CHECK TARGET
# ============================================================

if y_bank.isnull().any():
    raise ValueError(
        "Target contains unexpected values."
    )


print("\nTarget mapping:")
print("no  -> 0")
print("yes -> 1")


# ============================================================
# STEP 5 - TRAIN TEST SPLIT
# STRATIFIED 75/25
# ============================================================

X_train_bank, X_test_bank, y_train_bank, y_test_bank = (
    train_test_split(
        X_bank,
        y_bank,
        test_size=0.25,
        random_state=42,
        stratify=y_bank
    )
)


print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

print(
    f"Training rows: {len(X_train_bank)}"
)

print(
    f"Testing rows : {len(X_test_bank)}"
)

print(
    f"Train positive rate: "
    f"{y_train_bank.mean():.4f}"
)

print(
    f"Test positive rate : "
    f"{y_test_bank.mean():.4f}"
)


# ============================================================
# STEP 6 - PREPROCESSING
# ============================================================

bank_numeric = [
    "age",
    "balance",
    "duration",
    "campaign"
]

bank_categorical = [
    "job",
    "poutcome"
]


bank_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            RobustScaler(),
            bank_numeric
        ),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            bank_categorical
        )
    ]
)


# ============================================================
# STEP 7 - LOGISTIC REGRESSION PIPELINE
# ============================================================

logistic_pipeline = Pipeline(
    steps=[
        (
            "prep",
            bank_preprocessor
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
# STEP 8 - TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression...")

logistic_pipeline.fit(
    X_train_bank,
    y_train_bank
)

print("Training completed.")


# ============================================================
# STEP 9 - PREDICT PROBABILITIES
# ============================================================

y_prob_bank = logistic_pipeline.predict_proba(
    X_test_bank
)[:, 1]


# ============================================================
# STEP 10 - DEFAULT THRESHOLD
# ============================================================

y_pred_050 = (
    y_prob_bank >= 0.50
).astype(int)

recall_050 = recall_score(
    y_test_bank,
    y_pred_050
)

f1_050 = f1_score(
    y_test_bank,
    y_pred_050
)

print("\n" + "=" * 70)
print("DEFAULT THRESHOLD = 0.50")
print("=" * 70)

print(
    f"Recall: {recall_050:.4f}"
)

print(
    f"F1 Score: {f1_050:.4f}"
)


# ============================================================
# STEP 11 - REQUIRED THRESHOLD = 0.35
# ============================================================

threshold = 0.35

y_pred_bank = (
    y_prob_bank >= threshold
).astype(int)


# ============================================================
# STEP 12 - ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test_bank,
    y_prob_bank
)

average_precision = average_precision_score(
    y_test_bank,
    y_prob_bank
)

recall_035 = recall_score(
    y_test_bank,
    y_pred_bank
)

f1_035 = f1_score(
    y_test_bank,
    y_pred_bank
)


print("\n" + "=" * 70)
print("THRESHOLD = 0.35")
print("=" * 70)

print(
    f"ROC-AUC           : {roc_auc:.4f}"
)

print(
    f"Average Precision : {average_precision:.4f}"
)

print(
    f"Recall            : {recall_035:.4f}"
)

print(
    f"F1 Score          : {f1_035:.4f}"
)


# ============================================================
# STEP 13 - CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test_bank,
    y_pred_bank
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

tn, fp, fn, tp = cm.ravel()

print(f"\nTrue Negative : {tn}")
print(f"False Positive: {fp}")
print(f"False Negative: {fn}")
print(f"True Positive : {tp}")


# Save confusion matrix

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
    OUTPUT_DIR / "exercise2_confusion_matrix.csv"
)


# ============================================================
# STEP 14 - CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test_bank,
        y_pred_bank,
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

fpr, tpr, roc_thresholds = roc_curve(
    y_test_bank,
    y_prob_bank
)

plt.figure(figsize=(8, 5))

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

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Bank Marketing")

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "exercise2_roc_curve.png",
    dpi=150
)

plt.close()


# ============================================================
# STEP 16 - PRECISION RECALL CURVE
# ============================================================

precision, recall, pr_thresholds = (
    precision_recall_curve(
        y_test_bank,
        y_prob_bank
    )
)

plt.figure(figsize=(8, 5))

plt.plot(
    recall,
    precision
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(
    "Precision-Recall Curve - Bank Marketing"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "exercise2_precision_recall_curve.png",
    dpi=150
)

plt.close()


# ============================================================
# STEP 17 - SAVE PREDICTIONS
# ============================================================

exercise2_predictions = X_test_bank.copy()

exercise2_predictions["Actual"] = (
    y_test_bank.values
)

exercise2_predictions["Probability"] = (
    y_prob_bank
)

exercise2_predictions["Prediction"] = (
    y_pred_bank
)

exercise2_predictions["Threshold"] = (
    threshold
)

exercise2_predictions.to_csv(
    OUTPUT_DIR / "exercise2_predictions.csv",
    index=False
)


# ============================================================
# STEP 18 - ACCEPTANCE CHECK
# ============================================================

print("\n" + "=" * 70)
print("EXERCISE 2 ACCEPTANCE CHECK")
print("=" * 70)

auc_pass = roc_auc >= 0.78
recall_pass = recall_035 >= 0.70

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

print("\nExercise 2 completed.")


# ============================================================
# OUTPUT FILES
# ============================================================

print("\nExercise 2 generated files:")

exercise2_files = [
    "exercise2_confusion_matrix.csv",
    "exercise2_predictions.csv",
    "exercise2_roc_curve.png",
    "exercise2_precision_recall_curve.png"
]

for filename in exercise2_files:
    print(
        f" - {filename}"
    )
