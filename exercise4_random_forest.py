# ============================================================
# MACHINE LEARNING ENGINEERING LAB
# Exercise 4 - Random Forests
# Python 3.12 + Scikit-Learn
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "bank.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "exercise4"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXERCISE 4
# RANDOM FOREST
# ============================================================

print("=" * 70)
print("EXERCISE 4 - RANDOM FORESTS")
print("Bank Marketing Ensemble & Feature Importance")
print("=" * 70)


# ============================================================
# STEP 1 - LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(
    DATA_FILE,
    sep=";"
)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# STEP 2 - FEATURES AND TARGET
# ============================================================

# Include pdays because the lab guide explicitly identifies
# duration, pdays and age as important predictors.

feature_cols = [
    "age",
    "job",
    "balance",
    "duration",
    "campaign",
    "pdays",
    "poutcome"
]

X = df[feature_cols].copy()

y = df["y"].map(
    {
        "no": 0,
        "yes": 1
    }
)


# ============================================================
# STEP 3 - TRAIN TEST SPLIT
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

print(f"Training rows: {len(X_train)}")
print(f"Testing rows : {len(X_test)}")


# ============================================================
# STEP 4 - PREPROCESSING
# ============================================================

numeric_cols = [
    "age",
    "balance",
    "duration",
    "campaign",
    "pdays"
]

categorical_cols = [
    "job",
    "poutcome"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
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
# STEP 5 - BASELINE RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("BASELINE RANDOM FOREST")
print("=" * 70)

baseline_rf = Pipeline(
    steps=[
        (
            "prep",
            preprocessor
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=100,
                oob_score=True,
                n_jobs=-1,
                random_state=42
            )
        )
    ]
)

baseline_rf.fit(
    X_train,
    y_train
)

baseline_oob = (
    baseline_rf.named_steps["rf"].oob_score_
)

baseline_test_accuracy = baseline_rf.score(
    X_test,
    y_test
)

baseline_pred = baseline_rf.predict(
    X_test
)

baseline_f1 = f1_score(
    y_test,
    baseline_pred
)

print(
    f"OOB Score     : {baseline_oob:.4f}"
)

print(
    f"Test Accuracy : {baseline_test_accuracy:.4f}"
)

print(
    f"Test F1       : {baseline_f1:.4f}"
)


# ============================================================
# STEP 6 - GRID SEARCH
# ============================================================

print("\n" + "=" * 70)
print("GRID SEARCH")
print("=" * 70)

rf_pipeline = Pipeline(
    steps=[
        (
            "prep",
            ColumnTransformer(
                transformers=[
                    (
                        "num",
                        "passthrough",
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
        ),
        (
            "rf",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

param_grid = {
    "rf__n_estimators": [
        100,
        200
    ],
    "rf__max_features": [
        "sqrt",
        "log2"
    ],
    "rf__max_depth": [
        8,
        12
    ]
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# STEP 7 - GRID SEARCH RESULTS
# ============================================================

print("\nBest parameters:")

print(
    grid_search.best_params_
)

print(
    f"Best CV F1: "
    f"{grid_search.best_score_:.4f}"
)


grid_results = pd.DataFrame(
    grid_search.cv_results_
)

grid_results = grid_results[
    [
        "param_rf__n_estimators",
        "param_rf__max_features",
        "param_rf__max_depth",
        "mean_test_score",
        "std_test_score",
        "rank_test_score"
    ]
].sort_values(
    "rank_test_score"
)

grid_results.to_csv(
    OUTPUT_DIR / "grid_search_results.csv",
    index=False
)


# ============================================================
# STEP 8 - FINAL RANDOM FOREST WITH OOB
# ============================================================

best_params = grid_search.best_params_

final_rf_pipeline = Pipeline(
    steps=[
        (
            "prep",
            ColumnTransformer(
                transformers=[
                    (
                        "num",
                        "passthrough",
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
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=best_params[
                    "rf__n_estimators"
                ],
                max_features=best_params[
                    "rf__max_features"
                ],
                max_depth=best_params[
                    "rf__max_depth"
                ],
                oob_score=True,
                n_jobs=-1,
                random_state=42
            )
        )
    ]
)

final_rf_pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# STEP 9 - FINAL RANDOM FOREST EVALUATION
# ============================================================

final_rf = final_rf_pipeline.named_steps["rf"]

rf_pred = final_rf_pipeline.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_f1 = f1_score(
    y_test,
    rf_pred
)

rf_oob = final_rf.oob_score_

print("\n" + "=" * 70)
print("FINAL RANDOM FOREST")
print("=" * 70)

print(
    f"OOB Score     : {rf_oob:.4f}"
)

print(
    f"Test Accuracy : {rf_accuracy:.4f}"
)

print(
    f"Test F1       : {rf_f1:.4f}"
)


# ============================================================
# STEP 10 - FEATURE NAMES
# ============================================================

final_prep = (
    final_rf_pipeline
    .named_steps["prep"]
)

feature_names = (
    final_prep
    .get_feature_names_out()
)


# ============================================================
# STEP 11 - MDI FEATURE IMPORTANCE
# ============================================================

mdi_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": final_rf.feature_importances_
    }
)

mdi_importance = mdi_importance.sort_values(
    "Importance",
    ascending=False
)

mdi_importance.to_csv(
    OUTPUT_DIR / "mdi_feature_importance.csv",
    index=False
)


print("\nTop 10 MDI features:")

print(
    mdi_importance.head(10).to_string(
        index=False
    )
)


# ============================================================
# STEP 12 - AGGREGATE MDI BY ORIGINAL FEATURE
# ============================================================

expanded_source_features = []

for col in numeric_cols:
    expanded_source_features.append(col)

cat_encoder = (
    final_prep
    .named_transformers_["cat"]
)

for col, categories in zip(
    categorical_cols,
    cat_encoder.categories_
):
    # drop='first' removes first category
    for _ in categories[1:]:
        expanded_source_features.append(col)


if len(expanded_source_features) == len(
    feature_names
):

    aggregate_mdi = pd.DataFrame(
        {
            "OriginalFeature":
                expanded_source_features,
            "Importance":
                final_rf.feature_importances_
        }
    )

    aggregate_mdi = (
        aggregate_mdi
        .groupby(
            "OriginalFeature",
            as_index=False
        )["Importance"]
        .sum()
        .sort_values(
            "Importance",
            ascending=False
        )
    )

else:

    aggregate_mdi = mdi_importance.copy()

    aggregate_mdi.columns = [
        "OriginalFeature",
        "Importance"
    ]


aggregate_mdi.to_csv(
    OUTPUT_DIR / "aggregated_mdi_importance.csv",
    index=False
)

print("\nAggregated MDI importance:")

print(
    aggregate_mdi.to_string(
        index=False
    )
)


# ============================================================
# STEP 13 - MDI PLOT
# ============================================================

plot_mdi = aggregate_mdi.head(10).sort_values(
    "Importance"
)

plt.figure(
    figsize=(9, 6)
)

plt.barh(
    plot_mdi["OriginalFeature"],
    plot_mdi["Importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Random Forest - MDI Feature Importance"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "mdi_feature_importance.png",
    dpi=150
)

plt.close()


# ============================================================
# STEP 14 - PERMUTATION IMPORTANCE
# ============================================================

print("\nCalculating permutation importance...")

perm = permutation_importance(
    final_rf_pipeline,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=3,
    random_state=42,
    n_jobs=-1
)

permutation_df = pd.DataFrame(
    {
        "Feature": X_test.columns,
        "MeanImportance": (
            perm.importances_mean
        ),
        "StdImportance": (
            perm.importances_std
        )
    }
)

permutation_df = permutation_df.sort_values(
    "MeanImportance",
    ascending=False
)

permutation_df.to_csv(
    OUTPUT_DIR / "permutation_importance.csv",
    index=False
)

print("\nPermutation importance:")

print(
    permutation_df.to_string(
        index=False
    )
)


# ============================================================
# STEP 15 - PERMUTATION IMPORTANCE PLOT
# ============================================================

plot_perm = (
    permutation_df
    .head(10)
    .sort_values("MeanImportance")
)

plt.figure(
    figsize=(9, 6)
)

plt.barh(
    plot_perm["Feature"],
    plot_perm["MeanImportance"]
)

plt.xlabel(
    "Mean F1 Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Random Forest - Permutation Importance"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "permutation_importance.png",
    dpi=150
)

plt.close()


# ============================================================
# STEP 16 - OOB CONVERGENCE
# ============================================================

print("\n" + "=" * 70)
print("OOB CONVERGENCE")
print("=" * 70)

# Transform training data using the already-fitted preprocessor.
# Random Forest requires numeric input.

X_train_transformed = final_prep.transform(
    X_train
)

oob_rf = RandomForestClassifier(
    n_estimators=50,
    oob_score=True,
    warm_start=True,
    n_jobs=-1,
    random_state=42,
    max_features=best_params[
        "rf__max_features"
    ],
    max_depth=best_params[
        "rf__max_depth"
    ]
)

oob_counts = []
oob_scores = []

for n_trees in [50, 100, 150, 200]:

    oob_rf.set_params(
        n_estimators=n_trees
    )

    oob_rf.fit(
        X_train_transformed,
        y_train
    )

    oob_counts.append(
        n_trees
    )

    oob_scores.append(
        oob_rf.oob_score_
    )

    print(
        f"Trees: {n_trees:3d} | "
        f"OOB Score: {oob_rf.oob_score_:.4f}"
    )


oob_df = pd.DataFrame(
    {
        "Trees": oob_counts,
        "OOBScore": oob_scores
    }
)

oob_df.to_csv(
    OUTPUT_DIR / "oob_convergence.csv",
    index=False
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    oob_df["Trees"],
    oob_df["OOBScore"],
    marker="o"
)

plt.xlabel(
    "Number of Trees"
)

plt.ylabel(
    "OOB Score"
)

plt.title(
    "Random Forest OOB Convergence"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "oob_convergence.png",
    dpi=150
)

plt.close()

# ============================================================
# STEP 17 - LOGISTIC REGRESSION COMPARISON
# ============================================================

logistic_pipeline = Pipeline(
    steps=[
        (
            "prep",
            ColumnTransformer(
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
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42
            )
        )
    ]
)

logistic_pipeline.fit(
    X_train,
    y_train
)

logistic_pred = (
    logistic_pipeline.predict(
        X_test
    )
)

logistic_f1 = f1_score(
    y_test,
    logistic_pred
)


# ============================================================
# STEP 18 - DECISION TREE COMPARISON
# ============================================================

decision_tree_pipeline = Pipeline(
    steps=[
        (
            "prep",
            ColumnTransformer(
                transformers=[
                    (
                        "num",
                        "passthrough",
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
        ),
        (
            "model",
            DecisionTreeClassifier(
                criterion="gini",
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
        )
    ]
)

decision_tree_pipeline.fit(
    X_train,
    y_train
)

tree_pred = (
    decision_tree_pipeline.predict(
        X_test
    )
)

tree_f1 = f1_score(
    y_test,
    tree_pred
)


# ============================================================
# STEP 19 - MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "Test F1": [
            logistic_f1,
            tree_f1,
            rf_f1
        ],
        "Test Accuracy": [
            accuracy_score(
                y_test,
                logistic_pred
            ),
            accuracy_score(
                y_test,
                tree_pred
            ),
            rf_accuracy
        ]
    }
)

comparison.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# STEP 20 - FINAL ACCEPTANCE
# ============================================================

print("\n" + "=" * 70)
print("EXERCISE 4 ACCEPTANCE CHECK")
print("=" * 70)

oob_pass = rf_oob >= 0.88

f1_pass = rf_f1 >= 0.65

top_three = aggregate_mdi.head(3)[
    "OriginalFeature"
].tolist()

print(
    f"OOB Score >= 0.88 : "
    f"{'PASS' if oob_pass else 'CHECK'}"
)

print(
    f"F1 >= 0.65        : "
    f"{'PASS' if f1_pass else 'CHECK'}"
)

print(
    f"Final OOB Score   : "
    f"{rf_oob:.4f}"
)

print(
    f"Final Test F1     : "
    f"{rf_f1:.4f}"
)

print(
    f"Top 3 MDI Features: "
    f"{top_three}"
)


# ============================================================
# STEP 21 - SAVE PREDICTIONS
# ============================================================

rf_predictions = X_test.copy()

rf_predictions["Actual"] = y_test.values

rf_predictions["Prediction"] = rf_pred

rf_predictions.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)


# ============================================================
# OUTPUT FILES
# ============================================================

print("\nGenerated files:")

for file in sorted(
    OUTPUT_DIR.iterdir()
):
    print(
        f" - {file.name}"
    )

print("\nExercise 4 completed.")
