# ============================================================
# MACHINE LEARNING ENGINEERING LAB
# Exercise 3 - Decision Trees
# Python 3.12 + Scikit-Learn
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import (
    DecisionTreeClassifier,
    export_text,
    plot_tree
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "bank.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "exercise3"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXERCISE 3
# DECISION TREES
# ============================================================

print("=" * 70)
print("EXERCISE 3 - DECISION TREES")
print("Bank Marketing Decision Rule Mining")
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

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# STEP 2 - FEATURES AND TARGET
# ============================================================

feature_cols = [
    "age",
    "job",
    "balance",
    "duration",
    "campaign",
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
# STEP 5 - FIT PREPROCESSOR
# ============================================================

X_train_prep = preprocessor.fit_transform(
    X_train
)

X_test_prep = preprocessor.transform(
    X_test
)

feature_names = (
    preprocessor.get_feature_names_out()
)


# ============================================================
# STEP 6 - BASELINE TREE
# GINI VS ENTROPY
# ============================================================

print("\n" + "=" * 70)
print("GINI VS ENTROPY")
print("=" * 70)


gini_tree = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

entropy_tree = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)


gini_tree.fit(
    X_train_prep,
    y_train
)

entropy_tree.fit(
    X_train_prep,
    y_train
)


gini_train_acc = gini_tree.score(
    X_train_prep,
    y_train
)

gini_test_acc = gini_tree.score(
    X_test_prep,
    y_test
)

entropy_train_acc = entropy_tree.score(
    X_train_prep,
    y_train
)

entropy_test_acc = entropy_tree.score(
    X_test_prep,
    y_test
)


print("\nGini:")
print(f"Train Accuracy: {gini_train_acc:.4f}")
print(f"Test Accuracy : {gini_test_acc:.4f}")
print(f"Tree Depth    : {gini_tree.get_depth()}")

print("\nEntropy:")
print(f"Train Accuracy: {entropy_train_acc:.4f}")
print(f"Test Accuracy : {entropy_test_acc:.4f}")
print(f"Tree Depth    : {entropy_tree.get_depth()}")


comparison_df = pd.DataFrame(
    {
        "Criterion": [
            "Gini",
            "Entropy"
        ],
        "TrainAccuracy": [
            gini_train_acc,
            entropy_train_acc
        ],
        "TestAccuracy": [
            gini_test_acc,
            entropy_test_acc
        ],
        "Depth": [
            gini_tree.get_depth(),
            entropy_tree.get_depth()
        ]
    }
)

comparison_df.to_csv(
    OUTPUT_DIR / "gini_vs_entropy.csv",
    index=False
)


# ============================================================
# STEP 7 - UNCONSTRAINED TREE
# OVERFITTING DEMONSTRATION
# ============================================================

print("\n" + "=" * 70)
print("UNCONSTRAINED TREE - OVERFITTING CHECK")
print("=" * 70)

unconstrained_tree = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

unconstrained_tree.fit(
    X_train_prep,
    y_train
)

unconstrained_train_acc = (
    unconstrained_tree.score(
        X_train_prep,
        y_train
    )
)

unconstrained_test_acc = (
    unconstrained_tree.score(
        X_test_prep,
        y_test
    )
)

unconstrained_depth = (
    unconstrained_tree.get_depth()
)

print(
    f"Train Accuracy: "
    f"{unconstrained_train_acc:.4f}"
)

print(
    f"Test Accuracy : "
    f"{unconstrained_test_acc:.4f}"
)

print(
    f"Tree Depth    : "
    f"{unconstrained_depth}"
)


# ============================================================
# STEP 8 - CONSTRAINED TREE
# ============================================================

print("\n" + "=" * 70)
print("CONSTRAINED DECISION TREE")
print("=" * 70)

constrained_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

constrained_tree.fit(
    X_train_prep,
    y_train
)

constrained_train_acc = (
    constrained_tree.score(
        X_train_prep,
        y_train
    )
)

constrained_test_acc = (
    constrained_tree.score(
        X_test_prep,
        y_test
    )
)

constrained_depth = (
    constrained_tree.get_depth()
)


print(
    f"Train Accuracy: "
    f"{constrained_train_acc:.4f}"
)

print(
    f"Test Accuracy : "
    f"{constrained_test_acc:.4f}"
)

print(
    f"Tree Depth    : "
    f"{constrained_depth}"
)


# ============================================================
# STEP 9 - CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

constrained_pred = constrained_tree.predict(
    X_test_prep
)

print(
    classification_report(
        y_test,
        constrained_pred,
        target_names=[
            "No Deposit",
            "Subscribed"
        ],
        zero_division=0
    )
)


# ============================================================
# STEP 10 - COST COMPLEXITY PRUNING PATH
# ============================================================

print("\n" + "=" * 70)
print("COST COMPLEXITY PRUNING")
print("=" * 70)

path = unconstrained_tree.cost_complexity_pruning_path(
    X_train_prep,
    y_train
)

ccp_alphas = path.ccp_alphas
impurities = path.impurities


print(
    f"Number of candidate alphas: "
    f"{len(ccp_alphas)}"
)


# Avoid extremely tiny numerical alpha values
candidate_alphas = np.unique(
    ccp_alphas
)

pruned_results = []

for alpha in candidate_alphas:

    tree = DecisionTreeClassifier(
        criterion="gini",
        ccp_alpha=float(alpha),
        random_state=42
    )

    tree.fit(
        X_train_prep,
        y_train
    )

    train_accuracy = tree.score(
        X_train_prep,
        y_train
    )

    test_accuracy = tree.score(
        X_test_prep,
        y_test
    )

    depth = tree.get_depth()

    pruned_results.append(
        {
            "ccp_alpha": alpha,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "depth": depth
        }
    )


pruning_df = pd.DataFrame(
    pruned_results
)

pruning_df.to_csv(
    OUTPUT_DIR / "pruning_results.csv",
    index=False
)


# ============================================================
# STEP 11 - BEST PRUNED TREE
# Prefer accuracy, then smaller depth
# ============================================================

best_pruned_row = (
    pruning_df
    .sort_values(
        by=[
            "test_accuracy",
            "depth"
        ],
        ascending=[
            False,
            True
        ]
    )
    .iloc[0]
)


best_alpha = float(
    best_pruned_row["ccp_alpha"]
)

best_pruned_tree = DecisionTreeClassifier(
    criterion="gini",
    ccp_alpha=best_alpha,
    random_state=42
)

best_pruned_tree.fit(
    X_train_prep,
    y_train
)

best_pruned_train_acc = (
    best_pruned_tree.score(
        X_train_prep,
        y_train
    )
)

best_pruned_test_acc = (
    best_pruned_tree.score(
        X_test_prep,
        y_test
    )
)

best_pruned_depth = (
    best_pruned_tree.get_depth()
)


print(
    f"\nBest ccp_alpha : {best_alpha:.8f}"
)

print(
    f"Train Accuracy : "
    f"{best_pruned_train_acc:.4f}"
)

print(
    f"Test Accuracy  : "
    f"{best_pruned_test_acc:.4f}"
)

print(
    f"Tree Depth     : "
    f"{best_pruned_depth}"
)


# ============================================================
# STEP 12 - PRUNING CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    pruning_df["depth"],
    pruning_df["test_accuracy"],
    marker="o"
)

plt.xlabel(
    "Tree Depth"
)

plt.ylabel(
    "Test Accuracy"
)

plt.title(
    "Decision Tree Pruning: Depth vs Test Accuracy"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pruning_curve.png",
    dpi=150
)

plt.close()


# ============================================================
# STEP 13 - EXTRACT BUSINESS RULES
# ============================================================

rules = export_text(
    constrained_tree,
    feature_names=list(feature_names),
    max_depth=3
)

print("\n" + "=" * 70)
print("TOP DECISION RULES")
print("=" * 70)

print(rules)


with open(
    OUTPUT_DIR / "top_decision_rules.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Decision Tree Rules\n"
        "===================\n\n"
    )

    file.write(rules)


# ============================================================
# STEP 14 - SAVE TREE VISUALIZATION
# ============================================================

plt.figure(
    figsize=(20, 12)
)

plot_tree(
    constrained_tree,
    feature_names=feature_names,
    class_names=[
        "No Deposit",
        "Subscribed"
    ],
    filled=False,
    rounded=True,
    max_depth=3,
    fontsize=7
)

plt.title(
    "Constrained Decision Tree"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "decision_tree.png",
    dpi=150
)

plt.close()


# ============================================================
# STEP 15 - FINAL ACCEPTANCE CHECK
# ============================================================

print("\n" + "=" * 70)
print("EXERCISE 3 ACCEPTANCE CHECK")
print("=" * 70)

accuracy_pass = constrained_test_acc >= 0.85
depth_pass = constrained_depth <= 5

print(
    f"Test Accuracy >= 85% : "
    f"{'PASS' if accuracy_pass else 'CHECK'}"
)

print(
    f"Tree Depth <= 5       : "
    f"{'PASS' if depth_pass else 'CHECK'}"
)

print(
    f"Final Test Accuracy   : "
    f"{constrained_test_acc:.4f}"
)

print(
    f"Final Tree Depth      : "
    f"{constrained_depth}"
)


# ============================================================
# OUTPUT FILES
# ============================================================

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(
        f" - {file.name}"
    )

print("\nExercise 3 completed.")
