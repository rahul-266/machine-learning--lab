# ============================================================
# MACHINE LEARNING ENGINEERING LAB
# Exercise 5 - K-Means Clustering
# Python 3.12 + Scikit-Learn
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "wholesale_customers.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "exercise5"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXERCISE 5
# ============================================================

print("=" * 70)
print("EXERCISE 5 - K-MEANS CLUSTERING")
print("Wholesale Customer Spending Personas")
print("=" * 70)


# ============================================================
# STEP 1 - LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# STEP 2 - SELECT SPENDING FEATURES
# ============================================================

spending_cols = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]

X = df[spending_cols].copy()


# ============================================================
# STEP 3 - LOG1P TRANSFORMATION
# ============================================================

print("\n" + "=" * 70)
print("LOG1P TRANSFORMATION")
print("=" * 70)

X_log = np.log1p(X)

print("Log1p transformation completed.")


# ============================================================
# STEP 4 - STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_log)

print("Standardization completed.")


# ============================================================
# STEP 5 - ELBOW + SILHOUETTE
# k = 2 TO 8
# ============================================================

print("\n" + "=" * 70)
print("ELBOW AND SILHOUETTE ANALYSIS")
print("=" * 70)

k_values = range(2, 9)

inertias = []
silhouette_scores = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia = kmeans.inertia_

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    inertias.append(inertia)
    silhouette_scores.append(silhouette)

    print(
        f"k={k} | "
        f"Inertia={inertia:.2f} | "
        f"Silhouette={silhouette:.4f}"
    )


# ============================================================
# SAVE ANALYSIS
# ============================================================

k_results = pd.DataFrame(
    {
        "k": list(k_values),
        "inertia": inertias,
        "silhouette": silhouette_scores
    }
)

k_results.to_csv(
    OUTPUT_DIR / "k_analysis.csv",
    index=False
)


# ============================================================
# STEP 6 - ELBOW CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method - Wholesale Customers")
plt.xticks(list(k_values))

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "elbow_curve.png",
    dpi=150
)

plt.close()


# ============================================================
# STEP 7 - SILHOUETTE CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.axhline(
    0.42,
    linestyle="--",
    label="Target = 0.42"
)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis - Wholesale Customers")
plt.xticks(list(k_values))
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "silhouette_curve.png",
    dpi=150
)

plt.close()


# ============================================================
# STEP 8 - BEST K BY SILHOUETTE
# ============================================================

best_index = int(
    np.argmax(silhouette_scores)
)

best_k = list(k_values)[best_index]

best_silhouette = silhouette_scores[best_index]

print("\n" + "=" * 70)
print("BEST SILHOUETTE RESULT")
print("=" * 70)

print(f"Best k by silhouette : {best_k}")
print(f"Best silhouette      : {best_silhouette:.4f}")


# ============================================================
# STEP 9 - FINAL K = 4
# ============================================================

final_k = 4

print("\n" + "=" * 70)
print("FINAL K-MEANS MODEL")
print("=" * 70)

print(f"Selected final k: {final_k}")

final_kmeans = KMeans(
    n_clusters=final_k,
    random_state=42,
    n_init=10
)

cluster_labels = final_kmeans.fit_predict(
    X_scaled
)


# ============================================================
# STEP 10 - FINAL SILHOUETTE
# ============================================================

final_silhouette = silhouette_score(
    X_scaled,
    cluster_labels
)

print(
    f"Final k=4 silhouette: "
    f"{final_silhouette:.4f}"
)


# ============================================================
# STEP 11 - ASSIGN CLUSTERS
# ============================================================

clustered_df = df.copy()

clustered_df["Cluster"] = cluster_labels

clustered_df.to_csv(
    OUTPUT_DIR / "customer_clusters.csv",
    index=False
)


# ============================================================
# STEP 12 - CLUSTER CENTROIDS
# ORIGINAL SCALE
# ============================================================

centroid_df = (
    clustered_df
    .groupby("Cluster")[spending_cols]
    .mean()
    .round(2)
)

centroid_df.to_csv(
    OUTPUT_DIR / "cluster_centroids.csv"
)

print("\n" + "=" * 70)
print("CLUSTER CENTROIDS")
print("=" * 70)

print(centroid_df)


# ============================================================
# STEP 13 - CLUSTER SIZES
# ============================================================

cluster_counts = (
    clustered_df["Cluster"]
    .value_counts()
    .sort_index()
)

cluster_sizes = pd.DataFrame(
    {
        "Cluster": cluster_counts.index,
        "Customers": cluster_counts.values
    }
)

cluster_sizes.to_csv(
    OUTPUT_DIR / "cluster_sizes.csv",
    index=False
)

print("\nCluster sizes:")
print(
    cluster_sizes.to_string(index=False)
)


# ============================================================
# STEP 14 - BUSINESS PERSONAS
# ============================================================

overall_mean = X.mean()

persona_rows = []

for cluster_id, row in centroid_df.iterrows():

    relative_score = (
        row / overall_mean
    ).sort_values(
        ascending=False
    )

    top_features = (
        relative_score.index[:3].tolist()
    )

    top1 = top_features[0]
    top2 = top_features[1]

    if (
        top1 == "Grocery"
        and top2 in [
            "Detergents_Paper",
            "Milk"
        ]
    ):
        persona = "Supermarkets"

    elif top1 == "Delicassen":
        persona = "Boutique Delicatessens"

    elif (
        top1 == "Fresh"
        and top2 in [
            "Frozen",
            "Delicassen"
        ]
    ):
        persona = "Fresh Markets"

    else:
        persona = "Horeca / Restaurants"

    persona_rows.append(
        {
            "Cluster": cluster_id,
            "Persona": persona,
            "TopFeature1": top1,
            "TopFeature2": top2,
            "TopFeature3": top_features[2]
        }
    )


persona_df = pd.DataFrame(persona_rows)

persona_df.to_csv(
    OUTPUT_DIR / "cluster_personas.csv",
    index=False
)

print("\n" + "=" * 70)
print("BUSINESS PERSONAS")
print("=" * 70)

print(
    persona_df.to_string(index=False)
)


# ============================================================
# STEP 15 - CLUSTER PROFILE CHART
# ============================================================

plot_centroids = centroid_df.copy()

plot_centroids.index = (
    "Cluster "
    + plot_centroids.index.astype(str)
)

plot_centroids.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.xlabel("Customer Cluster")
plt.ylabel("Average Annual Spending")
plt.title("Wholesale Customer Cluster Profiles")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "cluster_profiles.png",
    dpi=150
)

plt.close()


# ============================================================
# STEP 16 - ACCEPTANCE CHECK
# ============================================================

print("\n" + "=" * 70)
print("EXERCISE 5 ACCEPTANCE CHECK")
print("=" * 70)

silhouette_pass = (
    final_silhouette >= 0.42
)

k_pass = (
    final_k in [3, 4]
)

print(
    f"Silhouette >= 0.42 : "
    f"{'PASS' if silhouette_pass else 'CHECK'}"
)

print(
    f"Selected k = 3/4   : "
    f"{'PASS' if k_pass else 'CHECK'}"
)

print(
    f"Final k             : {final_k}"
)

print(
    f"Final silhouette    : "
    f"{final_silhouette:.4f}"
)

print(
    f"Best k by silhouette: "
    f"{best_k}"
)

print(
    f"Best silhouette     : "
    f"{best_silhouette:.4f}"
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

print("\nExercise 5 completed.")
