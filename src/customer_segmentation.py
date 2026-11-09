import pandas as pd
import numpy as np

df = pd.read_csv("outputs/customer_features.csv")

# RFM features
X = df[
    ["recency_days", "completed_orders", "total_spend"]
].to_numpy(dtype=float)

# Standardize
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1
X = (X - mean) / std

# NumPy KMeans with multiple initializations
k = 4
best_labels = None
best_inertia = float("inf")

for seed in range(10):
    rng = np.random.default_rng(seed)

    centroids = X[
        rng.choice(len(X), k, replace=False)
    ].copy()

    for iteration in range(300):
        distances = np.sum(
            (X[:, None, :] - centroids[None, :, :]) ** 2,
            axis=2
        )

        labels = distances.argmin(axis=1)

        new_centroids = centroids.copy()

        for i in range(k):
            points = X[labels == i]

            if len(points) > 0:
                new_centroids[i] = points.mean(axis=0)
            else:
                # Reinitialize empty cluster
                farthest = distances.min(axis=1).argmax()
                new_centroids[i] = X[farthest]

        if np.allclose(centroids, new_centroids):
            centroids = new_centroids
            break

        centroids = new_centroids

    # Recalculate labels using final centroids
    distances = np.sum(
        (X[:, None, :] - centroids[None, :, :]) ** 2,
        axis=2
    )
    labels = distances.argmin(axis=1)

    inertia = np.sum(
        (X - centroids[labels]) ** 2
    )

    if inertia < best_inertia:
        best_inertia = inertia
        best_labels = labels.copy()

df["segment"] = best_labels

df.to_csv("outputs/customer_segments.csv", index=False)

print("\nCustomer Segment Summary:")

summary = df.groupby("segment").agg(
    customers=("customer_id", "count"),
    avg_recency=("recency_days", "mean"),
    avg_orders=("completed_orders", "mean"),
    avg_spend=("total_spend", "mean")
)

print(summary.round(2))
print("\nSaved: outputs/customer_segments.csv")
