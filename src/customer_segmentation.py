import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load existing customer features
df = pd.read_csv("outputs/customer_features.csv")

# Select RFM features
rfm = df[
    ["recency_days", "completed_orders", "total_spend"]
].copy()

rfm.columns = ["Recency", "Frequency", "Monetary"]

# Handle customers with missing recency
rfm["Recency"] = rfm["Recency"].fillna(rfm["Recency"].max() + 1)

# Scale features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Create customer segments
model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["segment"] = model.fit_predict(rfm_scaled)

# Save segmentation results
df.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

# Display segment summary
summary = df.groupby("segment").agg(
    customers=("customer_id", "count"),
    avg_recency=("recency_days", "mean"),
    avg_orders=("completed_orders", "mean"),
    avg_spend=("total_spend", "mean")
)

print("\nCustomer Segment Summary:")
print(summary.round(2))

# Assign meaningful customer segment names
segment_names = {
    0: "High-Spending Customers",
    1: "Low-Spending Customers",
    2: "Frequent Customers",
    3: "Inactive Customers"
}

df["segment_name"] = df["segment"].map(segment_names)

# Save updated segmentation results
df.to_csv("outputs/customer_segments.csv", index=False)

print("\nCustomer Segment Names:")
print(df[["customer_id", "segment", "segment_name"]].head(10))

print("\nSaved: outputs/customer_segments.csv")