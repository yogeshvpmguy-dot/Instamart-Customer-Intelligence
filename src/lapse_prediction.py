import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load data
customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# April-August: features only
features_orders = orders[
    (orders["order_ts"] < "2025-09-01")
].copy()

# September: label only
september_orders = orders[
    (orders["order_ts"].dt.month == 9) &
    (orders["status"] == "completed")
]

active_ids = set(september_orders["customer_id"])

customers["lapse"] = (
    ~customers["customer_id"].isin(active_ids)
).astype(int)

# Build customer features using April-August only
features = features_orders.groupby("customer_id").agg(
    total_orders=("order_id", "count"),
    completed_orders=("status", lambda x: (x == "completed").sum()),
    cancelled_orders=("status", lambda x: (x == "cancelled").sum()),
    total_spend=("order_amount", "sum"),
    last_order=("order_ts", "max")
).reset_index()

features["avg_order_value"] = (
    features["total_spend"] /
    features["completed_orders"].replace(0, 1)
)

features["cancellation_rate_pct"] = (
    features["cancelled_orders"] /
    features["total_orders"].replace(0, 1) * 100
)

# Recency measured at August 31
cutoff = pd.Timestamp("2025-08-31")
features["recency_days"] = (
    cutoff - features["last_order"]
).dt.days

# Merge all customers and labels
data = customers[["customer_id", "lapse"]].merge(
    features, on="customer_id", how="left"
)

# Handle customers with no earlier orders
data["recency_days"] = data["recency_days"].fillna(153)

feature_columns = [
    "total_orders",
    "completed_orders",
    "cancelled_orders",
    "total_spend",
    "avg_order_value",
    "cancellation_rate_pct",
    "recency_days"
]

data[feature_columns] = data[feature_columns].fillna(0)

X = data[feature_columns]
y = data["lapse"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Logistic Regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Active", "Lapse"],
    zero_division=0
))

# Predict lapse risk for all customers
data["lapse_probability"] = model.predict_proba(X)[:, 1]

data.to_csv("outputs/lapse_predictions.csv", index=False)

print("\nSaved: outputs/lapse_predictions.csv")