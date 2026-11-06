import pandas as pd

# Load lapse predictions
df = pd.read_csv("outputs/lapse_predictions.csv")

# Select high-risk customers
high_risk = df[df["lapse_probability"] >= 0.70].copy()

# Sort highest risk first
high_risk = high_risk.sort_values(
    "lapse_probability",
    ascending=False
)

# Save retention list
high_risk.to_csv(
    "outputs/high_risk_customers.csv",
    index=False
)

print("High-risk customers:", len(high_risk))
print("\nTop 10 high-risk customers:")
print(
    high_risk[
        ["customer_id", "lapse_probability", "total_spend"]
    ].head(10).round(3)
)

print("\nSaved: outputs/high_risk_customers.csv")