from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
BACKUP = ROOT / "data" / "backup_before_cleaning"

def main():
    CLEAN.mkdir(parents=True, exist_ok=True)

    # Preserve the current CSV files before replacing them.
    BACKUP.mkdir(parents=True, exist_ok=True)
    for name in ["customers.csv", "products.csv", "orders.csv"]:
        current = CLEAN / name
        if current.exists():
            shutil.copy2(current, BACKUP / name)

    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv")

    # Clean customers.
    customers["city"] = customers["city"].astype("string").str.strip().str.title()
    customers["gender"] = (
    customers["gender"]
    .astype("string")
    .str.strip()
    .str.lower()
    .replace({
        "m": "male",
        "male": "male",
        "f": "female",
        "female": "female",
    })
    .map({
        "male": "Male",
        "female": "Female",
    })
    .fillna("Unknown")
    )
    customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
    customers.loc[~customers["age"].between(10, 100), "age"] = pd.NA
    city_medians = customers.groupby("city")["age"].transform("median")
    customers["age"] = customers["age"].fillna(city_medians)
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # Clean products.
    products["product_name"] = products["product_name"].astype("string").str.strip()
    products["category"] = products["category"].astype("string").str.strip().str.title()
    products["price_inr"] = pd.to_numeric(products["price_inr"], errors="coerce")

    # Clean orders.
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
    orders = orders.drop_duplicates()
    orders = orders.sort_values(["order_ts", "order_id"])
    orders = orders.drop_duplicates("order_id", keep="last")
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
    orders["discount_pct"] = pd.to_numeric(orders["discount_pct"], errors="coerce")
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA

    orders = orders[orders["status"].str.lower() != "pending"].copy()
    orders["status"] = orders["status"].astype("string").str.strip().str.lower()

    # Fill missing amounts from product price, quantity, and discount.
    price_map = products.set_index("product_id")["price_inr"]
    estimated = (
        orders["product_id"].map(price_map)
        * orders["quantity"]
        * (1 - orders["discount_pct"].fillna(0) / 100)
    )
    orders["order_amount"] = orders["order_amount"].fillna(estimated)

    # Fill missing delivery time for completed orders using category median.
    category_map = products.set_index("product_id")["category"]
    orders["category"] = orders["product_id"].map(category_map)
    valid_completed = orders[
        (orders["status"] == "completed")
        & orders["delivery_minutes"].notna()
    ]
    category_medians = valid_completed.groupby("category")["delivery_minutes"].median()
    missing_completed = (
        (orders["status"] == "completed")
        & orders["delivery_minutes"].isna()
    )
    orders.loc[missing_completed, "delivery_minutes"] = (
        orders.loc[missing_completed, "category"].map(category_medians)
    )
    orders = orders.drop(columns=["category"])

    # Save cleaned CSVs.
    customers.to_csv(CLEAN / "customers.csv", index=False)
    products.to_csv(CLEAN / "products.csv", index=False)
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    orders.to_csv(CLEAN / "orders.csv", index=False)

    print("Cleaned customers:", len(customers))
    print("Cleaned products:", len(products))
    print("Cleaned orders:", len(orders))
    print("Backup saved to:", BACKUP)

if __name__ == "__main__":
    main()