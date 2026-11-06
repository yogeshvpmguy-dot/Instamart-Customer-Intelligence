import sqlite3
import pandas as pd
from pathlib import Path

print("Starting Part A validation...")

# Check cleaned CSV files
customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")
products = pd.read_csv("data/clean/products.csv")

# Check database
conn = sqlite3.connect("data/instamart.db")

db_customers = conn.execute(
    "SELECT COUNT(*) FROM customers"
).fetchone()[0]

db_orders = conn.execute(
    "SELECT COUNT(*) FROM orders"
).fetchone()[0]

db_products = conn.execute(
    "SELECT COUNT(*) FROM products"
).fetchone()[0]

checks = [
    ("Customers CSV = 1200", len(customers) == 1200),
    ("Products CSV = 40", len(products) == 40),
    ("Orders CSV = 12309", len(orders) == 12309),
    ("Database customers = 1200", db_customers == 1200),
    ("Database products = 40", db_products == 40),
    ("Database orders = 12309", db_orders == 12309),
    ("No duplicate order IDs", orders["order_id"].is_unique),
    ("No missing order amounts", orders["order_amount"].notna().all()),
]

for name, passed in checks:
    print(f"{'PASS' if passed else 'FAIL'}: {name}")

conn.close()

if all(passed for _, passed in checks):
    print("\nALL CHECKPOINTS PASSED")
else:
    print("\nSome checkpoints failed. Review the data.")