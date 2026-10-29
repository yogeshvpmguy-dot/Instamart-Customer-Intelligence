import csv
import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent
scenario_path = root / "data" / "raw" / "agent_scenarios.csv"
order_path = root / "data" / "clean" / "orders.csv"

with scenario_path.open(encoding="utf-8-sig", newline="") as f:
    scenarios = list(csv.DictReader(f))

with order_path.open(encoding="utf-8-sig", newline="") as f:
    orders = {row["order_id"]: row for row in csv.DictReader(f)}

for scenario in scenarios:
    match = re.search(r"ORD-\d+", scenario["message"])
    order_id = match.group(0) if match else None
    order = orders.get(order_id) if order_id else None

    print(f"\n{scenario['scenario_id']} | {order_id} | Expected: {scenario['expected_outcome']}")
    if order:
        print(f"  Status: {order['status']}, Amount: {order['order_amount']}, Delivery: {order['delivery_minutes']} minutes")
        print(f"  Customer matches: {order['customer_id'] == scenario['customer_id']}")
    else:
        print("  Order not found in cleaned orders.")
