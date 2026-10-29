import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_FILE = ROOT / "data" / "raw" / "agent_scenarios.csv"
ORDERS_FILE = ROOT / "data" / "clean" / "orders.csv"
OUTPUT_FILE = ROOT / "outputs" / "refund_decisions.csv"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def decide(scenario, orders_by_id):
    message = scenario["message"]
    expected = scenario["expected_outcome"]

    # Safety guard: do not repeatedly execute loop-like requests.
    if "[LOOP]" in message.upper():
        return "FALLBACK", 0.0, "Loop marker detected; human review required"

    match = re.search(r"ORD-\d+", message, flags=re.IGNORECASE)
    if not match:
        return "NOT_FOUND", 0.0, "No order ID found in the message"

    order_id = match.group(0).upper()
    order = orders_by_id.get(order_id)

    if order is None:
        return "NOT_FOUND", 0.0, "Order ID does not exist"

    if order["customer_id"] != scenario["customer_id"]:
        return "ESCALATE", 0.0, "Customer and order ownership do not match"

    if order["status"].strip().lower() != "completed":
        return "DENIED", 0.0, "Only completed orders are eligible for refunds"

    try:
        delivery_minutes = float(order["delivery_minutes"])
        amount = float(order["order_amount"])
    except (ValueError, TypeError):
        return "ESCALATE", 0.0, "Missing or invalid order data; manual review required"

    if delivery_minutes <= 45:
        return "DENIED", 0.0, "Delivery was 45 minutes or less"

    if amount < 0:
        return "ESCALATE", 0.0, "Invalid refund amount"

    if amount > 1000:
        return "ESCALATE", amount, "Human support lead approval required"

    return "REFUND", amount, (
        "Eligible under the late-delivery rule; "
        "7-day request window still needs verification"
    )


def main():
    scenarios = read_csv(SCENARIOS_FILE)
    orders = read_csv(ORDERS_FILE)
    orders_by_id = {row["order_id"].upper(): row for row in orders}

    results = []

    for scenario in scenarios:
        outcome, amount, reason = decide(scenario, orders_by_id)

        results.append({
            "scenario_id": scenario["scenario_id"],
            "customer_id": scenario["customer_id"],
            "expected_outcome": scenario["expected_outcome"],
            "predicted_outcome": outcome,
            "predicted_refund_inr": f"{amount:.2f}",
            "reason": reason,
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    correct = sum(
        row["expected_outcome"] == row["predicted_outcome"]
        for row in results
    )

    print(f"Scenarios tested: {len(results)}")
    print(f"Outcome matches: {correct}/{len(results)}")
    print("\nDecision details:")

    for row in results:
        print(
            f"{row['scenario_id']}: "
            f"Expected={row['expected_outcome']} | "
            f"Predicted={row['predicted_outcome']} | "
            f"Refund=INR {row['predicted_refund_inr']}"
        )
        print(f"  Reason: {row['reason']}")

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
