import csv
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
scenario_file = project_dir / "data" / "raw" / "agent_scenarios.csv"

with scenario_file.open("r", encoding="utf-8-sig", newline="") as file:
    scenarios = list(csv.DictReader(file))

print(f"Total scenarios: {len(scenarios)}")
print("-" * 45)

for scenario in scenarios:
    print(f"Scenario ID : {scenario['scenario_id']}")
    print(f"Customer ID : {scenario['customer_id']}")
    print(f"Message     : {scenario['message']}")
    print(f"Expected    : {scenario['expected_outcome']}")
    print(f"Refund INR  : {scenario['expected_refund_inr']}")
    print("-" * 45)
