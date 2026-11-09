import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"
SCENARIOS_FILE = ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_FILE = ROOT / "outputs" / "refund_agent_ledger.jsonl"

sys.path.insert(0, str(ROOT))

import mock_llm
import policy_rag
from pii_redaction import redact_pii


def get_order(order_id, customer_id):
    """Look up order and verify customer ownership."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?",
            (order_id,)
        ).fetchone()

    if row is None:
        return {"error": "Order not found"}

    order = dict(row)

    if order["customer_id"] != customer_id:
        return {"error": "Customer ownership mismatch"}

    return order


def search_policy(query, chunks, vectorizer, matrix):
    results = policy_rag.retrieve(
        query, chunks, vectorizer, matrix, top_k=3
    )

    return {
        "results": [
            {"source": item["source"], "text": item["text"]}
            for item in results
        ]
    }


def calculate_refund(order_amount, delivery_minutes):
    """Calculate refund for eligible late delivery."""

    amount = float(order_amount)
    delivery = float(delivery_minutes)

    if amount < 0 or delivery < 0:
        return {"error": "Invalid order values"}

    if delivery <= 45:
        return {"refund_inr": 0.0}

    return {"refund_inr": round(amount, 2)}


def run_agent(scenario, chunks, vectorizer, matrix):
    customer_id = scenario["customer_id"]
    message = scenario["message"]

    # Preserve order ID for order lookup; redact other PII.
    order_match = re.search(r"ORD-\d{6}", message)
    order_id = order_match.group(0) if order_match else None

    safe_message = redact_pii(message)

    if order_id:
        safe_message = safe_message.replace(
            "[ORDER_ID]", order_id, 1
        )

    mock_llm.reset()

    messages = [
        {
            "role": "system",
            "content": (
                "AGENT: You are an Instamart refund support agent. "
                "Use tools to look up orders and process refunds."
            )
        },
        {"role": "user", "content": safe_message}
    ]

    ledger = []
    final_answer = None
    verified_order = None
    calculated_refund = None

    for step in range(8):
        try:
            response = mock_llm.chat(messages)
        except Exception as error:
            final_answer = (
                "FALLBACK: Agent stopped safely; human review required. "
                f"Reason: {type(error).__name__}"
            )
            break

        assistant_message = response.choices[0].message

        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": assistant_message.tool_calls
        })

        if not assistant_message.tool_calls:
            final_answer = assistant_message.content
            break

        for call in assistant_message.tool_calls:
            tool_name = call.name
            arguments = call.arguments

            ledger_entry = {
                "scenario_id": scenario["scenario_id"],
                "step": step + 1,
                "tool": tool_name,
                "arguments": arguments
            }

            if tool_name == "get_order":
                result = get_order(
                    arguments["order_id"],
                    customer_id
                )

                if "error" not in result:
                    verified_order = result

            elif tool_name == "search_policy":
                result = search_policy(
                    arguments.get("query", ""),
                    chunks,
                    vectorizer,
                    matrix
                )

            elif tool_name == "calculate_refund":
                if verified_order is None:
                    result = {"error": "Order verification required"}

                elif verified_order["status"] != "completed":
                    result = {"error": "Only completed orders are eligible"}

                else:
                    # Use verified database values, not LLM values.
                    result = calculate_refund(
                        verified_order["order_amount"],
                        verified_order["delivery_minutes"]
                    )

                    if "refund_inr" in result:
                        calculated_refund = result["refund_inr"]

            elif tool_name == "issue_refund":
                requested_amount = float(
                    arguments.get("refund_inr", -1)
                )

                if verified_order is None:
                    final_answer = (
                        "DENIED: verified order required."
                    )

                elif verified_order["status"] != "completed":
                    final_answer = (
                        "DENIED: only completed orders can be refunded."
                    )

                elif calculated_refund is None:
                    final_answer = (
                        "DENIED: refund calculation required."
                    )

                elif calculated_refund <= 0:
                    final_answer = (
                        "DENIED: delivery was within 45 minutes."
                    )

                elif calculated_refund > 1000:
                    final_answer = (
                        "ESCALATE: refund above INR 1000 "
                        "requires human lead approval."
                    )

                elif abs(requested_amount - calculated_refund) > 0.01:
                    final_answer = (
                        "DENIED: requested refund amount "
                        "does not match verified calculation."
                    )

                else:
                    result = {
                        "status": "issued",
                        "refund_inr": calculated_refund
                    }

            elif tool_name == "escalate_to_human":
                requested_amount = float(
                    arguments.get("refund_inr", -1)
                )

                if verified_order is None:
                    final_answer = (
                        "DENIED: verified order required."
                    )

                elif calculated_refund is None:
                    final_answer = (
                        "DENIED: refund calculation required."
                    )

                elif calculated_refund <= 1000:
                    final_answer = (
                        "DENIED: human escalation rule "
                        "applies only above INR 1000."
                    )

                elif abs(requested_amount - calculated_refund) > 0.01:
                    final_answer = (
                        "DENIED: escalation amount mismatch."
                    )

                else:
                    result = {
                        "status": "escalated",
                        "refund_inr": calculated_refund,
                        "reason": "Human support lead approval required"
                    }

            else:
                result = {
                    "error": "Unknown tool",
                    "tool_name": tool_name
                }

            # Stop immediately if a safety validation failed.
            if final_answer is not None:
                ledger_entry["result"] = {
                    "status": "blocked",
                    "reason": final_answer
                }
                ledger.append(ledger_entry)
                break

            ledger_entry["result"] = result
            ledger.append(ledger_entry)

            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result),
                "tool_call_id": call.id
            })

        if final_answer is not None:
            break

    if final_answer is None:
        final_answer = (
            "FALLBACK: Maximum agent steps reached; human review required."
        )

    return final_answer, ledger


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    chunks = policy_rag.load_policy_chunks()
    vectorizer, matrix = policy_rag.build_rag(chunks)

    with open(
        SCENARIOS_FILE,
        newline="",
        encoding="utf-8-sig"
    ) as file:
        scenarios = list(csv.DictReader(file))

    total = 0
    matches = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as ledger_file:
        print("INSTAMART REFUND AGENT")
        print("-" * 45)

        for scenario in scenarios:
            answer, ledger = run_agent(
                scenario, chunks, vectorizer, matrix
            )

            expected = scenario["expected_outcome"]
            predicted = (
                answer.split(":")[0]
                if ":" in answer
                else "FALLBACK"
            )

            total += 1

            if predicted == expected:
                matches += 1

            print(
                f"{scenario['scenario_id']} | "
                f"Expected={expected} | Predicted={predicted}"
            )
            print(f"  Agent response: {answer}")

            for entry in ledger:
                ledger_file.write(json.dumps(entry) + "\n")

            ledger_file.write(json.dumps({
                "scenario_id": scenario["scenario_id"],
                "final_answer": answer,
                "expected_outcome": expected,
                "predicted_outcome": predicted
            }) + "\n")

    print("\n" + "-" * 45)
    print(f"Scenarios tested: {total}")
    print(f"Outcome matches: {matches}/{total}")
    print(f"Ledger saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()