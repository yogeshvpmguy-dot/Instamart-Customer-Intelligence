import csv
import json
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


def get_order(order_id, customer_id):
    """Look up an order in SQLite and verify ownership."""

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
    """Retrieve relevant policy chunks."""

    results = policy_rag.retrieve(
        query,
        chunks,
        vectorizer,
        matrix,
        top_k=3
    )

    return {
        "results": [
            {
                "source": item["source"],
                "text": item["text"]
            }
            for item in results
        ]
    }


def calculate_refund(order_amount, delivery_minutes):
    """Calculate refund using the late-delivery rule."""

    amount = float(order_amount)
    delivery = float(delivery_minutes)

    if delivery <= 45:
        return {"refund_inr": 0.0}

    return {"refund_inr": amount}


def run_agent(scenario, chunks, vectorizer, matrix):
    """Run mock LLM agent with tool calls."""

    customer_id = scenario["customer_id"]
    message = scenario["message"]

    # Reset mock LLM for each independent scenario.
    mock_llm.reset()

    messages = [
        {
            "role": "system",
            "content": (
                "AGENT: You are an Instamart refund support agent. "
                "Use tools to look up orders and process refunds."
            )
        },
        {
            "role": "user",
            "content": message
        }
    ]

    ledger = []
    final_answer = None

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

        # Agent returned a final answer.
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

            # Execute requested tool.
            if tool_name == "get_order":
                result = get_order(
                    arguments["order_id"],
                    customer_id
                )

            elif tool_name == "search_policy":
                result = search_policy(
                    arguments.get("query", ""),
                    chunks,
                    vectorizer,
                    matrix
                )

            elif tool_name == "calculate_refund":
                result = calculate_refund(
                    arguments["order_amount"],
                    arguments["delivery_minutes"]
                )

            elif tool_name == "issue_refund":
                result = {
                    "status": "issued",
                    "refund_inr": arguments["refund_inr"]
                }

            elif tool_name == "escalate_to_human":
                result = {
                    "status": "escalated",
                    "refund_inr": arguments["refund_inr"],
                    "reason": "Human support lead approval required"
                }

            else:
                result = {
                    "error": "Unknown tool",
                    "tool_name": tool_name
                }

            ledger_entry["result"] = result
            ledger.append(ledger_entry)

            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result),
                "tool_call_id": call.id
            })

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

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as ledger_file:

        print("INSTAMART REFUND AGENT")
        print("-" * 45)

        for scenario in scenarios:
            answer, ledger = run_agent(
                scenario,
                chunks,
                vectorizer,
                matrix
            )

            expected = scenario["expected_outcome"]

            # Extract outcome from final answer.
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
                f"Expected={expected} | "
                f"Predicted={predicted}"
            )
            print(f"  Agent response: {answer}")

            for entry in ledger:
                ledger_file.write(
                    json.dumps(entry) + "\n"
                )

            # Save final scenario outcome.
            ledger_file.write(
                json.dumps({
                    "scenario_id": scenario["scenario_id"],
                    "final_answer": answer,
                    "expected_outcome": expected,
                    "predicted_outcome": predicted
                }) + "\n"
            )

    print("\n" + "-" * 45)
    print(f"Scenarios tested: {total}")
    print(f"Outcome matches: {matches}/{total}")
    print(f"Ledger saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()