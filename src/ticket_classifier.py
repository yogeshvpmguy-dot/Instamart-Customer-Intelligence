import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent.parent
TICKETS_FILE = ROOT / "data" / "clean" / "tickets.csv"
OUTPUT_FILE = ROOT / "outputs" / "ticket_predictions.csv"


def main():
    with TICKETS_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    # Keep only rows with both a message and a valid category.
    valid_rows = [
        row for row in rows
        if row.get("text", "").strip()
        and row.get("true_category", "").strip().lower()
        in {"delivery", "fraud", "billing", "technical"}
    ]

    texts = [row["text"].strip() for row in valid_rows]
    labels = [row["true_category"].strip().lower() for row in valid_rows]

    if len(set(labels)) < 2:
        raise ValueError("At least two categories are required for training.")

    print(f"Total ticket rows: {len(rows)}")
    print(f"Usable labeled tickets: {len(valid_rows)}")
    print("\nCategory counts:")

    for category in sorted(set(labels)):
        print(f"  {category}: {labels.count(category)}")

    # A stratified split keeps category proportions similar in each set.
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        texts,
        labels,
        list(range(len(valid_rows))),
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\nTraining tickets: {len(X_train)}")
    print(f"Test tickets: {len(X_test)}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # Save held-out predictions for review.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "ticket_id",
                "true_category",
                "predicted_category",
                "correct",
            ],
        )
        writer.writeheader()

        for idx, prediction in zip(test_idx, predictions):
            row = valid_rows[idx]
            writer.writerow({
                "ticket_id": row["ticket_id"],
                "true_category": row["true_category"],
                "predicted_category": prediction,
                "correct": str(
                    row["true_category"].strip().lower() == prediction
                ),
            })

    print(f"\nSaved test predictions to: {OUTPUT_FILE}")
    print("Note: This is a small-data baseline, not a production-ready model.")


if __name__ == "__main__":
    main()
