import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from pii_redaction import redact_pii

ROOT = Path(__file__).resolve().parent.parent
TICKETS_FILE = ROOT / "data" / "clean" / "tickets.csv"
OUTPUT_FILE = ROOT / "outputs" / "ticket_predictions.csv"

VALID_CATEGORIES = {"delivery", "fraud", "billing", "technical"}

# Experimental confidence threshold
CONFIDENCE_THRESHOLD = 0.60


def main():
    with TICKETS_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    valid_rows = [
        row for row in rows
        if (row.get("text") or "").strip()
        and (row.get("true_category") or "").strip().lower()
        in VALID_CATEGORIES
    ]

    texts = [redact_pii(row["text"].strip()) for row in valid_rows]
    labels = [row["true_category"].strip().lower() for row in valid_rows]

    print(f"Total ticket rows: {len(rows)}")
    print(f"Usable labeled tickets: {len(valid_rows)}")
    print("PII redaction applied before model training.")

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

    # Get prediction confidence
    probabilities = model.predict_proba(X_test)
    confidence_scores = probabilities.max(axis=1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "ticket_id",
                "true_category",
                "predicted_category",
                "confidence",
                "review_required",
                "correct",
            ],
        )
        writer.writeheader()

        for idx, prediction, confidence in zip(
            test_idx, predictions, confidence_scores
        ):
            row = valid_rows[idx]

            review_required = confidence < CONFIDENCE_THRESHOLD

            writer.writerow({
                "ticket_id": row["ticket_id"],
                "true_category": row["true_category"].strip().lower(),
                "predicted_category": (
                    "HUMAN_REVIEW" if review_required else prediction
                ),
                "confidence": round(float(confidence), 4),
                "review_required": str(review_required),
                "correct": str(
                    row["true_category"].strip().lower() == prediction
                ),
            })

    print(f"\nConfidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Saved predictions to: {OUTPUT_FILE}")
    print("Low-confidence tickets are flagged for human review.")
    print("Note: Small-data baseline; not production-ready.")


if __name__ == "__main__":
    main()