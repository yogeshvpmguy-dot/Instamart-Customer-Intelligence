import csv
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pii_redaction import redact_pii

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "clean" / "tickets.csv"

texts = []
categories = []

# Read ticket data
with open(DATA_PATH, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        message = (row.get("text") or "").strip()
        category = (row.get("true_category") or "").strip().lower()

        if message and category in [
            "delivery", "fraud", "billing", "technical"
        ]:
            # Redact PII before using ticket text
            message = redact_pii(message)

            texts.append(message)
            categories.append(category)

print("Usable labeled tickets:", len(texts))
print("PII redaction applied before model training.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    categories,
    test_size=0.25,
    random_state=42,
    stratify=categories
)

# Convert text into numerical features
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=3000
)

X_train_vectorized = vectorizer.fit_transform(X_train)

# Train classifier
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_vectorized, y_train)


def classify_ticket(message):
    # Redact PII before prediction
    safe_message = redact_pii(message)

    message_vectorized = vectorizer.transform([safe_message])
    return model.predict(message_vectorized)[0]


# Test messages
test_messages = [
    "My order is very late",
    "I was charged twice for one order",
    "Someone made a payment without my permission",
    "The app keeps crashing",
    "Contact me at user@gmail.com, phone 9876543210, order ORD-100048"
]

print("\nINSTAMART SUPPORT TICKET CLASSIFIER")
print("-" * 40)

for message in test_messages:
    category = classify_ticket(message)

    # Print only redacted message
    safe_message = redact_pii(message)

    print("Customer:", safe_message)
    print("Predicted category:", category)
    print("-" * 40)