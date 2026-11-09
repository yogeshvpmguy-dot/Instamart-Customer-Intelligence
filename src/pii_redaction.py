import re


def redact_pii(text):
    """Redact email, phone number, and order ID from customer text."""
    if not isinstance(text, str):
        return ""

    # 1. Email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]",
        text
    )

    # 2. Indian phone numbers
    text = re.sub(
        r"(?<!\w)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\w)",
        "[PHONE]",
        text
    )

    # 3. Instamart order IDs
    text = re.sub(
        r"\bORD[-_ ]?\d{4,}\b",
        "[ORDER_ID]",
        text,
        flags=re.IGNORECASE
    )

    return text


if __name__ == "__main__":
    sample = (
        "Contact me at user@gmail.com, "
        "phone 9876543210, order ORD-100048"
    )

    print("Original:", sample)
    print("Redacted:", redact_pii(sample))
