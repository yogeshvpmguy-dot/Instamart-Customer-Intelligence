import csv
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs"

sys.path.insert(0, str(ROOT))
import mock_llm


def load_policy_chunks():
    chunks = []

    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")

        # Split policy into sentences
        sentences = [
            sentence.strip()
            for sentence in text.split(".")
            if sentence.strip()
        ]

        # Create chunks with one-sentence overlap
        for i in range(0, len(sentences), 2):
            chunk_sentences = sentences[i:i + 3]
            chunk = ". ".join(chunk_sentences) + "."

            chunks.append({
                "source": path.name,
                "text": chunk
            })

    return chunks


def build_rag(chunks):
    documents = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(documents)

    return vectorizer, matrix


def retrieve(question, chunks, vectorizer, matrix, top_k=3):
    query_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in top_indices:
        if scores[index] > 0:
            results.append(chunks[index])

    return results


def ask_policy(question, chunks, vectorizer, matrix):
    retrieved = retrieve(
        question,
        chunks,
        vectorizer,
        matrix
    )

    context = "\n".join(
        f"[{item['source']}] {item['text']}"
        for item in retrieved
    )

    messages = [
        {
            "role": "system",
            "content": "POLICY ASSISTANT: Answer using only the supplied policy context."
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"
        }
    ]

    response = mock_llm.chat(messages)

    return response.choices[0].message.content, retrieved


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    chunks = load_policy_chunks()

    if not chunks:
        raise RuntimeError("No policy documents found.")

    vectorizer, matrix = build_rag(chunks)

    with open(EVAL_FILE, newline="", encoding="utf-8") as file:
        evaluations = list(csv.DictReader(file))

    results = []
    correct = 0

    print("POLICY RAG ASSISTANT")
    print("-" * 45)

    for item in evaluations:
        question = item["question"]
        expected_fact = item["expected_fact"]

        answer, retrieved = ask_policy(
            question,
            chunks,
            vectorizer,
            matrix
        )

        fact_found = (
            expected_fact.lower() in answer.lower()
        )

        if fact_found:
            correct += 1

        results.append({
            "question": question,
            "expected_doc": item["expected_doc"],
            "expected_fact": expected_fact,
            "answer": answer,
            "retrieved_docs": ";".join(
                sorted({r["source"] for r in retrieved})
            ),
            "fact_found": fact_found
        })

        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")
        print(f"Expected fact found: {fact_found}")

    output_file = OUTPUT_DIR / "policy_rag_evaluation.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    accuracy = correct / len(evaluations) * 100

    print("\n" + "-" * 45)
    print(f"Policy evaluation: {correct}/{len(evaluations)}")
    print(f"Expected fact match: {accuracy:.2f}%")
    print(f"Saved results: {output_file}")


if __name__ == "__main__":
    main()