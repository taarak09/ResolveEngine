
import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from google import genai


MODEL_NAME = "gemini-3.6-flash"


def load_retrieval_index(index_dir: str):
    index_dir = Path(index_dir)
    vectorizer = joblib.load(index_dir / "tfidf_vectorizer.joblib")
    matrix = load_npz(index_dir / "tfidf_matrix.npz")
    metadata = pd.read_csv(index_dir / "retrieval_metadata.csv")
    return vectorizer, matrix, metadata


def retrieve(query, vectorizer, matrix, metadata, top_k):
    query_vector = vectorizer.transform([query])
    scores = np.asarray((matrix @ query_vector.T).toarray()).ravel()

    top_k = min(top_k, len(metadata))
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "similarity": float(scores[idx]),
            "customer": str(metadata.iloc[idx]["customer_text_clean"]),
            "support": str(metadata.iloc[idx]["support_text_clean"]),
        }
        for idx in top_indices
    ]


def build_prompt(customer_message, intent, cases):
    evidence = "\n\n".join(
        f"""HISTORICAL CASE {i}
Similarity: {case['similarity']:.4f}
Customer:
{case['customer']}
AppleSupport response:
{case['support']}"""
        for i, case in enumerate(cases, 1)
    )

    return f"""You are an AI customer-support assistant for AppleSupport.

CURRENT CUSTOMER MESSAGE:
{customer_message}

PREDICTED INTENT:
{intent}

HISTORICAL APPLESUPPORT EVIDENCE:
{evidence}

Instructions:
- Use the historical cases as the primary evidence.
- Do not invent Apple policies, prices, timelines, warranties, refunds,
  account actions, or troubleshooting steps.
- Do not claim an action was taken unless supported by the evidence.
- If evidence is weak or insufficient, prefer escalation.
- Be concise and professional.
- Do not mention that you are an AI.

Return exactly:
DECISION: AUTO-HANDLE or ESCALATE
REASON: one concise sentence
DRAFT_REPLY: one customer-facing reply
"""


def generate_response(message, intent, cases):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)
    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=build_prompt(message, intent, cases),
    )
    return interaction.output_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", required=True)
    parser.add_argument("--intent", default="UNKNOWN")
    parser.add_argument("--index-dir", default="data/retrieval_index")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("Loading retrieval index...")
    vectorizer, matrix, metadata = load_retrieval_index(args.index_dir)

    print("Retrieving historical cases...")
    cases = retrieve(args.message, vectorizer, matrix, metadata, args.top_k)

    print("\n" + "=" * 70)
    print("HISTORICAL EVIDENCE")
    print("=" * 70)

    for i, case in enumerate(cases, 1):
        print(f"\n--- Case {i} | Similarity {case['similarity']:.4f} ---")
        print("Customer:")
        print(case["customer"])
        print("\nAppleSupport:")
        print(case["support"])

    print("\n" + "=" * 70)
    print("GEMINI RESPONSE")
    print("=" * 70)
    print(generate_response(args.message, args.intent, cases))


if __name__ == "__main__":
    main()
