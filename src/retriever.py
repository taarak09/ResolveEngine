import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer


def build_index(input_file, output_dir, golden_file=None):
    print("Loading historical AppleSupport pairs...")

    df = pd.read_csv(input_file)

    df["customer_text_clean"] = (
        df["customer_text"].fillna("").astype(str).str.strip()
    )
    df["support_text_clean"] = (
        df["support_text"].fillna("").astype(str).str.strip()
    )

    df = df[df["customer_text_clean"].str.len() > 0].copy()

    print(f"Historical examples before leakage removal: {len(df):,}")

    if golden_file is not None:
        golden = pd.read_csv(golden_file)

        if "customer_tweet_id" not in golden.columns:
            raise ValueError("golden_set.csv must contain customer_tweet_id")

        golden_ids = set(
            pd.to_numeric(golden["customer_tweet_id"], errors="coerce")
            .dropna()
            .astype("int64")
            .tolist()
        )

        before = len(df)
        df = df[
            ~pd.to_numeric(
                df["customer_tweet_id"], errors="coerce"
            ).isin(golden_ids)
        ].copy()

        print(
            f"Golden examples removed from retrieval: "
            f"{before - len(df):,}"
        )

    print(f"Historical examples in retrieval corpus: {len(df):,}")

    print("Building TF-IDF index...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_features=20000,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(df["customer_text_clean"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Saving retrieval index...")

    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    save_npz(output_dir / "tfidf_matrix.npz", matrix)

    metadata = df[
        [
            "customer_tweet_id",
            "customer_created_at",
            "customer_text_clean",
            "support_tweet_id",
            "support_created_at",
            "support_text_clean",
        ]
    ].reset_index(drop=True)

    metadata.to_csv(
        output_dir / "retrieval_metadata.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("LEAKAGE-SAFE RETRIEVAL INDEX CREATED")
    print("=" * 60)
    print(f"Examples indexed: {len(df):,}")
    print(f"TF-IDF features: {matrix.shape[1]:,}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Saved to: {output_dir}")


def search_index(index_dir, query, top_k):
    index_dir = Path(index_dir)

    print("\nLoading retrieval index...")

    vectorizer = joblib.load(
        index_dir / "tfidf_vectorizer.joblib"
    )

    matrix = load_npz(
        index_dir / "tfidf_matrix.npz"
    )

    metadata = pd.read_csv(
        index_dir / "retrieval_metadata.csv"
    )

    query_vector = vectorizer.transform([query])

    scores = matrix @ query_vector.T
    scores = np.asarray(scores.toarray()).ravel()

    top_k = min(top_k, len(metadata))
    top_indices = np.argsort(scores)[::-1][:top_k]

    print("\n" + "=" * 70)
    print("TOP HISTORICAL MATCHES")
    print("=" * 70)

    for rank, idx in enumerate(top_indices, 1):
        row = metadata.iloc[idx]

        print(f"\n--- Match {rank} ---")
        print(f"Similarity: {scores[idx]:.4f}")

        print("\nCustomer:")
        print(row["customer_text_clean"])

        print("\nAppleSupport:")
        print(row["support_text_clean"])

        print("-" * 70)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--build",
        action="store_true",
        help="Build retrieval index",
    )

    parser.add_argument(
        "--input",
        default="data/apple_support_pairs.csv",
        help="Historical customer/support pairs",
    )

    parser.add_argument(
        "--gold-file",
        default=None,
        help="Golden evaluation set to exclude",
    )

    parser.add_argument(
        "--index-dir",
        default="data/retrieval_index",
        help="Directory for retrieval index",
    )

    parser.add_argument(
        "--query",
        default=None,
        help="Customer message to search",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of historical matches",
    )

    args = parser.parse_args()

    if args.build:
        build_index(
            input_file=args.input,
            output_dir=args.index_dir,
            golden_file=args.gold_file,
        )

    elif args.query:
        search_index(
            index_dir=args.index_dir,
            query=args.query,
            top_k=args.top_k,
        )

    else:
        parser.error("Use --build or provide --query.")


if __name__ == "__main__":
    main()
