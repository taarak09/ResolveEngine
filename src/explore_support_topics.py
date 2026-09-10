import argparse
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Decode the common HTML entity in the dataset
    text = text.replace("&amp;", "&")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--clusters", type=int, default=15)

    args = parser.parse_args()

    print("Loading data...")

    df = pd.read_csv(args.input)

    print(f"Total pairs: {len(df):,}")

    # Clean customer messages
    df["clean_text"] = df["customer_text"].map(clean_text)

    # Remove empty messages and very short replies
    df = df[df["clean_text"].str.len() >= 25].copy()

    print(f"Usable customer messages: {len(df):,}")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=10,
        max_features=15000,
        sublinear_tf=True
    )

    print("Creating TF-IDF representation...")

    X = vectorizer.fit_transform(df["clean_text"])

    print("Running clustering...")

    model = KMeans(
        n_clusters=args.clusters,
        random_state=42,
        n_init=10
    )

    df["cluster"] = model.fit_predict(X)

    terms = vectorizer.get_feature_names_out()

    for cluster_id in range(args.clusters):

        cluster_df = df[df["cluster"] == cluster_id]

        if cluster_df.empty:
            continue

        center = model.cluster_centers_[cluster_id]

        top_indices = center.argsort()[-15:][::-1]

        keywords = [
            terms[i]
            for i in top_indices
        ]

        print("\n")
        print("=" * 80)
        print(
            f"CLUSTER {cluster_id} "
            f"({len(cluster_df):,} messages)"
        )
        print("=" * 80)

        print("\nTop keywords:")
        print(", ".join(keywords))

        print("\nExample customer messages:")

        samples = cluster_df.sample(
            min(15, len(cluster_df)),
            random_state=42
        )["clean_text"]

        for i, text in enumerate(samples, 1):
            print(f"{i:02d}. {text}")


if __name__ == "__main__":
    main()