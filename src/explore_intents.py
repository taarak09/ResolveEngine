import argparse
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def clean_text(text: str) -> str:
    text = "" if pd.isna(text) else str(text)

    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def looks_like_low_information_message(text: str) -> bool:
    text = text.lower().strip()

    if not text:
        return True

    low_information = {
        "yes",
        "no",
        "thanks",
        "thank you",
        "okay",
        "ok",
        "sure",
        "great",
        "dm sent",
        "sent dm",
        "dmd thanks",
    }

    if text in low_information:
        return True

    # Version-only answers such as:
    # "11.1.2", "iOS 11.1.2", "iPhone 7"
    if re.fullmatch(r"(ios\s*)?\d+(\.\d+){1,3}", text):
        return True

    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--clusters", type=int, default=12)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    df["clean_text"] = df["customer_text"].map(clean_text)

    # Remove very short / low-information follow-up messages
    df = df[~df["clean_text"].map(looks_like_low_information_message)].copy()

    df = df[df["clean_text"].str.len() >= 20].copy()

    print("\n======================================")
    print("AppleSupport Intent Exploration")
    print("======================================")
    print(f"Usable customer messages: {len(df):,}")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=8,
        max_features=12000,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(df["clean_text"])

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

        top_indices = center.argsort()[-12:][::-1]
        keywords = [terms[i] for i in top_indices]

        print("\n")
        print("=" * 70)
        print(
            f"CLUSTER {cluster_id} "
            f"({len(cluster_df):,} examples)"
        )
        print("=" * 70)

        print("Keywords:")
        print(", ".join(keywords))

        print("\nRepresentative customer messages:")

        sample_size = min(10, len(cluster_df))

        samples = cluster_df.sample(
            sample_size,
            random_state=42
        )["clean_text"]

        for i, text in enumerate(samples, 1):
            print(f"{i:02d}. {text}")


if __name__ == "__main__":
    main()