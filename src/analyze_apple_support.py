import argparse
import re
from collections import Counter
import pandas as pd

def clean(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["clean_customer_text"] = df["customer_text"].map(clean)

    print("\n=== Dataset profile ===")
    print(f"Rows: {len(df):,}")
    print(f"Unique customers: {df['customer_author_id'].nunique():,}")
    print(f"Unique AppleSupport replies represented: {df['support_tweet_id'].nunique():,}")

    token_counter = Counter()
    for text in df["clean_customer_text"]:
        token_counter.update(
            t.lower()
            for t in re.findall(r"[A-Za-z]{3,}", text)
            if t.lower() not in {
                "the","and","for","that","this","with","you","have","from",
                "are","was","but","not","can","your","please","thanks",
                "thank","support","apple","applesupport"
            }
        )

    print("\n=== Frequent content words (first-pass exploration only) ===")
    for word, count in token_counter.most_common(40):
        print(f"{word:20s} {count:>7,}")

    print("\n=== Example customer messages ===")
    examples = df.sample(min(30, len(df)), random_state=42)["clean_customer_text"]
    for i, text in enumerate(examples, 1):
        print(f"{i:02d}. {text}")

if __name__ == "__main__":
    main()
