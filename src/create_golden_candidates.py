import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


INTENTS = {
    "I01": "IOS_UPDATE_PROBLEM",
    "I02": "POST_UPDATE_DEVICE_ISSUE",
    "I03": "BATTERY_CHARGING",
    "I04": "KEYBOARD_TEXT_BUG",
    "I05": "SCREEN_TOUCH_DISPLAY",
    "I06": "CONNECTIVITY",
    "I07": "APPS_AND_MEDIA",
    "I08": "ACCOUNT_AND_PAYMENT",
    "I09": "CALLS_MESSAGES_NOTIFICATIONS",
    "I10": "WATCH_AUDIO_ACCESSORIES",
    "I11": "MAC_ITUNES_DEVELOPER",
    "I12": "ORDERS_REPAIRS_SUPPORT",
    "I13": "HOW_TO_OR_FEATURE",
    "I14": "OTHER_OR_UNCLEAR",
}


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("&amp;", "&")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=250)

    args = parser.parse_args()

    print("Loading data...")

    df = pd.read_csv(args.input)

    df["customer_text_clean"] = df["customer_text"].map(clean_text)
    df["support_text_clean"] = df["support_text"].map(clean_text)

    # Remove extremely short messages from the candidate pool.
    df = df[df["customer_text_clean"].str.len() >= 25].copy()

    print(f"Candidate pool: {len(df):,}")

    # We use clustering only to get diversity.
    # It does NOT determine the final intent labels.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=10,
        max_features=15000,
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(df["customer_text_clean"])

    # Create 20 exploratory groups to diversify our manual sample.
    n_clusters = min(20, len(df))

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    df["exploration_cluster"] = model.fit_predict(X)

    # We want 250 total candidates.
    # Sample approximately evenly across exploratory clusters.
    per_cluster = max(1, args.n // n_clusters)

    sampled_parts = []

    for cluster_id, group in df.groupby("exploration_cluster"):

        take = min(per_cluster, len(group))

        sampled = group.sample(
            take,
            random_state=42 + int(cluster_id),
        )

        sampled_parts.append(sampled)

    candidates = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    # If rounding left us short, fill from the remaining pool.
    target = min(args.n, len(df))

    if len(candidates) < target:

        remaining = df[
            ~df["customer_tweet_id"].isin(
                candidates["customer_tweet_id"]
            )
        ]

        extra = remaining.sample(
            min(target - len(candidates), len(remaining)),
            random_state=123,
        )

        candidates = pd.concat(
            [candidates, extra],
            ignore_index=True,
        )

    # If we have too many, reduce deterministically.
    candidates = candidates.sample(
        min(target, len(candidates)),
        random_state=42,
    ).reset_index(drop=True)

    # Add blank columns for YOUR manual labels.
    candidates["intent_id"] = ""
    candidates["intent_name"] = ""
    candidates["should_escalate"] = ""
    candidates["label_notes"] = ""

    output_columns = [
        "customer_tweet_id",
        "customer_created_at",
        "customer_text_clean",
        "support_tweet_id",
        "support_text_clean",
        "exploration_cluster",
        "intent_id",
        "intent_name",
        "should_escalate",
        "label_notes",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidates[output_columns].to_csv(
        output_path,
        index=False,
    )

    print("\n======================================")
    print("Golden-set candidates created")
    print("======================================")
    print(f"Examples: {len(candidates):,}")
    print(f"Saved to: {output_path}")

    print("\nIntent labels:")
    for intent_id, intent_name in INTENTS.items():
        print(f"{intent_id}: {intent_name}")


if __name__ == "__main__":
    main()