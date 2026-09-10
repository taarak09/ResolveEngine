
import argparse
from pathlib import Path
import pandas as pd


BRAND = "AppleSupport"


def parse_ids(value):
    """Return tweet IDs from the comma-separated response_tweet_id field."""
    if pd.isna(value) or str(value).strip() == "":
        return []
    result = []
    for item in str(value).split(","):
        item = item.strip()
        if item.isdigit():
            result.append(int(item))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build customer->AppleSupport response pairs from TWCS."
    )
    parser.add_argument("--input", required=True, help="Path to twcs.csv")
    parser.add_argument(
        "--output",
        default="data/apple_support_pairs.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: collect all AppleSupport tweet IDs and their text.
    support_parts = []

    for chunk in pd.read_csv(
        input_path,
        usecols=[
            "tweet_id",
            "author_id",
            "created_at",
            "text",
        ],
        chunksize=300_000,
    ):
        support = chunk[chunk["author_id"].eq(BRAND)].copy()
        if not support.empty:
            support_parts.append(support)

    if not support_parts:
        raise RuntimeError(f"No tweets found for author_id={BRAND}")

    support = pd.concat(support_parts, ignore_index=True)
    support["tweet_id"] = support["tweet_id"].astype("int64")
    support = support.drop_duplicates("tweet_id")

    support_lookup = dict(
        zip(
            support["tweet_id"],
            zip(support["created_at"], support["text"]),
        )
    )
    support_ids = set(support_lookup)

    # Pass 2: find inbound customer tweets whose response_tweet_id
    # points to at least one AppleSupport tweet. This is the direction
    # we need for the support-agent task:
    #
    # customer message -> historical AppleSupport response.
    output_parts = []

    for chunk in pd.read_csv(
        input_path,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ],
        chunksize=300_000,
    ):
        inbound = (
            chunk["inbound"].astype(str).str.strip().str.lower().eq("true")
        )

        candidates = chunk[
            inbound
            & chunk["author_id"].ne(BRAND)
            & chunk["response_tweet_id"].notna()
        ].copy()

        if candidates.empty:
            continue

        candidates["apple_response_ids"] = candidates[
            "response_tweet_id"
        ].map(
            lambda value: [
                tweet_id
                for tweet_id in parse_ids(value)
                if tweet_id in support_ids
            ]
        )

        candidates = candidates[
            candidates["apple_response_ids"].map(bool)
        ]

        if candidates.empty:
            continue

        rows = []

        for _, row in candidates.iterrows():
            for support_id in row["apple_response_ids"]:
                support_created_at, support_text = support_lookup[support_id]

                rows.append(
                    {
                        "customer_tweet_id": int(row["tweet_id"]),
                        "customer_author_id": row["author_id"],
                        "customer_created_at": row["created_at"],
                        "customer_text": row["text"],
                        "support_tweet_id": int(support_id),
                        "support_created_at": support_created_at,
                        "support_text": support_text,
                        "in_response_to_tweet_id": row[
                            "in_response_to_tweet_id"
                        ],
                    }
                )

        if rows:
            output_parts.append(pd.DataFrame(rows))

    if not output_parts:
        raise RuntimeError("No customer->AppleSupport response pairs were found.")

    pairs = pd.concat(output_parts, ignore_index=True)

    # Remove exact duplicates and sort chronologically.
    pairs = pairs.drop_duplicates(
        subset=["customer_tweet_id", "support_tweet_id"]
    )

    pairs["customer_created_at"] = pd.to_datetime(
        pairs["customer_created_at"], errors="coerce"
    )
    pairs["support_created_at"] = pd.to_datetime(
        pairs["support_created_at"], errors="coerce"
    )

    pairs = pairs.sort_values(
        ["customer_created_at", "customer_tweet_id"]
    ).reset_index(drop=True)

    pairs.to_csv(output_path, index=False)

    unique_customers = pairs["customer_tweet_id"].nunique()
    unique_support = pairs["support_tweet_id"].nunique()

    print("======================================")
    print("AppleSupport customer-response pairs")
    print("======================================")
    print(f"AppleSupport tweets found: {len(support):,}")
    print(f"Customer->support pairs: {len(pairs):,}")
    print(f"Unique customer tweets: {unique_customers:,}")
    print(f"Unique AppleSupport responses linked: {unique_support:,}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
