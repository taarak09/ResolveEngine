import argparse
from pathlib import Path
import pandas as pd

BRAND = "AppleSupport"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to twcs.csv")
    parser.add_argument("--output", required=True, help="Output CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # First pass: keep AppleSupport tweets only and build a tweet-id lookup.
    apple_parts = []
    for chunk in pd.read_csv(input_path, chunksize=300_000):
        part = chunk[chunk["author_id"].eq(BRAND)].copy()
        if not part.empty:
            apple_parts.append(part)

    if not apple_parts:
        raise RuntimeError(f"No rows found for author_id={BRAND}")

    apple = pd.concat(apple_parts, ignore_index=True)
    apple["tweet_id"] = apple["tweet_id"].astype("int64")

    # Second pass: keep non-AppleSupport tweets whose parent tweet is an
    # AppleSupport tweet. These are the direct customer replies we can link.
    apple_ids = set(apple["tweet_id"].tolist())
    customer_parts = []

    for chunk in pd.read_csv(input_path, chunksize=300_000):
        mask = chunk["in_response_to_tweet_id"].notna()
        if mask.any():
            parent_ids = chunk.loc[mask, "in_response_to_tweet_id"].astype("int64")
            keep = mask & chunk["author_id"].ne(BRAND) & parent_ids.isin(apple_ids)
            part = chunk[keep].copy()
            if not part.empty:
                customer_parts.append(part)

    customers = (
        pd.concat(customer_parts, ignore_index=True)
        if customer_parts
        else pd.DataFrame(columns=apple.columns)
    )

    support = apple[["tweet_id", "author_id", "created_at", "text"]].rename(
        columns={"tweet_id": "support_tweet_id", "text": "support_text"}
    )

    customer = customers[
        ["tweet_id", "author_id", "created_at", "text", "in_response_to_tweet_id"]
    ].rename(
        columns={
            "tweet_id": "customer_tweet_id",
            "author_id": "customer_author_id",
            "created_at": "customer_created_at",
            "text": "customer_text",
            "in_response_to_tweet_id": "support_tweet_id",
        }
    )

    interactions = customer.merge(support, on="support_tweet_id", how="inner")
    interactions = interactions.sort_values(
        ["support_tweet_id", "customer_tweet_id"]
    ).reset_index(drop=True)

    interactions.to_csv(output_path, index=False)

    print(f"AppleSupport tweets: {len(apple):,}")
    print(f"Direct customer replies linked: {len(customers):,}")
    print(f"Interaction rows written: {len(interactions):,}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
