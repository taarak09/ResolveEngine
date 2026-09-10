
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description="Split the labeled pool into development and held-out golden evaluation sets."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--dev-size", type=int, default=100)
    parser.add_argument("--gold-size", type=int, default=150)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required = {"customer_tweet_id", "intent_name", "should_escalate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Drop completely unlabeled rows, if any.
    df = df.dropna(subset=["customer_tweet_id", "intent_name", "should_escalate"]).copy()

    total_needed = args.dev_size + args.gold_size
    if len(df) < total_needed:
        raise ValueError(
            f"Need at least {total_needed} labeled rows, but only {len(df)} are available."
        )

    # We want the held-out set to remain representative across the intent classes.
    # First sample the requested pool; then split it stratified by intent.
    pool = df.sample(total_needed, random_state=42)

    dev, gold = train_test_split(
        pool,
        test_size=args.gold_size,
        random_state=42,
        stratify=pool["intent_name"],
    )

    dev = dev.sort_values("customer_tweet_id").reset_index(drop=True)
    gold = gold.sort_values("customer_tweet_id").reset_index(drop=True)

    out_dir = Path(args.input).parent

    dev_path = out_dir / "development_set.csv"
    gold_path = out_dir / "golden_set.csv"

    # The development set can be used for baseline training.
    dev.to_csv(dev_path, index=False)

    # The golden set is held out for evaluation only.
    gold.to_csv(gold_path, index=False)

    overlap = set(dev["customer_tweet_id"]) & set(gold["customer_tweet_id"])

    print("=" * 60)
    print("DEVELOPMENT / GOLDEN SPLIT")
    print("=" * 60)
    print(f"Development examples: {len(dev)}")
    print(f"Golden evaluation examples: {len(gold)}")
    print(f"ID overlap: {len(overlap)}")

    print("\nDevelopment intent counts:")
    print(dev["intent_name"].value_counts().sort_index())

    print("\nGolden intent counts:")
    print(gold["intent_name"].value_counts().sort_index())

    print(f"\nSaved development set: {dev_path}")
    print(f"Saved golden set:      {gold_path}")

    if overlap:
        raise RuntimeError("Leakage detected: development and golden sets overlap.")


if __name__ == "__main__":
    main()
