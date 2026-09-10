import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

INPUT_FILE = "data/golden_candidates.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    # Find the most common intent in the labeled set.
    majority_intent = df["intent_name"].value_counts().idxmax()

    # Predict that intent for every example.
    y_true = df["intent_name"]
    y_pred = [majority_intent] * len(df)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("=" * 60)
    print("MAJORITY-CLASS BASELINE")
    print("=" * 60)

    print(f"Most common intent: {majority_intent}")
    print(f"Examples: {len(df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nPer-class report:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()