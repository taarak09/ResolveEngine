import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


INPUT_FILE = "data/golden_candidates.csv"


def main():
    print("Loading golden set...")

    df = pd.read_csv(INPUT_FILE)

    # Remove rows with missing labels.
    df = df.dropna(
        subset=["customer_text_clean", "intent_name"]
    ).copy()

    X = df["customer_text_clean"].astype(str)
    y = df["intent_name"].astype(str)

    print(f"Examples: {len(df)}")

    # Keep the test set separate.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")

    # Convert text into TF-IDF features.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")

    # Simple traditional ML classifier.
    model = LogisticRegression(
        max_iter=2000,
        C=2.0,
        class_weight="balanced"
    )

    print("Training classifier...")

    model.fit(
        X_train_tfidf,
        y_train
    )

    # Predict the held-out test examples.
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("TF-IDF + LOGISTIC REGRESSION BASELINE")
    print("=" * 60)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nPer-class report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()