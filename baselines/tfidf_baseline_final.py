import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


DEV_FILE = "data/development_set.csv"
GOLD_FILE = "data/golden_set.csv"


def main():
    print("Loading development set...")
    dev = pd.read_csv(DEV_FILE)

    print("Loading golden evaluation set...")
    gold = pd.read_csv(GOLD_FILE)

    # Training data
    X_train = dev["customer_text_clean"].astype(str)
    y_train = dev["intent_name"].astype(str)

    # Held-out evaluation data
    X_test = gold["customer_text_clean"].astype(str)
    y_test = gold["intent_name"].astype(str)

    print(f"Development examples: {len(dev)}")
    print(f"Golden examples: {len(gold)}")

    # TF-IDF learns vocabulary ONLY from development data.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")

    # Simple traditional ML classifier
    model = LogisticRegression(
        max_iter=2000,
        C=2.0,
        class_weight="balanced"
    )

    print("Training classifier...")
    model.fit(X_train_tfidf, y_train)

    # Evaluate ONLY on the held-out golden set.
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 65)
    print("FINAL SIMPLE BASELINE")
    print("TF-IDF + LOGISTIC REGRESSION")
    print("=" * 65)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    print("\nPer-class results:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()