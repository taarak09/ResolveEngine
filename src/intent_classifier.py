
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


DEV_FILE = "data/development_set.csv"
MODEL_DIR = "data/intent_model"


def train(dev_file: str, model_dir: str) -> None:
    dev = pd.read_csv(dev_file)

    required = {"customer_text_clean", "intent_name"}
    missing = required - set(dev.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    dev = dev.dropna(
        subset=["customer_text_clean", "intent_name"]
    ).copy()

    X = dev["customer_text_clean"].astype(str)
    y = dev["intent_name"].astype(str)

    print(f"Development examples: {len(dev)}")
    print(f"Intent classes: {y.nunique()}")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        max_features=20000,
    )

    X_tfidf = vectorizer.fit_transform(X)

    model = LogisticRegression(
        max_iter=3000,
        C=2.0,
        class_weight="balanced",
        random_state=42,
    )

    print("Training intent classifier...")
    model.fit(X_tfidf, y)

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        vectorizer,
        output_dir / "vectorizer.joblib",
    )

    joblib.dump(
        model,
        output_dir / "classifier.joblib",
    )

    labels = pd.DataFrame(
        {"intent_name": model.classes_}
    )
    labels.to_csv(
        output_dir / "labels.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("INTENT CLASSIFIER TRAINED")
    print("=" * 60)
    print(f"Features: {X_tfidf.shape[1]:,}")
    print(f"Classes: {len(model.classes_)}")
    print(f"Saved to: {output_dir}")


def predict(message: str, model_dir: str) -> None:
    model_dir = Path(model_dir)

    vectorizer = joblib.load(
        model_dir / "vectorizer.joblib"
    )
    model = joblib.load(
        model_dir / "classifier.joblib"
    )

    X = vectorizer.transform([message])

    probabilities = model.predict_proba(X)[0]
    order = np.argsort(probabilities)[::-1]

    predicted = model.classes_[order[0]]
    confidence = float(probabilities[order[0]])

    print("\n" + "=" * 60)
    print("INTENT PREDICTION")
    print("=" * 60)
    print(f"Message: {message}")
    print(f"Predicted intent: {predicted}")
    print(f"Confidence: {confidence:.4f}")

    print("\nTop 3 intents:")
    for idx in order[:3]:
        print(
            f"{model.classes_[idx]}: "
            f"{probabilities[idx]:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the classifier on development_set.csv",
    )

    parser.add_argument(
        "--predict",
        type=str,
        help="Predict the intent for one customer message",
    )

    parser.add_argument(
        "--dev-file",
        default=DEV_FILE,
    )

    parser.add_argument(
        "--model-dir",
        default=MODEL_DIR,
    )

    args = parser.parse_args()

    if args.train:
        train(
            dev_file=args.dev_file,
            model_dir=args.model_dir,
        )

    elif args.predict:
        predict(
            message=args.predict,
            model_dir=args.model_dir,
        )

    else:
        parser.error(
            "Use --train or --predict."
        )


if __name__ == "__main__":
    main()
