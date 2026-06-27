import argparse
import os
import zipfile

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import prepare_data, stratified_split
from src.evaluate import evaluate_model


def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Load dataset from either a CSV file or a ZIP file containing AI_Human.csv.
    """

    if data_path.endswith(".zip"):
        extract_dir = "data/extracted"
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(data_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        csv_path = os.path.join(extract_dir, "AI_Human.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                "AI_Human.csv was not found inside the ZIP file."
            )

        return pd.read_csv(csv_path)

    if data_path.endswith(".csv"):
        return pd.read_csv(data_path)

    raise ValueError("data_path must be a .csv or .zip file")


def train_tfidf_model(data_path: str):
    """
    Train TF-IDF + Logistic Regression model.
    Saves trained model, vectorizer, and metrics.
    """

    print("Loading dataset...")
    df = load_dataset(data_path)

    print("Preparing data...")
    df = prepare_data(df, text_col="text", label_col="generated")

    (
        X_train,
        X_val,
        X_test,
        _,
        _,
        _,
        y_train,
        y_val,
        y_test,
    ) = stratified_split(
        df,
        text_col="clean_text",
        raw_col="text",
        label_col="generated",
        seed=42,
    )

    print(f"Train size: {len(X_train):,}")
    print(f"Validation size: {len(X_val):,}")
    print(f"Test size: {len(X_test):,}")

    print("Training TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train_tfidf, y_train)

    print("Evaluating model...")
    val_preds = model.predict(X_val_tfidf)
    test_preds = model.predict(X_test_tfidf)

    val_result = evaluate_model(
        y_val,
        val_preds,
        "TF-IDF + Logistic Regression",
        "Validation",
    )

    test_result = evaluate_model(
        y_test,
        test_preds,
        "TF-IDF + Logistic Regression",
        "Test",
    )

    os.makedirs("artifacts", exist_ok=True)

    print("Saving model artifacts...")
    joblib.dump(vectorizer, "artifacts/tfidf_vectorizer.joblib")
    joblib.dump(model, "artifacts/tfidf_logreg_model.joblib")

    metrics_df = pd.DataFrame([val_result, test_result])
    metrics_df.to_csv("artifacts/tfidf_metrics.csv", index=False)

    print("\nTraining complete.")
    print("Saved files:")
    print("- artifacts/tfidf_vectorizer.joblib")
    print("- artifacts/tfidf_logreg_model.joblib")
    print("- artifacts/tfidf_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + Logistic Regression model for AI text detection."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to AI_Human.csv or AI_Human.csv.zip",
    )

    args = parser.parse_args()

    train_tfidf_model(args.data_path)