import re
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """
    Lightweight cleaning for classical ML and LSTM models.
    Transformer models should use raw text.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_data(df: pd.DataFrame, text_col: str = "text", label_col: str = "generated"):
    df = df.dropna(subset=[text_col, label_col]).copy()
    df["clean_text"] = df[text_col].apply(clean_text)
    df[label_col] = df[label_col].astype(int)
    return df


def stratified_split(df, text_col="clean_text", raw_col="text", label_col="generated", seed=42):
    X_clean = df[text_col]
    X_raw = df[raw_col]
    y = df[label_col]

    Xc_train, Xc_temp, Xr_train, Xr_temp, y_train, y_temp = train_test_split(
        X_clean, X_raw, y, test_size=0.30, random_state=seed, stratify=y
    )

    Xc_val, Xc_test, Xr_val, Xr_test, y_val, y_test = train_test_split(
        Xc_temp, Xr_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    return Xc_train, Xc_val, Xc_test, Xr_train, Xr_val, Xr_test, y_train, y_val, y_test
