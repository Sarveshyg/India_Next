import os
import re
import base64
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MaxAbsScaler

DATASET_DIR = "/kaggle/input/datasets/naserabdullahalam/phishing-email-dataset"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

URGENCY_KEYWORDS = [
    "urgent", "immediately", "suspended", "verify", "confirm", "expire",
    "limited time", "act now", "click here", "account blocked", "unusual activity",
    "security alert", "update required", "validate", "restricted", "locked"
]

COLUMN_MAPS = {
    "email text":   "text",
    "body":         "text",
    "message":      "text",
    "mail":         "text",
    "text":         "text",
    "email type":   "label",
    "label":        "label",
    "type":         "label",
    "class":        "label",
    "spam":         "label",
}

PHISHING_VALUES = {
    "phishing email", "phishing", "spam", "1", "yes", "true",
    "nigerian_fraud", "fraud"
}


def decode_base64_parts(text):
    pattern = r"[A-Za-z0-9+/]{40,}={0,2}"
    def try_decode(m):
        try:
            return base64.b64decode(m.group()).decode("utf-8", errors="ignore")
        except Exception:
            return m.group()
    return re.sub(pattern, try_decode, text)


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = decode_base64_parts(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " URLTOKEN ", text)
    text = re.sub(r"\S+@\S+", " EMAILTOKEN ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def handcrafted_features(raw_series):
    records = []
    for text in raw_series:
        if not isinstance(text, str):
            text = ""
        url_count        = len(re.findall(r"http\S+|www\S+", text, re.I))
        img_count        = len(re.findall(r"<img", text, re.I))
        exclaim_count    = text.count("!")
        words            = text.split()
        word_count       = max(len(words), 1)
        caps_ratio       = sum(1 for w in words if w.isupper()) / word_count
        has_html         = int(bool(re.search(r"<[a-z][\s\S]*?>", text, re.I)))
        urgency_flag     = int(any(kw in text.lower() for kw in URGENCY_KEYWORDS))
        link_ratio       = min(url_count / max(word_count / 50, 1), 1.0)
        records.append([
            url_count, img_count, exclaim_count / word_count,
            caps_ratio, has_html, urgency_flag, link_ratio, word_count
        ])
    return np.array(records, dtype=np.float32)


def normalise_label(val):
    return 1 if str(val).strip().lower() in PHISHING_VALUES else 0


def load_csv(path):
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")

    df.columns = [c.strip().lower() for c in df.columns]
    rename = {}
    for col in df.columns:
        if col in COLUMN_MAPS and COLUMN_MAPS[col] not in rename.values():
            rename[col] = COLUMN_MAPS[col]
    df = df.rename(columns=rename)

    if "text" not in df.columns or "label" not in df.columns:
        return None

    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].apply(normalise_label)
    return df


def load_all_datasets():
    frames = []
    for fname in os.listdir(DATASET_DIR):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(DATASET_DIR, fname)
        df = load_csv(path)
        if df is not None and len(df) > 0:
            print(f"  Loaded {fname}: {len(df)} rows | phishing={df['label'].sum()}")
            frames.append(df)
        else:
            print(f"  Skipped {fname}: unrecognised columns")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"])
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nTotal rows: {len(combined)} | phishing: {combined['label'].sum()} | safe: {(combined['label']==0).sum()}")
    return combined


def main():
    print("Loading datasets...")
    df = load_all_datasets()

    print("\nCleaning text...")
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.len() > 20].reset_index(drop=True)

    X_text = df["clean"].values
    y      = df["label"].values

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nFitting TF-IDF vectoriser...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf  = vectorizer.transform(X_test_text)

    print("Building handcrafted features...")
    X_train_raw = df["text"].values[:len(X_train_text)]
    X_test_raw  = df["text"].values[len(X_train_text):]

    X_train_hand = handcrafted_features(X_train_raw)
    X_test_hand  = handcrafted_features(X_test_raw)

    scaler = MaxAbsScaler()
    X_train_hand = scaler.fit_transform(X_train_hand)
    X_test_hand  = scaler.transform(X_test_hand)

    X_train = sp.hstack([X_train_tfidf, sp.csr_matrix(X_train_hand)])
    X_test  = sp.hstack([X_test_tfidf,  sp.csr_matrix(X_test_hand)])

    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["safe", "phishing"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(vectorizer, f"{MODEL_DIR}/email_vectorizer.pkl")
    joblib.dump(model,      f"{MODEL_DIR}/email_classifier.pkl")
    joblib.dump(scaler,     f"{MODEL_DIR}/email_scaler.pkl")
    print(f"\nModels saved to /{MODEL_DIR}/")


if __name__ == "__main__":
    main()
