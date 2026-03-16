import os
import re
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MaxAbsScaler

DATASET_PATH = "/kaggle/working/injection_data/injection_dataset.csv"
MODEL_DIR    = "/kaggle/working/models"
os.makedirs(MODEL_DIR, exist_ok=True)

INJECTION_KEYWORDS = [
    "ignore", "disregard", "forget", "override", "bypass", "disable",
    "previous instructions", "system prompt", "your instructions",
    "pretend", "roleplay", "act as", "you are now", "from now on",
    "jailbreak", "dan mode", "developer mode", "unrestricted",
    "no restrictions", "no limitations", "safety filters",
    "reveal", "output your", "print your", "show your", "repeat your",
    "hidden instruction", "note to ai", "when an ai", "language model",
    "to the ai", "ai assistant", "attention ai",
    "new task", "new instruction", "real task", "actual task",
    "your real purpose", "your true purpose",
]

IMPERATIVE_VERBS = [
    "ignore", "forget", "disregard", "pretend", "act", "be", "become",
    "switch", "override", "bypass", "reveal", "output", "print", "show",
    "repeat", "disable", "enter", "activate", "enable", "execute",
    "perform", "simulate", "roleplay", "imagine", "assume",
]

META_REFERENCES = [
    "ai", "model", "system", "prompt", "instruction", "assistant",
    "bot", "chatbot", "llm", "gpt", "language model", "neural network",
    "context", "training", "guidelines", "rules", "policy", "filter",
]


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def structural_features(text):
    if not isinstance(text, str):
        text = ""
    lower = text.lower()
    words = lower.split()
    word_count = max(len(words), 1)
    sentences  = re.split(r"[.!?]", text)
    sent_count = max(len([s for s in sentences if s.strip()]), 1)

    keyword_hits   = sum(1 for kw in INJECTION_KEYWORDS if kw in lower)
    imperative_hits = sum(1 for v in IMPERATIVE_VERBS if re.search(rf"\b{v}\b", lower))
    meta_hits      = sum(1 for m in META_REFERENCES if re.search(rf"\b{m}\b", lower))

    second_person  = len(re.findall(r"\byou\b|\byour\b|\byourself\b", lower))
    quoted_cmds    = len(re.findall(r'["\']([^"\']{5,})["\']', text))
    has_brackets   = int(bool(re.search(r"\[.*?\]|\{.*?\}|<.*?>", text)))
    all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)

    return np.array([
        keyword_hits / max(word_count / 10, 1),
        imperative_hits / word_count,
        meta_hits / word_count,
        second_person / word_count,
        quoted_cmds,
        has_brackets,
        all_caps_words / word_count,
        word_count,
        sent_count,
        keyword_hits,
        imperative_hits,
        meta_hits,
    ], dtype=np.float32)


STRUCT_FEATURE_NAMES = [
    "keyword_density", "imperative_ratio", "meta_reference_ratio",
    "second_person_ratio", "quoted_command_count", "has_brackets",
    "all_caps_ratio", "word_count", "sentence_count",
    "keyword_hits_raw", "imperative_hits_raw", "meta_hits_raw",
]


def build_features(texts, vectorizer, scaler, fit=False):
    cleaned = [clean_text(t) for t in texts]

    if fit:
        tfidf = vectorizer.fit_transform(cleaned)
    else:
        tfidf = vectorizer.transform(cleaned)

    struct = np.array([structural_features(t) for t in texts], dtype=np.float32)

    if fit:
        struct = scaler.fit_transform(struct)
    else:
        struct = scaler.transform(struct)

    return sp.hstack([tfidf, sp.csr_matrix(struct)])


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total: {len(df)} | Injections: {df['label'].sum()} | Safe: {(df['label']==0).sum()}")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"].values, df["label"].values,
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=1,
        strip_accents="unicode",
        analyzer="word",
    )
    scaler = MaxAbsScaler()

    print("Building features...")
    X_train = build_features(X_train_text, vectorizer, scaler, fit=True)
    X_test  = build_features(X_test_text,  vectorizer, scaler, fit=False)
    print(f"Feature dimensions: {X_train.shape}")

    print("Training Logistic Regression...")
    model = LogisticRegression(
        C=2.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["safe", "injection"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    joblib.dump(model,      f"{MODEL_DIR}/injection_classifier.pkl")
    joblib.dump(vectorizer, f"{MODEL_DIR}/injection_vectorizer.pkl")
    joblib.dump(scaler,     f"{MODEL_DIR}/injection_scaler.pkl")
    joblib.dump(STRUCT_FEATURE_NAMES, f"{MODEL_DIR}/injection_struct_features.pkl")
    print(f"\nSaved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
