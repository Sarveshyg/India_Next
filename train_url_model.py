import os
import re
import math
import joblib
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from collections import Counter

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = "/kaggle/input/datasets/sid321axn/malicious-urls-dataset/malicious_phish.csv"
MODEL_DIR    = "/kaggle/working/models"
os.makedirs(MODEL_DIR, exist_ok=True)

KNOWN_BRANDS = [
    "paypal", "amazon", "apple", "google", "microsoft", "facebook",
    "netflix", "instagram", "twitter", "linkedin", "dropbox", "github",
    "whatsapp", "zoom", "wellsfargo", "bankofamerica", "chase", "citibank",
    "hsbc", "dhl", "fedex", "ups", "usps", "irs", "ebay", "walmart"
]

SUSPICIOUS_EXTENSIONS = [".exe", ".zip", ".bat", ".js", ".vbs", ".ps1", ".msi", ".dmg", ".apk"]

SUSPICIOUS_PATH_KEYWORDS = [
    "login", "signin", "sign-in", "verify", "secure", "account",
    "update", "confirm", "banking", "authenticate", "validation",
    "recover", "password", "credential", "submit", "checkout"
]

TLD_RISK = {
    ".tk": 0.95, ".ml": 0.95, ".ga": 0.95, ".cf": 0.95, ".gq": 0.95,
    ".xyz": 0.70, ".top": 0.70, ".club": 0.70, ".online": 0.70,
    ".site": 0.70, ".info": 0.65, ".biz": 0.60,
    ".co": 0.40, ".io": 0.35,
    ".com": 0.15, ".net": 0.18, ".org": 0.15,
    ".gov": 0.02, ".edu": 0.02, ".mil": 0.02,
    ".uk": 0.20, ".in": 0.22, ".de": 0.18, ".fr": 0.18,
    ".au": 0.18, ".ca": 0.18, ".jp": 0.20, ".br": 0.25,
    ".ru": 0.55, ".cn": 0.50,
}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "short.link", "rebrand.ly",
    "cutt.ly", "shorturl.at", "tiny.cc"
}


def shannon_entropy(s):
    if not s:
        return 0.0
    freq  = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def is_ip_address(hostname):
    ipv4 = re.match(r"^(\d{1,3}\.){3}\d{1,3}$", hostname or "")
    if ipv4:
        return all(0 <= int(p) <= 255 for p in hostname.split("."))
    return bool(re.match(r"^\[?[0-9a-fA-F:]+\]?$", hostname or ""))


def get_registered_domain(hostname):
    parts = hostname.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else hostname


def extract_features(url):
    if not isinstance(url, str):
        url = ""
    url = url.strip()

    try:
        parsed    = urlparse(url if "://" in url else "http://" + url)
        scheme    = parsed.scheme.lower()
        hostname  = parsed.hostname or ""
        path      = parsed.path or ""
        query     = parsed.query or ""
        netloc    = parsed.netloc or ""
    except Exception:
        hostname, scheme, path, query, netloc = "", "http", "", "", ""

    full_url       = url.lower()
    registered_dom = get_registered_domain(hostname.lower())
    subdomain_part = hostname.lower().replace(registered_dom, "").strip(".")
    tld            = "." + registered_dom.split(".")[-1] if "." in registered_dom else ""

    f = {}

    f["url_length"]           = len(url)
    f["hostname_length"]      = len(hostname)
    f["path_length"]          = len(path)
    f["query_length"]         = len(query)

    f["dot_count"]            = url.count(".")
    f["hyphen_count"]         = url.count("-")
    f["underscore_count"]     = url.count("_")
    f["slash_count"]          = url.count("/")
    f["question_count"]       = url.count("?")
    f["equals_count"]         = url.count("=")
    f["ampersand_count"]      = url.count("&")
    f["at_symbol"]            = int("@" in netloc)
    f["hash_count"]           = url.count("#")
    f["double_slash_in_path"] = int("//" in path)

    digits_in_url             = sum(c.isdigit() for c in url)
    specials                  = sum(not c.isalnum() and c not in "/:.-_?=&#@%" for c in url)
    f["digit_count"]          = digits_in_url
    f["digit_ratio"]          = digits_in_url / max(len(url), 1)
    f["special_char_count"]   = specials
    f["special_char_ratio"]   = specials / max(len(url), 1)

    f["url_entropy"]          = round(shannon_entropy(full_url), 4)
    f["domain_entropy"]       = round(shannon_entropy(hostname.lower()), 4)

    f["is_ip_hostname"]       = int(is_ip_address(hostname))
    f["is_https"]             = int(scheme == "https")
    f["is_shortener"]         = int(registered_dom in URL_SHORTENERS)

    f["subdomain_count"]      = len([s for s in subdomain_part.split(".") if s]) if subdomain_part else 0
    f["subdomain_length"]     = len(subdomain_part)

    f["tld_risk_score"]       = TLD_RISK.get(tld, 0.80)

    path_lower                = path.lower()
    f["suspicious_keywords"]  = int(any(kw in path_lower or kw in query.lower() for kw in SUSPICIOUS_PATH_KEYWORDS))
    f["suspicious_ext"]       = int(any(path_lower.endswith(ext) for ext in SUSPICIOUS_EXTENSIONS))

    f["digit_ratio_domain"]   = sum(c.isdigit() for c in hostname) / max(len(hostname), 1)

    f["non_ascii_ratio"]      = sum(1 for c in hostname if ord(c) > 127) / max(len(hostname), 1)

    f["has_hex_encoding"]     = int(bool(re.search(r"%[0-9a-fA-F]{2}", url)))

    f["path_depth"]           = len([p for p in path.split("/") if p])

    words_in_path             = re.findall(r"[a-zA-Z]{3,}", path)
    f["longest_word_path"]    = max((len(w) for w in words_in_path), default=0)

    brand_in_url              = any(brand in full_url for brand in KNOWN_BRANDS)
    brand_in_domain           = any(brand in registered_dom for brand in KNOWN_BRANDS)
    f["brand_impersonation"]  = int(brand_in_url and not brand_in_domain)

    f["unique_char_ratio"]    = len(set(url)) / max(len(url), 1)

    return f


FEATURE_NAMES = list(extract_features("http://example.com").keys())


def build_feature_matrix(urls):
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES).values.astype(np.float32)


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    url_col  = next((c for c in df.columns if "url"  in c), df.columns[0])
    type_col = next((c for c in df.columns if "type" in c or "label" in c), df.columns[1])

    df           = df[[url_col, type_col]].dropna()
    df.columns   = ["url", "label"]
    df           = df.drop_duplicates(subset=["url"])
    df           = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total URLs: {len(df)}")
    print(df["label"].value_counts())

    le              = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    num_classes     = len(le.classes_)
    print(f"Classes: {list(le.classes_)}")

    print("\nExtracting features (this takes ~2 minutes)...")
    X = build_feature_matrix(df["url"].tolist())
    y = df["label_enc"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {X_train.shape[1]}")

    class_counts   = np.bincount(y_train)
    sample_weights = np.array([1.0 / class_counts[l] for l in y_train])
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)

    objective       = "binary:logistic" if num_classes == 2 else "multi:softprob"
    eval_metric     = "auc"             if num_classes == 2 else "mlogloss"
    extra_params    = {}                if num_classes == 2 else {"num_class": num_classes}

    print(f"\nTraining XGBoost ({objective})...")
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective=objective,
        eval_metric=eval_metric,
        early_stopping_rounds=25,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
        **extra_params
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    print("\nEvaluating on test set...")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print(classification_report(y_test, y_pred, target_names=list(le.classes_)))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if num_classes == 2:
        auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    print(f"ROC-AUC: {auc:.4f}")

    joblib.dump(model,         f"{MODEL_DIR}/url_classifier.pkl")
    joblib.dump(le,            f"{MODEL_DIR}/url_label_encoder.pkl")
    joblib.dump(FEATURE_NAMES, f"{MODEL_DIR}/url_feature_names.pkl")

    print(f"\nSaved to {MODEL_DIR}/")
    print(f"Features: {len(FEATURE_NAMES)}")


if __name__ == "__main__":
    main()
