import os
import re
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
from collections import Counter
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

MODEL_DIR    = "/kaggle/working/models"
DATASET_PATH = "/kaggle/input/datasets/sid321axn/malicious-urls-dataset/malicious_phish.csv"
OUTPUT_DIR   = "/kaggle/working/plots/url"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model         = joblib.load(f"{MODEL_DIR}/url_classifier.pkl")
label_encoder = joblib.load(f"{MODEL_DIR}/url_label_encoder.pkl")
feature_names = joblib.load(f"{MODEL_DIR}/url_feature_names.pkl")
classes       = list(label_encoder.classes_)
is_binary     = len(classes) == 2
benign_idx    = classes.index("benign") if "benign" in classes else 0

KNOWN_BRANDS = [
    "paypal","amazon","apple","google","microsoft","facebook","netflix",
    "instagram","twitter","linkedin","dropbox","github","whatsapp","zoom",
    "wellsfargo","bankofamerica","chase","citibank","hsbc","dhl","fedex",
    "ups","usps","irs","ebay","walmart"
]
SUSPICIOUS_EXTENSIONS    = [".exe",".zip",".bat",".js",".vbs",".ps1",".msi",".dmg",".apk"]
SUSPICIOUS_PATH_KEYWORDS = [
    "login","signin","sign-in","verify","secure","account","update","confirm",
    "banking","authenticate","validation","recover","password","credential","submit","checkout"
]
TLD_RISK = {
    ".tk":0.95,".ml":0.95,".ga":0.95,".cf":0.95,".gq":0.95,
    ".xyz":0.70,".top":0.70,".club":0.70,".online":0.70,".site":0.70,
    ".info":0.65,".biz":0.60,".co":0.40,".io":0.35,
    ".com":0.15,".net":0.18,".org":0.15,".gov":0.02,".edu":0.02,
    ".uk":0.20,".in":0.22,".de":0.18,".ru":0.55,".cn":0.50,
}
URL_SHORTENERS = {"bit.ly","tinyurl.com","goo.gl","t.co","ow.ly","is.gd","buff.ly","cutt.ly"}
PAL = ["#55A868","#C44E52","#DD8452","#4C72B0"]


def shannon_entropy(s):
    if not s: return 0.0
    freq = Counter(s); n = len(s)
    return -sum((c/n)*math.log2(c/n) for c in freq.values())

def is_ip(h):
    m = re.match(r"^(\d{1,3}\.){3}\d{1,3}$", h or "")
    return bool(m) and all(0 <= int(p) <= 255 for p in h.split("."))

def get_reg_dom(h):
    p = h.split(".")
    return ".".join(p[-2:]) if len(p) >= 2 else h

def extract_features(url):
    if not isinstance(url, str): url = ""
    url = url.strip()
    try:
        parsed   = urlparse(url if "://" in url else "http://"+url)
        scheme   = parsed.scheme.lower()
        hostname = parsed.hostname or ""
        path     = parsed.path or ""
        query    = parsed.query or ""
        netloc   = parsed.netloc or ""
    except Exception:
        hostname,scheme,path,query,netloc = "","http","","",""
    full    = url.lower()
    reg_dom = get_reg_dom(hostname.lower())
    sub     = hostname.lower().replace(reg_dom,"").strip(".")
    tld     = "."+reg_dom.split(".")[-1] if "." in reg_dom else ""
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
    d = sum(c.isdigit() for c in url)
    s = sum(not c.isalnum() and c not in "/:.-_?=&#@%" for c in url)
    f["digit_count"]          = d
    f["digit_ratio"]          = d/max(len(url),1)
    f["special_char_count"]   = s
    f["special_char_ratio"]   = s/max(len(url),1)
    f["url_entropy"]          = round(shannon_entropy(full),4)
    f["domain_entropy"]       = round(shannon_entropy(hostname.lower()),4)
    f["is_ip_hostname"]       = int(is_ip(hostname))
    f["is_https"]             = int(scheme=="https")
    f["is_shortener"]         = int(reg_dom in URL_SHORTENERS)
    f["subdomain_count"]      = len([x for x in sub.split(".") if x]) if sub else 0
    f["subdomain_length"]     = len(sub)
    f["tld_risk_score"]       = TLD_RISK.get(tld,0.80)
    pl = path.lower()
    f["suspicious_keywords"]  = int(any(k in pl or k in query.lower() for k in SUSPICIOUS_PATH_KEYWORDS))
    f["suspicious_ext"]       = int(any(pl.endswith(e) for e in SUSPICIOUS_EXTENSIONS))
    f["digit_ratio_domain"]   = sum(c.isdigit() for c in hostname)/max(len(hostname),1)
    f["non_ascii_ratio"]      = sum(1 for c in hostname if ord(c)>127)/max(len(hostname),1)
    f["has_hex_encoding"]     = int(bool(re.search(r"%[0-9a-fA-F]{2}",url)))
    f["path_depth"]           = len([p for p in path.split("/") if p])
    words = re.findall(r"[a-zA-Z]{3,}",path)
    f["longest_word_path"]    = max((len(w) for w in words),default=0)
    bi = any(b in full for b in KNOWN_BRANDS)
    bd = any(b in reg_dom for b in KNOWN_BRANDS)
    f["brand_impersonation"]  = int(bi and not bd)
    f["unique_char_ratio"]    = len(set(url))/max(len(url),1)
    return f

def build_X(urls):
    return pd.DataFrame([extract_features(u) for u in urls], columns=feature_names).values.astype(np.float32)

def get_confidence(prob_row):
    if is_binary:
        return round(float(prob_row[1-benign_idx])*100, 1)
    return round((1-float(prob_row[benign_idx]))*100, 1) if benign_idx >= 0 else round(float(prob_row[1])*100, 1)


print("Loading dataset and rebuilding test split...")
df           = pd.read_csv(DATASET_PATH)
df.columns   = [c.strip().lower() for c in df.columns]
url_col      = next(c for c in df.columns if "url"  in c)
type_col     = next(c for c in df.columns if "type" in c or "label" in c)
df           = df[[url_col,type_col]].dropna().drop_duplicates(subset=[url_col])
df.columns   = ["url","label"]
df           = df.sample(frac=1,random_state=42).reset_index(drop=True)
df["label_enc"] = label_encoder.transform(df["label"])

split   = int(len(df)*0.8)
test    = df.iloc[split:].reset_index(drop=True)
print(f"Test set: {len(test)} URLs | Classes: {classes}")

print("Extracting features from test set...")
X_test  = build_X(test["url"].tolist())
y_true  = test["label_enc"].values
y_proba = model.predict_proba(X_test)
y_pred  = model.predict(X_test)
mal_proba = np.array([get_confidence(r)/100 for r in y_proba])

print(classification_report(y_true, y_pred, target_names=classes))


# ── GRAPH 1: Confusion Matrix ──────────────────────────────
fig, ax = plt.subplots(figsize=(max(6, len(classes)*2), max(5, len(classes)*2)))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes,
            ax=ax, linewidths=1.5, linecolor="white", cbar=False,
            annot_kws={"size":11,"weight":"bold"})
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("URL Classifier — Confusion Matrix", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/01_confusion_matrix.png"); plt.show()


# ── GRAPH 2: ROC Curve ────────────────────────────────────
if is_binary:
    fpr,tpr,_ = roc_curve(y_true, mal_proba)
    auc       = roc_auc_score(y_true, mal_proba)
    fig, ax   = plt.subplots(figsize=(7,6))
    ax.plot(fpr,tpr,lw=2.5,color="#4C72B0",label=f"AUC = {auc:.4f}")
    ax.plot([0,1],[0,1],"--",color="gray",lw=1.2)
    ax.fill_between(fpr,tpr,alpha=0.08,color="#4C72B0")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — URL Classifier", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/02_roc_curve.png"); plt.show()
    print(f"ROC-AUC: {auc:.4f}")
else:
    auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    print(f"Multi-class ROC-AUC (OvR weighted): {auc:.4f}")


# ── GRAPH 3: Class Distribution ───────────────────────────
fig, ax = plt.subplots(figsize=(7,5))
counts  = df["label"].value_counts()
cols    = PAL[:len(counts)]
bars    = ax.bar(counts.index, counts.values, color=cols, edgecolor="white", width=0.5)
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+500,
            f"{int(bar.get_height()):,}", ha="center", fontsize=10)
ax.set_title("Dataset Class Distribution", fontsize=13, fontweight="bold")
ax.set_ylabel("Count")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/03_class_distribution.png"); plt.show()


# ── GRAPH 4: Top 20 XGBoost Feature Importances ───────────
importances = model.feature_importances_
top_idx     = np.argsort(importances)[-20:]
fig, ax     = plt.subplots(figsize=(9,8))
bar_cols    = ["#C44E52" if importances[i] > np.median(importances) else "#4C72B0" for i in top_idx]
ax.barh([feature_names[i] for i in top_idx], importances[top_idx], color=bar_cols, edgecolor="white", height=0.65)
ax.set_xlabel("Feature Importance (gain)", fontsize=12)
ax.set_title("Top 20 Most Important URL Features", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/04_feature_importance.png"); plt.show()


# ── GRAPH 5: Confidence Distribution ──────────────────────
fig, ax = plt.subplots(figsize=(9,5))
for idx, cls in enumerate(classes):
    mask   = y_true == idx
    scores = y_proba[mask, idx]
    ax.hist(scores, bins=50, alpha=0.6, color=PAL[idx%len(PAL)], label=cls, density=True)
ax.axvline(x=0.5, color="black", linestyle="--", lw=1.5, label="Threshold (0.5)")
ax.set_xlabel("Predicted Class Probability"); ax.set_ylabel("Density")
ax.set_title("Confidence Distribution by True Class", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/05_confidence_distribution.png"); plt.show()


# ── GRAPH 6: URL Length by Class ──────────────────────────
df["url_length"] = df["url"].apply(len).clip(upper=300)
fig, ax = plt.subplots(figsize=(9,5))
for idx, (cls, grp) in enumerate(df.groupby("label")):
    ax.hist(grp["url_length"], bins=60, alpha=0.55, color=PAL[idx%len(PAL)], label=cls, density=True)
ax.set_xlabel("URL Length (clipped at 300)"); ax.set_ylabel("Density")
ax.set_title("URL Length Distribution by Class", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/06_url_length.png"); plt.show()


# ── GRAPH 7: Entropy Distribution ─────────────────────────
sample = df.sample(n=min(5000, len(df)), random_state=42).copy()
sample["entropy"] = sample["url"].apply(lambda u: shannon_entropy(u.lower()))
fig, ax = plt.subplots(figsize=(9,5))
for idx, (cls, grp) in enumerate(sample.groupby("label")):
    ax.hist(grp["entropy"], bins=50, alpha=0.6, color=PAL[idx%len(PAL)], label=cls, density=True)
ax.axvline(x=3.5, color="black", linestyle="--", lw=1.5, label="High entropy (3.5)")
ax.set_xlabel("Shannon Entropy"); ax.set_ylabel("Density")
ax.set_title("URL Entropy Distribution\n(high entropy = likely DGA/malware domain)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/07_entropy_distribution.png"); plt.show()


# ── GRAPH 8: SHAP Summary ─────────────────────────────────
try:
    import shap
    X_shap      = X_test[:min(500, len(X_test))]
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        target   = 1 if len(shap_values) == 2 else min(1, len(shap_values)-1)
        shap_mat = shap_values[target]
    else:
        shap_mat = shap_values if shap_values.ndim == 2 else shap_values[:,:,1]
    mean_abs = np.abs(shap_mat).mean(axis=0)
    top_s    = np.argsort(mean_abs)[-20:]
    fig, ax  = plt.subplots(figsize=(9,8))
    cols_s   = ["#C44E52" if mean_abs[i] > np.median(mean_abs) else "#4C72B0" for i in top_s]
    ax.barh([feature_names[i] for i in top_s], mean_abs[top_s], color=cols_s, edgecolor="white", height=0.65)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("SHAP Feature Importance — Top 20", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/08_shap_importance.png"); plt.show()
    print("Saved: 08_shap_importance.png")
except Exception as e:
    print(f"SHAP skipped: {e}")


# ── GRAPH 9: Adversarial Robustness ───────────────────────
ADVERSARIAL_TESTS = [
    ("https://www.google.com",                             "baseline safe",               False),
    ("http://paypa1.com/verify-account",                   "homograph digit substitution", True),
    ("http://secure.paypal.com.verify.tk/login",           "subdomain brand abuse",        True),
    ("http://192.168.1.1/admin/login.php",                 "private IP in URL",            True),
    ("http://xkqmvzpjth.tk/index.php",                    "DGA high entropy domain",      True),
    ("https://bit.ly/3xK9mZp",                            "URL shortener hidden dest",    True),
    ("http://amazon-secure-login.verify-account.ml/auth", "brand impersonation + bad TLD", True),
    ("http://g00gle.com/login?redirect=paypal.com",        "homograph + brand redirect",   True),
    ("http://update-your-account.info/confirm/banking",    "keyword heavy + bad TLD",      True),
    ("http://normal-blog.com/2024/01/my-post-today",       "benign long path",             False),
]

adv_results = []
for url, attack, expect_malicious in ADVERSARIAL_TESTS:
    fvec = build_X([url])
    prob = model.predict_proba(fvec)[0]
    conf = get_confidence(prob)
    caught = (conf >= 50) == expect_malicious
    adv_results.append((attack, conf, caught, expect_malicious))

fig, ax = plt.subplots(figsize=(12,6))
attacks   = [r[0] for r in adv_results]
confs     = [r[1] for r in adv_results]
bar_cols  = ["#C44E52" if c >= 50 else "#55A868" for c in confs]
bars = ax.barh(attacks, confs, color=bar_cols, edgecolor="white", height=0.55)
ax.axvline(x=50, color="black", linestyle="--", lw=1.5, label="Decision threshold (50%)")
for bar, conf, result in zip(bars, confs, adv_results):
    label = f"{conf}%  {'CAUGHT' if result[2] else 'MISSED'}"
    ax.text(min(conf+1, 97), bar.get_y()+bar.get_height()/2,
            label, va="center", fontsize=8.5,
            color="#27500A" if result[2] else "#A32D2D")
ax.set_xlabel("Malicious Confidence Score (%)", fontsize=12)
ax.set_title("Adversarial Robustness Test — 10 Attack Patterns", fontsize=13, fontweight="bold")
ax.set_xlim([0, 115])
ax.legend(fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/09_adversarial_robustness.png"); plt.show()

caught_n = sum(r[2] for r in adv_results)
print(f"\nAdversarial catch rate: {caught_n}/{len(adv_results)}")
for attack, conf, caught, expect in adv_results:
    status = "CAUGHT" if caught else "MISSED"
    print(f"  [{status}]  {attack:<45}  confidence: {conf}%")


print(f"\nAll graphs saved to {OUTPUT_DIR}/")
print(f"Features: {len(feature_names)} | Classes: {classes}")
