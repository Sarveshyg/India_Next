import os
import re
import base64
import joblib
import numpy as np
import scipy.sparse as sp
from lime.lime_text import LimeTextExplainer

MODEL_DIR = "/kaggle/working/models"

vectorizer = joblib.load(os.path.join(MODEL_DIR, "email_vectorizer.pkl"))
classifier = joblib.load(os.path.join(MODEL_DIR, "email_classifier.pkl"))
scaler     = joblib.load(os.path.join(MODEL_DIR, "email_scaler.pkl"))

URGENCY_KEYWORDS = [
    "urgent", "immediately", "suspended", "verify", "confirm", "expire",
    "limited time", "act now", "click here", "account blocked", "unusual activity",
    "security alert", "update required", "validate", "restricted", "locked"
]

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

def handcrafted_features(raw_text):
    if not isinstance(raw_text, str):
        raw_text = ""
    url_count     = len(re.findall(r"http\S+|www\S+", raw_text, re.I))
    img_count     = len(re.findall(r"<img", raw_text, re.I))
    exclaim_count = raw_text.count("!")
    words         = raw_text.split()
    word_count    = max(len(words), 1)
    caps_ratio    = sum(1 for w in words if w.isupper()) / word_count
    has_html      = int(bool(re.search(r"<[a-z][\s\S]*?>", raw_text, re.I)))
    urgency_flag  = int(any(kw in raw_text.lower() for kw in URGENCY_KEYWORDS))
    link_ratio    = min(url_count / max(word_count / 50, 1), 1.0)
    return np.array([[
        url_count, img_count, exclaim_count / word_count,
        caps_ratio, has_html, urgency_flag, link_ratio, word_count
    ]], dtype=np.float32)

def risk_tier(confidence):
    if confidence >= 86: return "CRITICAL"
    if confidence >= 61: return "HIGH"
    if confidence >= 31: return "MEDIUM"
    return "LOW"

def predict_fn_for_lime(texts):
    cleaned = [clean_text(t) for t in texts]
    tfidf   = vectorizer.transform(cleaned)
    hand    = np.array([handcrafted_features(t)[0] for t in texts], dtype=np.float32)
    hand_sc = scaler.transform(hand)
    combined = sp.hstack([tfidf, sp.csr_matrix(hand_sc)])
    return classifier.predict_proba(combined)

def analyze(raw_text, label=None):
    clean   = clean_text(raw_text)
    tfidf   = vectorizer.transform([clean])
    hand    = handcrafted_features(raw_text)
    hand_sc = scaler.transform(hand)
    X       = sp.hstack([tfidf, sp.csr_matrix(hand_sc)])

    proba      = classifier.predict_proba(X)[0]
    confidence = round(float(proba[1]) * 100, 1)
    if len(clean.split()) < 20:
        confidence = min(confidence, 70.0)
    prediction = "PHISHING" if confidence >= 50 else "SAFE"
    tier       = risk_tier(confidence)

    explainer = LimeTextExplainer(class_names=["safe", "phishing"])
    exp       = explainer.explain_instance(clean, predict_fn_for_lime, num_features=8, num_samples=200)
    evidence  = [(word, round(float(w), 4)) for word, w in exp.as_list()]

    print("=" * 60)
    if label:
        print(f"  ACTUAL LABEL : {label.upper()}")
    print(f"  PREDICTION   : {prediction}")
    print(f"  CONFIDENCE   : {confidence}%")
    print(f"  RISK TIER    : {tier}")
    print(f"\n  TOP EVIDENCE TOKENS (positive = phishing signal):")
    for token, weight in sorted(evidence, key=lambda x: abs(x[1]), reverse=True):
        bar   = "+" * int(abs(weight) * 30) if weight > 0 else "-" * int(abs(weight) * 30)
        arrow = ">>>" if weight > 0 else "   "
        print(f"    {arrow}  {token:<20} {weight:+.4f}  {bar}")
    print("=" * 60)
    print()

TEST_EMAILS = [
    {
        "label": "phishing",
        "text": """Dear Customer,
Your account has been suspended due to unusual activity.
You must verify your identity immediately or your account will be permanently locked.
Click here to confirm your details: http://secure-paypal-verify.tk/login
This is urgent. Failure to act within 24 hours will result in permanent suspension.
Regards, Security Team"""
    },
    {
        "label": "phishing",
        "text": """CONGRATULATIONS! You have been selected for a $1,000,000 prize.
To claim your reward, confirm your bank account details immediately.
Send your full name, address, and account number to claim@nigerian-lottery-win.com
Act now - this offer expires in 24 hours!"""
    },
    {
        "label": "safe",
        "text": """Hi John,
Just following up on our meeting from yesterday. I have attached the project proposal
for your review. Please let me know if you have any questions or need any changes.
Looking forward to hearing your thoughts.
Best regards, Sarah"""
    },
    {
        "label": "safe",
        "text": """Team,
The sprint planning meeting for next week has been moved to Tuesday at 2pm.
Please update your calendars accordingly. The agenda will be shared by end of day Friday.
Thanks, Manager"""
    },
    {
        "label": "phishing",
        "text": """Your Apple ID has been locked for security reasons.
To restore access verify your account now: http://apple-id-restore.suspicious-site.com
If you do not verify within 12 hours your account will be disabled permanently.
Apple Support"""
    },
    {
        "label": "safe",
        "text": """Hi,
Your order #98234 has been shipped and will arrive by Thursday.
You can track your package using the tracking number provided in the previous email.
Thank you for shopping with us."""
    },
]

print("\nRUNNING MODEL TESTS")
print(f"Loaded vectorizer vocab size : {len(vectorizer.vocabulary_)}")
print(f"Classifier classes           : {classifier.classes_}")
print()

for sample in TEST_EMAILS:
    analyze(sample["text"], label=sample["label"])
