"""
Fake News Detector - demo project
NLP pipeline: text cleaning -> TF-IDF -> Logistic Regression -> Flask web UI

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""
import re

from flask import Flask, render_template_string, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 1. Tiny demo dataset (label 1 = REAL, 0 = FAKE).
#    Replace with a real dataset, e.g. the Kaggle "Fake and Real News" CSVs,
#    by loading them with pandas in load_data().
# ---------------------------------------------------------------------------
REAL = [
    "The central bank announced a quarter point interest rate change after its scheduled meeting",
    "Researchers at the university published a peer reviewed study on coastal erosion trends",
    "The city council approved the annual budget following a public hearing on Tuesday",
    "Government statistics show unemployment fell slightly in the last quarter",
    "The health ministry confirmed vaccination drives will begin in rural districts next month",
    "Engineers completed inspection of the bridge and reported no structural issues",
    "The national cricket team won the series after a close final match on Sunday",
    "Scientists reported new measurements of rainfall patterns across the monsoon season",
    "The railway department announced new train schedules effective from the first of next month",
    "Officials said the exam results will be released on the university website this week",
    "The company reported quarterly earnings in line with analyst expectations",
    "Local authorities opened a new public library after two years of construction",
    "The space agency successfully launched a weather observation satellite on Thursday",
    "Doctors recommend regular exercise and balanced diet according to a new health survey",
    "The election commission published the final voter list ahead of the polling date",
]
FAKE = [
    "SHOCKING secret cure doctors don't want you to know about, share before it gets deleted",
    "You won't believe what this celebrity did, the media is hiding the truth from you",
    "Aliens have secretly taken over the government and insiders are finally revealing everything",
    "Drink this miracle water and lose twenty kilos in one night, scientists are furious",
    "BREAKING: free phones for everyone, just click this link and forward to ten friends",
    "Government putting mind control chips in all vaccines, whistleblower exposes the plot",
    "Moon landing was filmed in a studio and NASA finally admits it in leaked video",
    "Eat this one fruit and never get sick again, big pharma hates this trick",
    "Famous actor dead in tragic accident, forward this message to stop the curse",
    "Banks will shut down tomorrow and your money will vanish, withdraw everything now",
    "Secret society controls the weather and causes every flood, shocking proof inside",
    "Click now to claim your lottery prize, you have been randomly selected winner",
    "Scientists confirm the earth is flat, mainstream media silenced the story",
    "Unbelievable! This weird home remedy cures every disease instantly, doctors stunned",
    "Leaked documents prove the whole election was fake, they don't want you to see this",
]


def load_data():
    texts = REAL + FAKE
    labels = [1] * len(REAL) + [0] * len(FAKE)
    return texts, labels


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# 2. Train the model
# ---------------------------------------------------------------------------
texts, labels = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

model = Pipeline(
    [
        ("tfidf", TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)
model.fit(X_train, y_train)
ACCURACY = accuracy_score(y_test, model.predict(X_test))
print(f"Held-out accuracy on demo data: {ACCURACY:.0%}")


def explain(text: str, top_n: int = 5):
    """Return the words in the text that pushed the prediction the most."""
    vec = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    row = vec.transform([text])
    names = vec.get_feature_names_out()
    contributions = row.toarray()[0] * clf.coef_[0]
    idx = [i for i in contributions.argsort() if contributions[i] != 0]
    fake_words = [names[i] for i in idx[:top_n] if contributions[i] < 0]
    real_words = [names[i] for i in reversed(idx[-top_n:]) if contributions[i] > 0]
    return fake_words, real_words


# ---------------------------------------------------------------------------
# 3. Web UI
# ---------------------------------------------------------------------------
app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fake News Detector</title>
<style>
  body{font-family:system-ui,sans-serif;background:#0f1115;color:#e8eaed;margin:0;padding:24px}
  .card{max-width:680px;margin:40px auto;background:#181b22;border:1px solid #2a2f3a;border-radius:14px;padding:28px}
  h1{margin-top:0;font-size:1.6rem}
  textarea{width:100%;height:150px;background:#0f1115;color:#e8eaed;border:1px solid #2a2f3a;border-radius:8px;padding:12px;font-size:1rem;box-sizing:border-box}
  button{margin-top:12px;background:#6c8cff;color:#fff;border:0;border-radius:8px;padding:10px 22px;font-size:1rem;cursor:pointer}
  .result{margin-top:22px;padding:16px;border-radius:10px}
  .real{background:#12301f;border:1px solid #2e7d4f}
  .fake{background:#3a1717;border:1px solid #a33}
  .bar{height:10px;background:#2a2f3a;border-radius:5px;overflow:hidden;margin:8px 0}
  .bar span{display:block;height:100%;background:#6c8cff}
  small{color:#9aa0ac}
</style>
</head>
<body>
<div class="card">
  <h1>Fake News Detector</h1>
  <small>TF-IDF + Logistic Regression &middot; demo accuracy {{ acc }} on a tiny sample dataset</small>
  <form method="post">
    <textarea name="text" placeholder="Paste a headline or news paragraph...">{{ text }}</textarea>
    <button type="submit">Analyze</button>
  </form>
  {% if label %}
  <div class="result {{ 'real' if label == 'REAL' else 'fake' }}">
    <strong>Prediction: {{ label }}</strong>
    <div class="bar"><span style="width: {{ conf }}%"></span></div>
    Confidence: {{ conf }}%
    {% if fake_words %}<p><small>Words pushing toward FAKE: {{ fake_words|join(', ') }}</small></p>{% endif %}
    {% if real_words %}<p><small>Words pushing toward REAL: {{ real_words|join(', ') }}</small></p>{% endif %}
  </div>
  {% endif %}
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {"text": "", "label": None, "conf": 0, "fake_words": [], "real_words": [], "acc": f"{ACCURACY:.0%}"}
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        ctx["text"] = text
        if text:
            proba = model.predict_proba([text])[0]
            pred = int(proba.argmax())
            ctx["label"] = "REAL" if pred == 1 else "FAKE"
            ctx["conf"] = round(float(proba.max()) * 100, 1)
            ctx["fake_words"], ctx["real_words"] = explain(text)
    return render_template_string(PAGE, **ctx)


if __name__ == "__main__":
    app.run(debug=True)