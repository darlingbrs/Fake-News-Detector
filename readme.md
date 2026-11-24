# Fake News Detector (Demo)

A small end-to-end NLP project: text cleaning, TF-IDF features, Logistic Regression, and a Flask web UI that shows the prediction, a confidence score, and the words that influenced it.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 and paste a headline.

## How it works

1. **Clean**: lowercase, strip URLs and punctuation.
2. **TF-IDF**: turn text into weighted word and bigram features (stop words removed).
3. **Logistic Regression**: learns which terms signal fake vs real.
4. **Explain**: multiplies each TF-IDF value by the model coefficient to show the most influential words.

## Make it real

The built-in dataset has only 30 sample headlines, so the accuracy figure is not meaningful. To upgrade:

- Download the Kaggle "Fake and Real News" dataset (`Fake.csv`, `True.csv`).
- Edit `load_data()` in `app.py` to read them with pandas and return `texts, labels`.
- Compare other models (Naive Bayes, Linear SVM) and add a confusion matrix.
- Try embeddings or a fine-tuned transformer as a next step.