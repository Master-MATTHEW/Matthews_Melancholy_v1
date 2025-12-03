from flask import Flask, render_template, request
import os
import nltk
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Do NOT call nltk.download() on Heroku
# nltk.download('punkt')
# nltk.download('wordnet')

lemmer = nltk.stem.WordNetLemmatizer()
remove_punct_dict = dict((ord(p), None) for p in string.punctuation)

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(BASE_DIR, "M.txt")

try:
    with open(CORPUS_PATH, "r", errors="ignore") as f:
        raw = f.read().lower()
except FileNotFoundError:
    raw = ""
    print("WARNING: M.txt not found. Corpus is empty.")

paragraphs = [
    p.replace("\n", " ").strip()
    for p in raw.split("\n\n")
    if p.strip()
]

if paragraphs:
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf_matrix = TfidfVec.fit_transform(paragraphs)
else:
    TfidfVec = None
    tfidf_matrix = None

def response(user_response):
    if not paragraphs or TfidfVec is None or tfidf_matrix is None:
        return "Corpus is not available right now."

    query_vec = TfidfVec.transform([user_response])
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]
    best_idx = int(np.argmax(sims))
    max_sim = float(sims[best_idx])

    if max_sim == 0:
        return "I am sorry, I don't understand you."
    return paragraphs[best_idx]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_text = request.args.get("msg", "").strip().lower()
    if user_text in ["thanks", "thank you"]:
        return "You are welcome."
    return response(user_text)

if __name__ == "__main__":
    app.run()
