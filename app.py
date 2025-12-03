from flask import Flask, render_template, request

import nltk
import numpy as np
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

application = Flask(__name__)

# -------------------------------------------------------------------
# Load and prepare corpus
# -------------------------------------------------------------------

with open("M.txt", "r", errors="ignore") as f:
    raw = f.read().lower()

nltk.download("punkt")
nltk.download("wordnet")

lemmer = nltk.stem.WordNetLemmatizer()
remove_punct_dict = dict((ord(p), None) for p in string.punctuation)

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Split corpus into paragraphs (separated by blank lines)
# and clean internal newlines to avoid extra blank lines in responses
paragraphs = [
    p.replace("\n", " ").strip()
    for p in raw.split("\n\n")
    if p.strip()
]

# -------------------------------------------------------------------
# Vector store using TF-IDF (precomputed once)
# -------------------------------------------------------------------

TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
tfidf_matrix = TfidfVec.fit_transform(paragraphs)

# -------------------------------------------------------------------
# OPTIONAL: SBERT / BERT-style semantic vector store (commented out)
# -------------------------------------------------------------------
# from sentence_transformers import SentenceTransformer, util
# sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
# sbert_embeddings = sbert_model.encode(paragraphs, convert_to_tensor=True)
#
# def semantic_response(user_response):
#     query_emb = sbert_model.encode(user_response, convert_to_tensor=True)
#     cos_scores = util.cos_sim(query_emb, sbert_embeddings)[0]
#     best_idx = int(np.argmax(cos_scores))
#     max_score = float(cos_scores[best_idx])
#     if max_score <= 0:
#         return "I am sorry, I don't understand you."
#     return paragraphs[best_idx]

# -------------------------------------------------------------------
# Response function using TF-IDF vector store
# -------------------------------------------------------------------

def response(user_response):
    query_vec = TfidfVec.transform([user_response])
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]

    best_idx = np.argmax(sims)
    max_sim = sims[best_idx]

    if max_sim == 0:
        return "I am sorry, I don't understand you."
    return paragraphs[best_idx]

# -------------------------------------------------------------------
# Flask routes
# -------------------------------------------------------------------

@application.route("/")
def home():
    return render_template("index.html")

@application.route("/get")
def get_bot_response():
    user_text = request.args.get("msg", "").strip().lower()

    if user_text in ["thanks", "thank you"]:
        return "You are welcome."

    # res = semantic_response(user_text)  # if you switch to SBERT
    res = response(user_text)             # TF-IDF-based response

    return str(res)

if __name__ == "__main__":
    application.run()

