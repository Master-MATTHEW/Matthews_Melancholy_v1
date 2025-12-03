#appV1

from flask import Flask, render_template, request
import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

application = Flask(__name__)

# Load and clean corpus
f = open('M.txt', 'r', errors='ignore')
raw = f.read().lower()

nltk.download('punkt')
nltk.download('wordnet')

# Tokenize by sentences and words
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# Lemmatizer
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(p), None) for p in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Get the entire paragraph containing the best-matching sentence
def get_paragraph(sentence):
    para_list = raw.split("\n\n")
    for p in para_list:
        if sentence in p:
            return p.replace("\n", " ")  # remove line breaks but keep paragraph structure
    return sentence

def response(user_response):
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)

    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]

    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        sent_tokens.remove(user_response)
        return "I am sorry, I don't understand you."

    best_sentence = sent_tokens[idx]
    sent_tokens.remove(user_response)

    return get_paragraph(best_sentence)

@application.route("/")
def home():
    return render_template("index.html")

@application.route("/get")
def get_bot_response():
    userText = request.args.get("msg").lower()

    if userText in ["thanks", "thank you"]:
        return "You are welcome."

    res = response(userText)
    return str(res)

if __name__ == "__main__":
    application.run()
