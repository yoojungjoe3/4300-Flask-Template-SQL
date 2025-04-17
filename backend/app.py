#Confirm: Are the search results rankings changed based on the user feedback thru the like and dislike buttons?
import json
import os
import re
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
# import sqlalchemy as db 
# engine = db.create_engine("mysql+pymysql://admin:admin@4300showcase.infosci.cornell.edu/kardashiandb")
# cursor = engine.connect()

# Set ROOT_PATH for linking with all your files.
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Database credentials (adjust if needed)
LOCAL_MYSQL_USER = "root"
LOCAL_MYSQL_USER_PASSWORD = "bobbob"
LOCAL_MYSQL_PORT = 3306
LOCAL_MYSQL_DATABASE = "kardashiandb"

# Initialize database handler and load init.sql into the database
mysql_engine = MySQLDatabaseHandler(LOCAL_MYSQL_USER, LOCAL_MYSQL_USER_PASSWORD, LOCAL_MYSQL_PORT, LOCAL_MYSQL_DATABASE)
mysql_engine.load_file_into_db()

app = Flask(__name__)
#Flask session secret key
app.secret_key = 'bobby24'
CORS(app)

#Using the Flask session object
#setting session data
session['feedback'] = [{"doc_index": 1, "feedback": 1}]
#accessing session data
data = session.get('feedback', [])
#clearing session data
session.pop('feedback', None)

current_dir = os.path.dirname(os.path.abspath(__file__))
init_sql_path = os.path.join(current_dir, "..", "init.sql")

# Precomputed models will be stored here:
precomputed = {}

def precompute_field(field_texts, n_components=100):
    """
    Given a list of texts, create and return:
      - a TfidfVectorizer (configured with analyzer='char_wb', ngram_range=(3,5), stop_words='english')
      - a fitted TruncatedSVD model (with a proper n_components)
      - the SVD-reduced matrix for the field texts.
    Replace empty texts with a single space.
    """
    # Ensure no field is empty:
    texts = [text if text.strip() != "" else " " for text in field_texts]
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    # Ensure n_components is at least 1 and does not exceed available features - 1:
    n_comp = max(1, min(n_components, tfidf_matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp)
    reduced_matrix = svd.fit_transform(tfidf_matrix)
    return {"vectorizer": vectorizer, "svd": svd, "matrix": reduced_matrix}

def initialize_precomputed():
    """
    Loads all entries from the database and precomputes the TF-IDF + SVD representations
    for fields: names, fandoms, ships, reviews, abstracts.
    Also stores the raw lists for later reconstruction.
    """
    query = "SELECT Name, Fandom, Ships, Rating, Link, Review, Abstract FROM fics;"
    rows = list(mysql_engine.query_selector(query))
    # Extract raw fields:
    names     = [r[0] for r in rows]
    fandoms   = [r[1] for r in rows]
    ships     = [r[2] for r in rows]
    ratings   = [r[3] for r in rows]
    links     = [r[4] for r in rows]
    reviews   = [r[5] for r in rows]
    abstracts = [r[6] for r in rows]

    global precomputed
    precomputed['names']     = precompute_field(names)
    precomputed['fandoms']   = precompute_field(fandoms)
    precomputed['ships']     = precompute_field(ships)
    precomputed['reviews']   = precompute_field(reviews)
    precomputed['abstracts'] = precompute_field(abstracts)

    # Store raw versions to reconstruct final Entry objects:
    precomputed['names_raw']     = names
    precomputed['fandoms_raw']   = fandoms
    precomputed['ships_raw']     = ships
    precomputed['ratings']       = ratings
    precomputed['links']         = links
    precomputed['reviews_raw']   = reviews
    precomputed['abstracts_raw'] = abstracts

# Precompute on startup
initialize_precomputed()

#function creates object in the format that we want printed out
class Entry:
    def __init__(self, name, ship, fandom, rating, abstract, link):
        self.name = name
        self.ship = ship
        self.fandom = fandom
        self.rating = rating
        self.abstract = abstract
        self.link = link
        if fandom == ('"Harry Potter"') or fandom == ('Harry Potter'):
            self.image = "/static/images/dumbly.jpg"
        elif fandom == ('"Kardashians"'):
            self.image ="/static/images/KIM.jpg"
        elif fandom == ('"Merlin"'):
            self.image ="/static/images/Merlin.jpg"
        elif fandom == ('"One Direction"'):
            self.image = "/static/images/OneD.jpeg"
        elif fandom == ('"Hunger Games"'):
            self.image = "/static/images/HG.jpeg"
        elif fandom == ('"The Princess Diaries"'):
            self.image = "/static/images/PD.jpg"
        else:
            self.image = "/static/images/fandom.jpeg"

    def to_dict(self):
        return {
            "name": self.name,
            "ship": self.ship,
            "fandom": self.fandom,
            "rating": self.rating,
            "abstract": self.abstract,
            "link": self.link,
            "image": self.image
        }

    def __repr__(self):
        return f"Entry(Name: {self.name}, Ships: {self.ship}, Fandoms: {self.fandom}, Ratings: {self.rating}, Abstracts: {self.abstract}, Links: {self.link}, Image: {self.image})"

def compute_svd_similarity(texts, query, n_components=100, return_matrix=False):
    """
    Compute cosine similarity between a user query and a list of texts
    using TF-IDF vectorization followed by SVD dimensionality reduction.
    """
    # Replace empty or whitespace-only texts with a placeholder (space)
    texts = [text if text.strip() != "" else " " for text in texts]

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts + [query])
    
    # Reduce number of dimensions to capture semantic patterns
    n_components = max(1, min(n_components, tfidf_matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components)
    reduced_matrix = svd.fit_transform(tfidf_matrix)
    
    query_vector = reduced_matrix[-1]
    candidate_vectors = reduced_matrix[:-1]
    similarities = cosine_similarity([query_vector], candidate_vectors).flatten()

    if return_matrix:
        return similarities, query_vector, candidate_vectors
    else:
        return similarities
    
#Rocchio feedback function for implementing user feedback through likes and dislikes
def apply_rocchio_feedback(query_vec, doc_matrix):
    feedback = session.pop('feedback', [])
    liked = [f['doc_index'] for f in feedback if f['feedback'] == 1]
    disliked = [f['doc_index'] for f in feedback if f['feedback'] == -1]

    alpha = 1.0
    beta = 0.75
    gamma = 0.25

    if not liked and not disliked:
        return query_vec

    adjustment = np.zeros_like(query_vec)
    valid_liked = [i for i in liked if 0 <= i < len(doc_matrix)]
    valid_disliked = [i for i in disliked if 0 <= i < len(doc_matrix)]
    if valid_liked:
        liked_vecs = doc_matrix[valid_liked]
        adjustment += beta * liked_vecs.mean(axis=0)
    if disliked:
        disliked_vecs = doc_matrix[disliked]
        adjustment -= gamma * disliked_vecs.mean(axis=0)

    return alpha * query_vec + adjustment

def compute_precomputed_similarity(precomputed_matrix, query_text, query_vector=None):
    """
    Given a precomputed dictionary (with vectorizer, svd, and matrix)
    and an injected query vector, transform the query and compute cosine similarities.
    Returns a 1D array of cosine similarities.
    """
    if query_vector is None:
        tfidf_vec = precomputed['vectorizer'].transform([query_text])
        query_vector = precomputed['svd'].transform(tfidf_vec)
    return cosine_similarity(query_vector, precomputed_matrix).flatten()

def SVD_vector_search(user_query):
    """
    Compute combined similarity scores using precomputed TF-IDF + SVD representations.
    Applies field weights: Name (3.0), Fandom (2.0), Ship (1.5), Abstract (1.0), Review (1.0).
    Returns a list of Entry objects for the best matches.
    """
    cleaned_query = clean_text(user_query)

    # Generate query vector from TF-IDF + SVD
    tfidf_vec = precomputed['vectorizer'].transform([cleaned_query])
    query_vector = precomputed['svd'].transform(tfidf_vec)

    # Apply Rocchio feedback based on one field (e.g., fandoms)
    query_vector = apply_rocchio_feedback(query_vector, precomputed['fandoms'])

    # Now compute similarities using the adjusted query vector
    sim_names     = compute_precomputed_similarity(precomputed['names'], cleaned_query, query_vector)
    sim_fandoms   = compute_precomputed_similarity(precomputed['fandoms'], cleaned_query, query_vector)
    sim_ships     = compute_precomputed_similarity(precomputed['ships'], cleaned_query, query_vector)
    sim_abstracts = compute_precomputed_similarity(precomputed['abstracts'], cleaned_query, query_vector)
    sim_reviews   = compute_precomputed_similarity(precomputed['reviews'], cleaned_query, query_vector)
    # Set weights for each field
    weight_names     = 3.0
    weight_fandoms   = 2.0
    weight_ships     = 1.5
    weight_abstracts = 1.0
    weight_reviews   = 1.0

    # Combine weighted similarities (elementwise sum)
    combined_similarities = (
        weight_names     * sim_names +
        weight_fandoms   * sim_fandoms +
        weight_ships     * sim_ships +
        weight_abstracts * sim_abstracts +
        weight_reviews   * sim_reviews
    )

    # Create dictionary mapping record index (starting at 1) to similarity score
    total_sim_dict = {i + 1: float(score) for i, score in enumerate(combined_similarities)}
    sorted_keys = sorted(total_sim_dict, key=total_sim_dict.get, reverse=True)
    # print("total_sim_dict" + str(total_sim_dict))
    # print("sorted_keys" + str(sorted_keys))


    # Optional: filter by threshold relative to average nonzero similarity
    nonzero = [score for score in total_sim_dict.values() if score != 0]
    avg = sum(nonzero)/len(nonzero) if nonzero else 0

    ourentries = []
    for idx in sorted_keys:
        if total_sim_dict[idx] > avg and len(ourentries) < 10:
            i = idx - 1
            entry = Entry(
                precomputed['names_raw'][i],
                precomputed['ships_raw'][i],
                precomputed['fandoms_raw'][i],
                precomputed['ratings'][i],
                precomputed['abstracts_raw'][i],
                precomputed['links'][i]
            )
            ourentries.append(entry)
    return ourentries

 
@app.route("/")
def home():
    return render_template('base.html', Name="sample html")


def clean_text(user_query):
    """Convert text to lowercase and remove punctuation."""
    return re.sub(r'[^\w\s]', '', user_query.lower())


@app.route("/fics")
def fics_search():
    user_query = request.args.get("Name")
    if not user_query:
        return ("Please input a query :)"), 400


    ourentries = SVD_vector_search(user_query)
    ourentries_dicts = [entry.to_dict() for entry in ourentries]

    return jsonify({
        "ourentries": ourentries_dicts,
    })

#the /submit_feedback route
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()

    feedback_data = session.get('feedback', [])

    feedback_data.append({
        "query_text": data['query_text'],
        "doc_index": data['doc_index'],
        "feedback": data['feedback']
    })

    session['feedback'] = feedback_data

    return jsonify({"status": "success", "stored_feedback": feedback_data})
if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
