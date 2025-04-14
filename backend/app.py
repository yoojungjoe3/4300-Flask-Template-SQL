import json
import os
import re
import numpy as np
from flask import Flask, render_template, request, jsonify
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
CORS(app)

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

def vector_search(user_query):
    """
    Compute combined similarity scores for the query against both fandoms and ships.
    Also prints the top two fanfic titles.
    Returns a dictionary mapping record number (starting at 1) to the combined similarity score.
    """
    # Clean and prepare the query
    words = clean_text(user_query).split()

    query = "SELECT Name, Fandom, Ships, Rating, Link, Review, Abstract FROM fics;"
    row = list(mysql_engine.query_selector(query))

    names = [r[0] for r in row]
    fandoms = [r[1] for r in row]
    ships = [r[2] for r in row]
    ratings = [r[3] for r in row]
    links = [r[4] for r in row]
    reviews = [r[5] for r in row]
    abstracts = [r[6] for r in row]

    #vectorizer = TfidfVectorizer()
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), stop_words='english')

    # Create a list to store the similarity scores for the query words
    similarity_scores_fandoms = []
    similarity_scores_ships = []
    similarity_scores_abstracts = []
    similarity_scores_reviews = []

    # Iterate through each word in the query and compare with fandoms and ships
    for word in words:
        # fandoms 
        join_fandom_query = fandoms + [word]
        tfidf_matrix_fandoms = vectorizer.fit_transform(join_fandom_query) # PROBLEM LINE!!!
        query_vector_fandoms = tfidf_matrix_fandoms[-1]
        candidate_vectors_fandoms = tfidf_matrix_fandoms[:-1]
        similarities_fandoms = cosine_similarity(query_vector_fandoms, candidate_vectors_fandoms).flatten()
        similarity_scores_fandoms.append(similarities_fandoms)

        # ships 
        join_ship_query = ships + [word]
        tfidf_matrix_ships = vectorizer.fit_transform(join_ship_query)
        query_vector_ships = tfidf_matrix_ships[-1]
        candidate_vectors_ships = tfidf_matrix_ships[:-1]
        similarities_ships = cosine_similarity(query_vector_ships, candidate_vectors_ships).flatten()
        similarity_scores_ships.append(similarities_ships)

        # abstract
        join_abstract_query = abstracts + [word]
        tfidf_matrix_abstracts = vectorizer.fit_transform(join_abstract_query)
        query_vector_abstracts = tfidf_matrix_abstracts[-1]
        candidate_vectors_abstracts = tfidf_matrix_abstracts[:-1]
        similarities_abstracts = cosine_similarity(query_vector_abstracts, candidate_vectors_abstracts).flatten()
        similarity_scores_abstracts.append(similarities_abstracts)

        # reviews
        join_reviews_query = reviews + [word]
        tfidf_matrix_reviews = vectorizer.fit_transform(join_reviews_query)
        query_vector_reviews = tfidf_matrix_reviews[-1]
        candidate_vectors_reviews = tfidf_matrix_reviews[:-1]
        similarities_reviews = cosine_similarity(query_vector_reviews, candidate_vectors_reviews).flatten()
        similarity_scores_reviews.append(similarities_reviews)

    # Combine the similarity scores for each query word (sum of all word similarities)
    combined_fandom_similarities = np.sum(np.array(similarity_scores_fandoms), axis=0)
    combined_ship_similarities = np.sum(np.array(similarity_scores_ships), axis=0)
    combined_abstract_similarities = np.sum(np.array(similarity_scores_abstracts), axis=0)
    combined_review_similarities = np.sum(np.array(similarity_scores_reviews), axis=0)

    # Combine fandom and ship similarities
    combined_similarities = combined_fandom_similarities + combined_ship_similarities + combined_abstract_similarities + combined_review_similarities
    total_sim_dict = {i + 1: total for i, total in enumerate(combined_similarities)}
    # print(total_sim_dict)

    # Sort keys (record indices) by similarity score (highest first)
    sorted_keys = sorted(total_sim_dict, key=total_sim_dict.get, reverse=True)

    ourentries =[]

    nonzero_values = [v for v in total_sim_dict.values() if v != 0]
    if nonzero_values:
        average = sum(nonzero_values) / len(nonzero_values)
    else:
        average = 0

    for x in sorted_keys: 
        if total_sim_dict[x] > average * 2: 
            final_name = names[x-1]
            final_ship = ships[x-1]
            final_fandom = fandoms[x-1]
            final_rating = ratings[x-1]
            final_abstract = abstracts[x-1]
            final_link = links[x-1]


            e = Entry(final_name, final_ship, final_fandom, final_rating, final_abstract, final_link)
            ourentries.append(e)
     
    return ourentries

def compute_svd_similarity0(texts, query, n_components=100):
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
    return similarities

def SVD_vector_search0(user_query):
    """
    Compute combined similarity scores for the query against both fandoms and ships,
    abstracts, and reviews using SVD-reduced TF-IDF vectors.
    Returns a list of Entry objects for the best matches.
    """
    words = clean_text(user_query)

    query = "SELECT Name, Fandom, Ships, Rating, Link, Review, Abstract FROM fics;"
    row = list(mysql_engine.query_selector(query))

    names = [r[0] for r in row]
    fandoms = [r[1] for r in row]
    ships = [r[2] for r in row]
    ratings = [r[3] for r in row]
    links = [r[4] for r in row]
    reviews = [r[5] for r in row]
    abstracts = [r[6] for r in row]

    # Compute similarity scores using SVD-enhanced TF-IDF
    similarity_scores_fandoms = compute_svd_similarity(fandoms, words)
    similarity_scores_ships = compute_svd_similarity(ships, words)
    similarity_scores_abstracts = compute_svd_similarity(abstracts, words)
    similarity_scores_reviews = compute_svd_similarity(reviews, words)
    similarity_scores_names = compute_svd_similarity(names, words)

    # # Combine all similarities into a single score
    # combined_similarities = (
    #     similarity_scores_fandoms +
    #     similarity_scores_ships +
    #     similarity_scores_abstracts +
    #     similarity_scores_reviews + 
    #     similarity_scores_names
    # )

    # Set weights:
    weight_names     = 3.0  
    weight_fandoms   = 2.0
    weight_ships     = 1.5
    weight_abstracts = 1.0
    weight_reviews   = 1.0

    # Combine weighted similarities (elementwise sum)
    combined_similarities = (
        weight_names     * similarity_scores_names +
        weight_fandoms   * similarity_scores_fandoms +
        weight_ships     * similarity_scores_ships +
        weight_abstracts * similarity_scores_abstracts +
        weight_reviews   * similarity_scores_reviews
    )

    total_sim_dict = {i + 1: total for i, total in enumerate(combined_similarities)}
    sorted_keys = sorted(total_sim_dict, key=total_sim_dict.get, reverse=True)

    # Filter and return results above a similarity threshold
    nonzero_values = [v for v in total_sim_dict.values() if v != 0]
    average = sum(nonzero_values) / len(nonzero_values) if nonzero_values else 0

    ourentries = []
    for x in sorted_keys:
        if total_sim_dict[x] > average * 2:
            final_name = names[x - 1]
            final_ship = ships[x - 1]
            final_fandom = fandoms[x - 1]
            final_rating = ratings[x - 1]
            final_abstract = abstracts[x - 1]
            final_link = links[x - 1]

            e = Entry(final_name, final_ship, final_fandom, final_rating, final_abstract, final_link)
            ourentries.append(e)

    return ourentries

def compute_precomputed_similarity(precomputed_obj, query):
    """
    Given a precomputed dictionary (with vectorizer, svd, and matrix)
    and a query, transform the query and compute cosine similarities.
    Returns a 1D array of cosine similarities.
    """
    vectorizer = precomputed_obj["vectorizer"]
    svd = precomputed_obj["svd"]
    matrix = precomputed_obj["matrix"]
    query_tfidf = vectorizer.transform([query])
    query_reduced = svd.transform(query_tfidf)
    return cosine_similarity(query_reduced, matrix).flatten()

def SVD_vector_search(user_query):
    """
    Compute combined similarity scores using precomputed TF-IDF + SVD representations.
    Applies field weights: Name (3.0), Fandom (2.0), Ship (1.5), Abstract (1.0), Review (1.0).
    Returns a list of Entry objects for the best matches.
    """
    cleaned_query = clean_text(user_query)

    # Compute similarities for each field using the precomputed objects:
    sim_names     = compute_precomputed_similarity(precomputed['names'], cleaned_query)
    sim_fandoms   = compute_precomputed_similarity(precomputed['fandoms'], cleaned_query)
    sim_ships     = compute_precomputed_similarity(precomputed['ships'], cleaned_query)
    sim_abstracts = compute_precomputed_similarity(precomputed['abstracts'], cleaned_query)
    sim_reviews   = compute_precomputed_similarity(precomputed['reviews'], cleaned_query)

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

    # Optional: filter by threshold relative to average nonzero similarity
    nonzero = [score for score in total_sim_dict.values() if score != 0]
    avg = sum(nonzero)/len(nonzero) if nonzero else 0

    ourentries = []
    for idx in sorted_keys:
        if total_sim_dict[idx] > avg * 2:
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


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
