import json
import os
import re
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

#function creates object in the format that we want printed out
class Entry:
    def __init__(self, name, ships, fandoms, ratings, abstracts, links):
        self.name = name
        self.ships = ships
        self.fandoms = fandoms
        self.ratings = ratings
        self.abstracts = abstracts
        self.links = links
        #self.pic = tbd

    def __repr__(self):
        return f"Entry(Name: {self.names}, Ships: {self.ships}, Fandoms: {self.fandoms}, Ratings: {self.ratings}, Abstracts: {self.abstracts}, Links: {self.links})"

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

    vectorizer = TfidfVectorizer()

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

    for x in sorted_keys: 
        if total_sim_dict[x] != 0: 
            final_names = names[x-1]
            final_ships = ships[x-1]
            final_fandoms = fandoms[x-1]
            final_ratings = ratings[x-1]
            final_abstracts = abstracts[x-1]
            final_links = links[x-1]

            e = Entry(final_names, final_ships, final_fandoms, final_ratings, final_abstracts, final_links)
            ourentries.append(e)


    print(ourentries)        
    return ourentries


# def sql_search(text):
#     """
#     Perform an SQL search using the LIKE operator.
#     This is a sample function. Adjust it as needed to combine with vector search results.
#     """
#     keys = ["Name", "Fandom", "Ships", "Rating", "Link", "Review", "Abstract"]
#     data = mysql_engine.query_selector(query_sql)
#     return json.dumps([dict(zip(keys, record)) for record in data])
#     #return [dict(zip(keys, record)) for record in data]


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


    sim_dict, top_fic, second_fic = vector_search(user_query)
    #results = sql_search(user_query)

    return jsonify({
        "similarities": sim_dict,
        "top_fic": top_fic,
        "second_fic": second_fic
    })


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
