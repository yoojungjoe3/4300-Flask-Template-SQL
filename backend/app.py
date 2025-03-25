import json
import os
import re
import numpy
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sqlalchemy as db 
engine = db.create_engine("mysql+pymysql://admin:admin@4300showcase.infosci.cornell.edu/kardashiandb")
cursor = engine.connect()

# Set ROOT_PATH for linking with all your files.
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Database credentials (adjust if needed)
LOCAL_MYSQL_USER = "root"
LOCAL_MYSQL_USER_PASSWORD = "admin"
LOCAL_MYSQL_PORT = 3306
LOCAL_MYSQL_DATABASE = "kardashiandb"

# Initialize database handler and load init.sql into the database
mysql_engine = MySQLDatabaseHandler(LOCAL_MYSQL_USER, LOCAL_MYSQL_USER_PASSWORD, LOCAL_MYSQL_PORT, LOCAL_MYSQL_DATABASE)
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

# # Lists to hold extracted data from init.sql
# fandoms = []
# ships = []
# names = []

# Regex to capture Name, Fandom, and Ship(s)
# pattern = re.compile(r"VALUES\s*\(\s*'\"(.*?)\"',\s*'\"(.*?)\"',\s*'\"(.*?)\"',")

# current_dir = os.path.dirname(os.path.abspath(__file__))
# init_sql_path = os.path.join(current_dir, "..", "init.sql")

# Read the init.sql file and populate the lists

# with open(os.path.join(os.environ['ROOT_PATH'],'init.sql'), "r", encoding="utf-8") as file:
#     for line in file:
#         find = pattern.search(line)
#         if find:
#             names.append(find.group(1))
#             fandom_clean = clean_text(find.group(2))
#             ship_clean = clean_text(find.group(3))
#             # Append even if duplicate, if your ordering needs to match the SQL records.
#             fandoms.append(fandom_clean)
#             ships.append(ship_clean)

def vector_search(user_query):
    """
    Compute combined similarity scores for the query against both fandoms and ships.
    Also prints the top two fanfic titles.
    Returns a dictionary mapping record number (starting at 1) to the combined similarity score.
    """
    # Clean and prepare the query
    words = clean_text(user_query).split()
    query_words = " ".join(words)
    print(query_words)

    #creating the lists
    # use cursor 


    query = "SELECT Fandom, Ship(s) FROM fics;"
    cursor.execute(query)
    row = cursor.fetchall()
    fandoms = [r[0] for r in row]
    ships = [r[1] for r in row]
    # fandoms = f"""SELECT Fandom FROM fics"""
    # ships = f"""SELECT Ship(s) FROM fics"""

    vectorizer = TfidfVectorizer()

    # Create a list to store the similarity scores for the query words
    similarity_scores_fandoms = []
    similarity_scores_ships = []

    # Iterate through each word in the query and compare with fandoms and ships
    for word in query_words:
        join_fandom_query = [fandoms] + [word]
        tfidf_matrix_fandoms = vectorizer.fit_transform(join_fandom_query)
        query_vector_fandoms = tfidf_matrix_fandoms[-1]
        candidate_vectors_fandoms = tfidf_matrix_fandoms[:-1]
        similarities_fandoms = cosine_similarity(query_vector_fandoms, candidate_vectors_fandoms).flatten()
        similarity_scores_fandoms.append(similarities_fandoms)

        join_ship_query = ships + [word]
        tfidf_matrix_ships = vectorizer.fit_transform(join_ship_query)
        query_vector_ships = tfidf_matrix_ships[-1]
        candidate_vectors_ships = tfidf_matrix_ships[:-1]
        similarities_ships = cosine_similarity(query_vector_ships, candidate_vectors_ships).flatten()
        similarity_scores_ships.append(similarities_ships)

    # Combine the similarity scores for each query word (sum of all word similarities)
    combined_fandom_similarities = np.sum(np.array(similarity_scores_fandoms), axis=0)
    combined_ship_similarities = np.sum(np.array(similarity_scores_ships), axis=0)

    # Combine fandom and ship similarities
    combined_similarities = combined_fandom_similarities + combined_ship_similarities
    total_sim_dict = {i + 1: total for i, total in enumerate(combined_similarities)}
    print(total_sim_dict)

    # Sort keys (record indices) by similarity score (highest first)
    sorted_keys = sorted(total_sim_dict, key=total_sim_dict.get, reverse=True)

    # Get the keys for the highest and second-highest scores
    highest_key = sorted_keys[0]
    second_highest_key = sorted_keys[1]

    # Adjust index for names list (keys start at 1, list is zero-indexed)
    top_fic = names[sorted_keys[0] - 1] if sorted_keys else None
    second_fic = names[sorted_keys[1] - 1] if len(sorted_keys) > 1 else None

    #print(total_sim_dict)

    # Print similarity values and top results
    print(f"Top result: {top_fic} with similarity score: {total_sim_dict[highest_key]}")
    print(f"Second result: {second_fic} with similarity score: {total_sim_dict[second_highest_key]}")


    return total_sim_dict, top_fic, second_fic
   
    # vectorizer = TfidfVectorizer()

    # # Compare query to fandoms
    # join_fandom_query = fandoms + [query_text]
    # tfidf_matrix_fandoms = vectorizer.fit_transform(join_fandom_query)
    # query_vector_fandoms = tfidf_matrix_fandoms[-1]
    # candidate_vectors_fandoms = tfidf_matrix_fandoms[:-1]
    # similarities_fandoms = cosine_similarity(query_vector_fandoms, candidate_vectors_fandoms).flatten()

    # # Compare query to ships
    # join_ship_query = ships + [query_text]
    # tfidf_matrix_ships = vectorizer.fit_transform(join_ship_query)
    # query_vector_ships = tfidf_matrix_ships[-1]
    # candidate_vectors_ships = tfidf_matrix_ships[:-1]
    # similarities_ships = cosine_similarity(query_vector_ships, candidate_vectors_ships).flatten()
   
    # # Combine the similarity scores element-wise
    # combined_similarities = numpy.array(similarities_fandoms) + numpy.array(similarities_ships)
    # total_sim_dict = {i + 1: total for i, total in enumerate(combined_similarities)}
   
    # # Sort keys (record indices) by similarity score (highest first)
    # sorted_keys = sorted(total_sim_dict, key=total_sim_dict.get, reverse=True)
   
    # # Get the keys for the highest and second-highest scores
    # highest_key = sorted_keys[0]
    # second_highest_key = sorted_keys[1]
   
    # # Adjust index for names list (keys start at 1, list is zero-indexed)
    # top_fic = names[sorted_keys[0] - 1] if sorted_keys else None
    # second_fic = names[sorted_keys[1] - 1] if len(sorted_keys) > 1 else None
   
    # return total_sim_dict, top_fic, second_fic

def sql_search(text):
    """
    Perform an SQL search using the LIKE operator.
    This is a sample function. Adjust it as needed to combine with vector search results.
    """
    query_sql = f"""SELECT * FROM fics WHERE LOWER( Name ) LIKE '%%{text.lower()}%%' limit 10"""
    keys = ["Name", "Fandom", "Ship(s)", "Rating", "Link", "Review", "Abstract"]
    data = mysql_engine.query_selector(query_sql)
    return json.dumps([dict(zip(keys, record)) for record in data])
    #return [dict(zip(keys, record)) for record in data]

@app.route("/")
def home():
    # x = render_template('base.html', Name="sample html")
    # fics_search()
    return render_template('base.html', Name="sample html")

def clean_text(user_query):
    """Convert text to lowercase and remove punctuation."""
    return re.sub(r'[^\w\s]', '', user_query.lower())

@app.route("/fics")
def fics_search():
    #print("help")
    user_query = request.args.get("Name")
    #print("pls: " + str(user_query))
    # user_query = "harry"
    #if not user_query:
    #    return ("Please input a query :)"), 400


    #sim_dict, top_fic, second_fic = vector_search(user_query)
    #results = sql_search(user_query)
   
    #for i, record in enumerate(results):
    #    record["similarity"] = sim_dict.get(i + 1, 0)
   
    #results_sorted = sorted(results, key=lambda x: x["similarity"], reverse=True)
   
    #response = {
    #    "results": results_sorted,
    #    "top_fic": top_fic,
    #    "second_fic": second_fic,
    #}

    #return render_template("base.html", results= results_sorted ,top_fic= top_fic , second_fic=second_fic)
    #return json.dumps(response), 200, {"Content-Type": "application/json"}
    return vector_search(user_query)

if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
