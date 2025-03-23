import pandas as pd

# Define filenames
csv_file = "DataSet.csv"  # Use cleaned data
sql_file = "init.sql"
table_name = "quillquestdb"  # Your MySQL table name

# Load CSV
df = pd.read_csv(csv_file)

COPY quillquestdb(Name, Fandom, Ships, Rating, Link, Review, Abstract)
FROM '/workspaces/4300-Flask-Template-SQL/backend/csv_to_sql.py'
WITH (FORMAT csv, HEADER true);
