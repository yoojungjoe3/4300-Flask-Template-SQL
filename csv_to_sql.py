import pandas as pd
import os

csv_file = "DataSet.csv"  
sql_file = "init.sql"
table_name = "quillquestdb"  

# Load CSV
df = pd.read_csv(csv_file)

absolute_csv_path = os.path.abspath(csv_file).replace("\\", "/")

load_command = f"""
LOAD DATA INFILE '{absolute_csv_path}'
INTO TABLE {table_name}
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\\n'
IGNORE 1 LINES
(Name, Fandom, Ships, Rating, Link, Review, Abstract);
"""

with open(sql_file, "w") as f:
    f.write(load_command)

print(f"MySQL LOAD DATA command written to {sql_file}")

#Run: mysql -u root -p quillquestdb < /workspaces/4300-Flask-Template-SQL/init.sql

