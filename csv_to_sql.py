import os

csv_file = "DataSet.csv"
sql_file = "init.sql"
table_name = "quillquestdb"

# Get absolute path
absolute_csv_path = os.path.abspath(csv_file)

with open(sql_file, "w") as f:
    f.write(f"""
LOAD DATA INFILE '{absolute_csv_path}'
INTO TABLE {table_name}
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\\n'
IGNORE 1 ROWS
(Name, Fandom, Ships, Rating, Link, Review, Abstract);
""")
print(f"SQL file written to {sql_file}")

#Run: sudo mysql quillquestdb < /workspaces/4300-Flask-Template-SQL/init.sql


