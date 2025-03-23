import os

csv_filename = "DataSet.csv"
sql_filename = "init.sql"
table_name = "fanfics"

secure_mysql_dir = "/var/lib/mysql-files"
secure_csv_path = os.path.join(secure_mysql_dir, csv_filename)

# Write init.sql with the correct path (no LOCAL)
with open(sql_filename, "w") as f:
    f.write(f"""
LOAD DATA INFILE '{secure_csv_path}'
INTO TABLE {table_name}
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\\n'
IGNORE 1 ROWS
(Name, Fandom, Ships, Rating, Link, Review, Abstract);
""")

# Move the file
os.system(f"sudo cp {csv_filename} {secure_csv_path}")
print(f"CSV copied to {secure_csv_path}")
print(f"SQL file written to {sql_filename}")

SECURE_DIR = /var/lib/mysql-files/
