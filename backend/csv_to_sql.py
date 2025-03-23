import os
table_name = "quillquestdb"
df = pd.read_csv(csv_file)

df.columns = df.columns.str.strip()

df = df.drop_duplicates()

insert_sql = ""
 for _, row in df.iterrows():
     values = "', '".join(map(str, row.tolist()))  # Convert row values to strings
     insert_sql += f"INSERT INTO `{table_name}` ({', '.join(df.columns)}) VALUES ('{values}');\n"

with open(sql_file, "w", encoding="utf-8") as f:
     f.write(insert_sql)

print(f"SQL INSERT file '{sql_file}' created successfully with no duplicates.")

#Optimized Code:
#csv_filename = "DataSet_clean.csv"
#sql_filename = "init.sql"
#table_name = "fanfics"

#secure_mysql_dir = "/var/lib/mysql-files"
#secure_csv_path = os.path.join(secure_mysql_dir, csv_filename)

# Write init.sql with the correct path (no LOCAL)
#with open(sql_filename, "w") as f:
#    f.write(f"""
#LOAD DATA INFILE '{secure_csv_path}'
#INTO TABLE {table_name}
#FIELDS TERMINATED BY ',' 
#ENCLOSED BY '"'
#LINES TERMINATED BY '\\n'
#IGNORE 1 ROWS
#(Name, Fandom, Ships, Rating, Link, Review, Abstract);
#""")

# Move the file
#os.system(f"sudo cp {csv_filename} {secure_csv_path}")
#print(f"CSV copied to {secure_csv_path}")
#print(f"SQL file written to {sql_filename}")

#SECURE_DIR = /var/lib/mysql-files/

#To go into sql: sudo mysql -D quillquestdb
