import pandas as pd

# Define CSV and SQL file names
csv_file = "DataSet.csv"  
sql_file = "insert_data.sql"
table_name = "quillquestdb"  # Your MySQL table name

# Load CSV into Pandas DataFrame
df = pd.read_csv(csv_file)

# Ensure DataFrame columns match the SQL table structure
expected_columns = ["Name", "Fandom", "Ships", "Rating", "Link", "Review", "Abstract"]
df = df[expected_columns]  # Select only relevant columns

# Generate INSERT statements
insert_sql = ""
for _, row in df.iterrows():
    values = "', '".join(map(str, row.tolist()))  # Convert values to strings
    insert_sql += f"INSERT INTO {table_name} (Name, Fandom, Ships, Rating, Link, Review, Abstract) VALUES ('{values}');\n"

# Save to SQL file
with open(sql_file, "w", encoding="utf-8") as f:
    f.write(insert_sql)

print(f"SQL INSERT file '{sql_file}' created successfully.")
