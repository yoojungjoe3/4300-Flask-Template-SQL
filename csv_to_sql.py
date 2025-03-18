import pandas as pd

# Define filenames
csv_file = "cleaned_data.csv"  # Use cleaned data
sql_file = "insert_data.sql"
table_name = "quillquestdb"  # Your MySQL table name

# Load CSV
df = pd.read_csv(csv_file)

# Remove leading/trailing spaces and convert column names to lowercase
df.columns = df.columns.str.strip()

# Ensure unique rows before generating SQL
df = df.drop_duplicates()

# Generate INSERT statements
insert_sql = ""
for _, row in df.iterrows():
    values = "', '".join(map(str, row.tolist()))  # Convert row values to strings
    insert_sql += f"INSERT INTO `{table_name}` ({', '.join(df.columns)}) VALUES ('{values}');\n"

# Save to SQL file
with open(sql_file, "w", encoding="utf-8") as f:
    f.write(insert_sql)

print(f"SQL INSERT file '{sql_file}' created successfully with no duplicates.")
