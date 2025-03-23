import re
import csv

input_path = "DataSet.csv"
output_path = "DataSet_clean.csv"

# This regex handles quoted fields with commas and inner quotes
pattern = re.compile(r'"((?:[^"]|"")*?)"|([^,]+)')

def extract_fields(line):
    fields = []
    for match in pattern.finditer(line):
        value = match.group(1) or match.group(2)
        if value is not None:
            value = value.replace('""', '"').strip()
            fields.append(value)
    return fields[:7]  # Return only first 7 fields

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", newline='', encoding="utf-8") as outfile:
    
    writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
    for line in infile:
        if line.strip():  # skip empty lines
            fields = extract_fields(line)
            if len(fields) == 7:
                writer.writerow(fields)
