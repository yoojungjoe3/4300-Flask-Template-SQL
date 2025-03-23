import csv
import re

with open("DataSet.csv", "r", encoding="utf-8") as infile, \
     open("DataSet_clean.csv", "w", newline='', encoding="utf-8") as outfile:

    writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)

    for line in infile:
        # Step 1: Remove triple quotes → single quotes
        line = re.sub(r'""+"', '"', line)

        # Step 2: Strip leading/trailing spaces or newlines
        line = line.strip()

        # Step 3: Split by commas
        row = next(csv.reader([line]))

        # Step 4: Trim to 7 columns max
        writer.writerow(row[:7])
