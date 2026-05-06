import json
import csv

with open("ingredients.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("ingredients_import.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["data"])

    for row in data:
        writer.writerow([json.dumps(row)])