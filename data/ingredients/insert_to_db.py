import json
import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

with open("ingredients.json") as f:
    ingredients = json.load(f)

conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", 5432)),
    dbname=os.getenv("DB_NAME", "postgres"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", ""),
)
cur = conn.cursor()

for row in ingredients:
    cur.execute("""
        INSERT INTO skincare_advisor.ingredients (
            inci_name, common_names, functions,
            suitable_skin_types, avoid_skin_types, conditions,
            known_conflicts, known_conflicts_detail,
            safety_notes, usage_guidance, confidence, sources
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (inci_name) DO NOTHING
    """, (
        row["inci_name"],
        row["common_names"],
        row["functions"],
        row["suitable_skin_types"],
        row["avoid_skin_types"],
        row["conditions"],
        row["known_conflicts"],
        json.dumps(row["known_conflicts_detail"]),  # JSONB
        row["safety_notes"],
        row["usage_guidance"],
        row["confidence"],
        json.dumps(row["sources"]),           
    ))

conn.commit()
cur.close()
conn.close()