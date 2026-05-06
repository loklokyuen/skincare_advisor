import json
import psycopg2
import psycopg2.extras

with open("ingredients.json") as f:
    ingredients = json.load(f)

conn = psycopg2.connect("postgresql://postgres:Zp8)baeUzBEB^c`M@34.147.213.190/skincare_advisor")
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