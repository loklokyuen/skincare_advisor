import pandas as pd

df = pd.read_csv("products_clean.csv")

out = pd.DataFrame({
    "obf_code": df["obf_code"],
    "product_name": df["product_name"],
    "brand": df["brand"],
    "quantity": df["quantity"],
    "image_url": df["image_url"],
    "ingredients_raw": df["ingredients_raw"],
    "ingredients_parsed": df["ingredients_parsed_json"],
    "categories": df["categories_json"],
    "country": df["country"],
    "source": df["source"],
})

out.to_csv("products_import_no_header.csv", index=False, header=False)