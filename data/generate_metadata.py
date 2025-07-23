import pandas as pd
import json
import os
import base64

DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(DIR, "products_full_data.csv")
json_path = os.path.join(DIR, "product_metadata.json")

df = pd.read_csv(csv_path)
df = df.dropna(subset=["description", "name"])  # Bỏ dòng trống

# Chuyển cột image_urls thành list nếu đang là string
def parse_images(val):
    try:
        return eval(val) if isinstance(val, str) else val
    except:
        return []

df["image_urls"] = df["image_urls"].apply(parse_images)

# Chỉ giữ các trường cần thiết
metadata = []
for _, row in df.iterrows():
    metadata.append({
        "name": row["name"],
        "description": row["description"],
        "price_vnd": row.get("price_vnd", ""),
        "url": row.get("url", ""),
        "image_urls": row.get("image_urls", [])
    })

# Lưu thành JSON
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"✅ Đã tạo {json_path} chứa {len(metadata)} sản phẩm")