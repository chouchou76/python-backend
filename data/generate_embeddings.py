import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

# Khởi tạo Firebase với file service account
cred = credentials.Certificate("C:/Users/HP/python-backend/credentials/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Đọc dữ liệu từ CSV
df = pd.read_csv("products_full_data.csv")
required_columns = {"name", "price_vnd", "description", "image_urls", "url"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Thiếu cột trong file CSV. Yêu cầu các cột: {required_columns}")

# Bỏ các dòng thiếu mô tả
df = df.dropna(subset=["description"]).reset_index(drop=True)

# Encode mô tả
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
descriptions = df["description"].tolist()
embeddings = model.encode(descriptions, convert_to_numpy=True, show_progress_bar=True)

# Ghi vào Firestore
products_ref = db.collection("products")
count = 0

for i, row in df.iterrows():
    try:
        product_data = {
            "name": row["name"],
            "price_vnd": row["price_vnd"],
            "image_urls": eval(row["image_urls"]) if isinstance(row["image_urls"], str) else row["image_urls"],
            "url": row["url"],
            "description": row["description"],
            "embedding": embeddings[i].tolist()
        }
        doc_id = row["url"] or row["name"]
        products_ref.document(doc_id).set(product_data)
        count += 1
    except Exception as e:
        print(f"Lỗi tại dòng {i} ({row.get('name', 'unknown')}): {e}")

print(f"✅ Đã lưu {count} sản phẩm có embedding vào Firestore!")
