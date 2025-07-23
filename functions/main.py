import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import base64

# Đường dẫn input & output
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "product_metadata.json")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "..", "index", "faiss.index")
FAISS_METADATA_PATH = os.path.join(BASE_DIR, "..", "index", "faiss_metadata.json")

# Load dữ liệu metadata
with open(DATA_PATH, "r", encoding="utf-8") as f:
    products = json.load(f)

products = [p for p in products if p.get("description")]

# Khởi tạo mô hình embedding
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
descriptions = [p["description"] for p in products]
embeddings = model.encode(descriptions, convert_to_numpy=True, show_progress_bar=True)

# Tạo FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# Lưu FAISS index
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
faiss.write_index(index, FAISS_INDEX_PATH)

# Lưu metadata kèm product_id base64
metadata = []
for i, product in enumerate(products):
    url = product.get("url", "")
    product_id = base64.urlsafe_b64encode(url.encode()).decode() if url else f"product_{i}"
    metadata.append({
        "product_id": product_id,
        "name": product.get("name"),
        "description": product.get("description"),
        "url": url,
        "image_urls": product.get("image_urls", []),
        "price_vnd": product.get("price_vnd")
    })

with open(FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"✅ FAISS index và metadata đã được lưu vào thư mục 'index/'")