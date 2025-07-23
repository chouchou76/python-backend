# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from waitress import serve
# import firebase_admin
# from firebase_admin import credentials, firestore
# from sentence_transformers import SentenceTransformer
# from rapidfuzz import fuzz
# import numpy as np

# # Khởi tạo Firebase
# cred = credentials.Certificate("C:/Users/HP/python-backend/credentials/serviceAccountKey.json")
# firebase_admin.initialize_app(cred)
# db = firestore.client()

# # Mô hình semantic
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # Flask app
# app = Flask(__name__)
# CORS(app)

# @app.route("/search", methods=["POST"])
# def search():
#     data = request.get_json()
#     query = data.get("query", "").strip()
#     top_k = data.get("top_k", 500)

#     if not query:
#         return jsonify({"error": "Missing query"}), 400

#     query_vector = model.encode([query], convert_to_numpy=True).flatten()

#     # Lấy toàn bộ sản phẩm
#     docs = db.collection("products").stream()
#     products = [doc.to_dict() for doc in docs]

#     scored = []
#     for product in products:
#         if "embedding" not in product or not product["embedding"]:
#             continue

#         try:
#             emb = np.array(product["embedding"])
#             cosine_score = float(np.dot(query_vector, emb) / (np.linalg.norm(query_vector) * np.linalg.norm(emb)))

#             # Fuzzy matching theo tên sản phẩm
#             name_score = fuzz.partial_ratio(query.lower(), product["name"].lower()) / 100.0  # scale về 0–1

#             # Tính tổng điểm ưu tiên
#             total_score = 0.7 * cosine_score + 0.3 * name_score

#             scored.append((total_score, product, name_score, cosine_score))
#         except Exception as e:
#             print(f"Lỗi tính điểm cho sản phẩm '{product.get('name', 'unknown')}':", e)

#     # Ưu tiên sản phẩm tên gần đúng (> 0.95 fuzzy)
#     scored.sort(key=lambda x: (x[2] > 0.95, x[0]), reverse=True)

#     # Trả kết quả
#     top_products = [s[1] for s in scored[:top_k]]
#     return jsonify(top_products)

# if __name__ == "__main__":
#     serve(app, host="127.0.0.1", port=5000)

from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
import faiss
import base64
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import firebase_admin
from firebase_admin import credentials, firestore
import os

# === Đường dẫn ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# FAISS_INDEX_PATH = os.path.join(BASE_DIR, "..", "data", "product_index.faiss")
FAISS_INDEX_PATH = "C:\\Users\\HP\\python-backend\\index\\faiss.index"
# METADATA_PATH = os.path.join(BASE_DIR, "..", "data", "product_metadata.json")
METADATA_PATH = os.path.join(BASE_DIR, "..", "index", "faiss_metadata.json")
FIREBASE_CRED_PATH = os.path.join(BASE_DIR, "..", "credentials", "serviceAccountKey.json")

# === Firebase ===
cred = credentials.Certificate(FIREBASE_CRED_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# === FAISS ===
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

product_ids = [item["product_id"] for item in metadata if "product_id" in item]

# === Mô hình semantic ===
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === Flask App ===
app = Flask(__name__)
CORS(app)

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 25))

    if not query:
        return jsonify({"error": "Query missing"}), 400

    # Encode truy vấn thành vector
    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")

    # FAISS trả về index kết quả
    D, I = index.search(query_vector, top_k)

    matched_ids = []
    for i in I[0]:
        if i < len(product_ids):
            matched_ids.append(product_ids[i])

    # Lấy batch metadata từ Firestore
    doc_refs = [db.collection("products").document(pid) for pid in matched_ids]
    docs = db.get_all(doc_refs)
    results = [doc.to_dict() for doc in docs if doc.exists]

    return jsonify(results)

if __name__ == "__main__":
    serve(app, host="127.0.0.1", port=5000)


