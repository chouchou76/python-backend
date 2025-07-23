from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

# === Đường dẫn ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "..", "index", "faiss.index")
METADATA_PATH = os.path.join(BASE_DIR, "..", "index", "faiss_metadata.json")

# === Tải FAISS index ===
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

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
        return jsonify(metadata[:top_k])  # Trả toàn bộ nếu query rỗng

    # Semantic embedding
    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(query_vector, top_k)

    # Ghép kết quả FAISS
    results = []
    for i in I[0]:
        if i < len(metadata):
            item = metadata[i]
            cosine_score = 1 - D[0][list(I[0]).index(i)]  # khoảng cách nhỏ = điểm cao
            item["cosine_score"] = float(cosine_score)
            results.append(item)

    # Fuzzy matching theo tên sản phẩm
    for item in results:
        item["fuzzy_score"] = fuzz.partial_ratio(query.lower(), item["name"].lower()) / 100.0

    # Tổng điểm ưu tiên (weighted)
    for item in results:
        item["score"] = 0.7 * item["cosine_score"] + 0.3 * item["fuzzy_score"]

    # Sắp xếp theo score
    results.sort(key=lambda x: x["score"], reverse=True)

    return jsonify(results[:top_k])

if __name__ == "__main__":
    serve(app, host="127.0.0.1", port=5000)