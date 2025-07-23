from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import os

# === Đường dẫn ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "..", "index", "faiss.index")
METADATA_PATH = os.path.join(BASE_DIR, "..", "index", "faiss_metadata.json")

# === Load FAISS và metadata ===
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
        return jsonify({"error": "Query missing"}), 400

    # Encode truy vấn
    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")

    # Tìm kiếm FAISS
    D, I = index.search(query_vector, top_k)

    # Trả kết quả metadata tương ứng
    results = []
    for i in I[0]:
        if i < len(metadata):
            results.append(metadata[i])

    return jsonify(results)

if __name__ == "__main__":
    serve(app, host="127.0.0.1", port=5000)



