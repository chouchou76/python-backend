from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import datetime

# === Đường dẫn ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "..", "index", "faiss.index")
METADATA_PATH = os.path.join(BASE_DIR, "..", "index", "faiss_metadata.json")
SEARCH_LOG_PATH = os.path.join(BASE_DIR, "..", "index", "search_logs.json")

# === Tải FAISS index và metadata ===
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# === Mô hình semantic ===
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === Flask App ===
app = Flask(__name__)
CORS(app)

# === Hàm ghi log tìm kiếm ===
def log_search_query(query, ip_address, user_agent):
    log_entry = {
        "query": query,
        "timestamp": datetime.datetime.now().isoformat(),
        "ip": ip_address,
        "user_agent": user_agent
    }

    try:
        with open(SEARCH_LOG_PATH, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.insert(0, log_entry)
    logs = logs[:1000]  # giữ lại 1000 bản ghi gần nhất

    with open(SEARCH_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

# === Route tìm kiếm ===
@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 25))

    if not query:
        return jsonify(metadata[:top_k])

    # Lấy IP và User-Agent
    ip_address = request.remote_addr
    user_agent = request.headers.get("User-Agent", "Unknown")

    # Ghi log tìm kiếm
    log_search_query(query, ip_address, user_agent)

    # Semantic embedding
    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(query_vector, top_k)

    # Ghép kết quả FAISS
    results = []
    for i in I[0]:
        if i < len(metadata):
            item = metadata[i].copy()
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

# === Route xem lịch sử tìm kiếm ===
@app.route("/search/logs", methods=["GET"])
def get_search_logs():
    try:
        with open(SEARCH_LOG_PATH, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except:
        logs = []

    return jsonify(logs)

# === Chạy app bằng waitress ===
if __name__ == "__main__":
    serve(app, host="127.0.0.1", port=5000)