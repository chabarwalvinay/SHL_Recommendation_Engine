import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from retrieval.process import preprocess_query


# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INDEX_DIR = DATA_DIR / "index"
PROCESSED_DIR = DATA_DIR / "processed"

EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
ID_MAP_FILE = INDEX_DIR / "id_map.json"
META_FILE = INDEX_DIR / "meta.json"
ASSESSMENTS_FILE = PROCESSED_DIR / "shl_assessments.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 50
TOP_K_VECTOR = 100
TOP_K_BM25 = 100

VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3


# =========================================================
# LAZY GLOBALS (CRITICAL FOR MEMORY)
# =========================================================
_embeddings = None
_faiss_index = None
_bm25 = None
_model = None
_id_map = None
_assessment_lookup = None


# =========================================================
# LOAD STATIC FILES (LIGHTWEIGHT)
# =========================================================
def load_metadata():
    global _id_map, _assessment_lookup

    if _id_map is None:
        with open(ID_MAP_FILE, "r", encoding="utf-8") as f:
            _id_map = json.load(f)

    if _assessment_lookup is None:
        with open(ASSESSMENTS_FILE, "r", encoding="utf-8") as f:
            assessments = json.load(f)
        _assessment_lookup = {a["assessment_id"]: a for a in assessments}

    return _id_map, _assessment_lookup


# =========================================================
# EMBEDDINGS
# =========================================================
def get_embeddings():
    global _embeddings

    if _embeddings is None:
        print("🔹 Loading embeddings.npy")
        _embeddings = np.load(EMBEDDINGS_FILE)

    return _embeddings


# =========================================================
# FAISS INDEX (COSINE / IP)
# =========================================================
def get_faiss_index():
    global _faiss_index

    if _faiss_index is None:
        embeddings = get_embeddings()
        dim = embeddings.shape[1]

        print("🔹 Building FAISS index")
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        _faiss_index = index

    return _faiss_index


# =========================================================
# BM25 INDEX
# =========================================================
def bm25_text(a: Dict) -> str:
    return " ".join(
        [
            a.get("name", ""),
            a.get("description", ""),
            " ".join(a.get("test_type", [])),
            a.get("job_levels", "") or "",
            a.get("languages", "") or "",
        ]
    ).lower()


def get_bm25():
    global _bm25

    if _bm25 is None:
        print("🔹 Building BM25 index")

        id_map, assessment_lookup = load_metadata()

        corpus = [
            bm25_text(assessment_lookup[id_map[str(i)]]).split()
            for i in range(len(id_map))
        ]

        _bm25 = BM25Okapi(corpus)

    return _bm25


# =========================================================
# MODEL
# =========================================================
def get_model():
    global _model

    if _model is None:
        print("🔹 Loading SentenceTransformer model")
        _model = SentenceTransformer(MODEL_NAME)

    return _model


# =========================================================
# SEARCH (PHASE-2 PURE)
# =========================================================
def search(query: str) -> List[Dict]:
    clean_query = preprocess_query(query)
    if not clean_query:
        return []

    id_map, _ = load_metadata()
    model = get_model()
    embeddings = get_embeddings()
    faiss_index = get_faiss_index()
    bm25 = get_bm25()

    # ---- Vector Search ----
    q_vec = model.encode([clean_query], normalize_embeddings=True).astype("float32")

    vec_scores, vec_ids = faiss_index.search(q_vec, TOP_K_VECTOR)

    vector_results = {
        int(idx): float(score) for idx, score in zip(vec_ids[0], vec_scores[0])
    }

    # ---- BM25 Search ----
    tokens = clean_query.lower().split()
    bm25_raw = bm25.get_scores(tokens)

    bm25_max = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
    bm25_scores = bm25_raw / bm25_max

    top_bm25_ids = np.argsort(bm25_scores)[::-1][:TOP_K_BM25]
    bm25_results = {int(idx): float(bm25_scores[idx]) for idx in top_bm25_ids}

    # ---- Hybrid Merge ----
    all_ids = set(vector_results) | set(bm25_results)
    merged = []

    for idx in all_ids:
        score = VECTOR_WEIGHT * vector_results.get(
            idx, 0.0
        ) + BM25_WEIGHT * bm25_results.get(idx, 0.0)
        merged.append((idx, score))

    merged.sort(key=lambda x: x[1], reverse=True)

    # ---- Phase-2 Output ----
    results = []
    for idx, score in merged[:TOP_K]:
        results.append(
            {
                "assessment_id": id_map[str(idx)],
                "score": round(score, 4),
            }
        )

    return results


# =========================================================
# MANUAL SMOKE TEST
# =========================================================
if __name__ == "__main__":
    q = "I need a test for .NET 4.5 developer"
    out = search(q)
    for r in out[:10]:
        print(r)
