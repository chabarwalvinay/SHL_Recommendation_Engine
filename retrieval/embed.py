import json
import hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_JSON = BASE_DIR / "data" / "processed" / "shl_assessments.json"
INDEX_DIR = BASE_DIR / "data" / "index"

EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
ID_MAP_FILE = INDEX_DIR / "id_map.json"
META_FILE = INDEX_DIR / "meta.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


# ============================================================
# UTILS
# ============================================================
def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(value: str | None) -> str:
    if not value:
        return "Not specified"
    return value.strip()


def build_embedding_text(assessment: Dict) -> str:
    """
    Build a deterministic, one-way embedding text.
    This text is NEVER parsed back.
    """

    parts = [
        f"Assessment Name: {normalize_text(assessment.get('name'))}.",
        f"Description: {normalize_text(assessment.get('description'))}.",
        f"Test Type: {', '.join(assessment.get('test_type', [])) or 'Not specified'}.",
        f"Duration: {assessment.get('duration', 'Not specified')} minutes.",
        f"Job Levels: {normalize_text(assessment.get('job_levels'))}.",
        f"Languages: {normalize_text(assessment.get('languages'))}.",
        f"Remote Testing: {normalize_text(assessment.get('remote_support'))}.",
    ]

    return " ".join(parts)


# ============================================================
# MAIN
# ============================================================
def main():
    print("🔹 Phase-2 Embedding Pipeline (Canonical-Safe)")
    print("🔹 Loading canonical data...")

    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_JSON}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        assessments: List[Dict] = json.load(f)

    if not assessments:
        raise ValueError("Input JSON is empty")

    # --------------------------------------------------------
    # Deterministic ordering (CRITICAL)
    # --------------------------------------------------------
    assessments = sorted(assessments, key=lambda x: x["assessment_id"])

    embedding_texts: List[str] = []
    id_map: Dict[int, str] = {}

    for idx, assessment in enumerate(assessments):
        text = build_embedding_text(assessment)
        embedding_texts.append(text)
        id_map[idx] = assessment["assessment_id"]

    print(f"🔹 Assessments to embed: {len(embedding_texts)}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    print(f"🔹 Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------
    print("🔹 Generating embeddings...")
    embeddings = model.encode(
        embedding_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------
    print("🔹 Saving index artifacts...")

    np.save(EMBEDDINGS_FILE, embeddings)

    with open(ID_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2)

    meta = {
        "model": MODEL_NAME,
        "vector_dim": embeddings.shape[1],
        "num_vectors": embeddings.shape[0],
        "input_file": str(INPUT_JSON),
        "input_hash": compute_file_hash(INPUT_JSON),
        "schema_version": "v2-canonical",
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # --------------------------------------------------------
    # Safety checks (ANTI-SILENT-FAILURE)
    # --------------------------------------------------------
    print("🔹 Running sanity checks...")

    assert embeddings.shape[0] == len(id_map), "Mismatch: vectors vs id_map"
    assert embeddings.shape[1] == 384, "Unexpected embedding dimension"

    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms.mean(), 1.0, atol=1e-2), "Embeddings not normalized"

    print("✅ Embedding pipeline complete")
    print(f"📦 vectors: {embeddings.shape}")
    print(f"📁 saved in: {INDEX_DIR}")


if __name__ == "__main__":
    main()
