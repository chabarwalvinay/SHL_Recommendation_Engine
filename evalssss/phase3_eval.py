import sys
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from retrieval.search import search
from reranking.reranker import rerank


# =========================================================
# CONFIG
# =========================================================
TOP_K_RETRIEVAL = 50
FINAL_K = 10

ASSESSMENTS_FILE = PROJECT_ROOT / "data" / "processed" / "shl_assessments.json"
OUTPUT_FILE = Path(__file__).resolve().parent / "data" / "ranking_eval.csv"


# =========================================================
# LOAD CANONICAL DATA
# =========================================================
with open(ASSESSMENTS_FILE, "r", encoding="utf-8") as f:
    ASSESSMENTS = {a["assessment_id"]: a for a in json.load(f)}


# =========================================================
# EVALUATION
# =========================================================
def evaluate(query: str, true_ids: set):
    retrieved = search(query)[:TOP_K_RETRIEVAL]

    # Attach metadata
    candidates = []
    for r in retrieved:
        aid = r.get("assessment_id")
        if aid in ASSESSMENTS:
            c = ASSESSMENTS[aid].copy()
            c["assessment_id"] = aid
            c["retrieval_score"] = r.get("retrieval_score", 0)
            candidates.append(c)

    reranked = rerank(query, candidates, FINAL_K)

    rows = []
    for rank, c in enumerate(reranked, start=1):
        rows.append(
            {
                "query": query,
                "assessment_id": c["assessment_id"],
                "rank": rank,
                "is_relevant": int(c["assessment_id"] in true_ids),
                "test_type": c.get("test_type"),
            }
        )

    return rows


# =========================================================
# MAIN
# =========================================================
def main():
    from evalssss.phase2_eval import load_url_to_id_map, load_training_data

    url_to_id = load_url_to_id_map()
    queries, _ = load_training_data(url_to_id)

    all_rows = []

    for query, true_ids in queries:
        all_rows.extend(evaluate(query, true_ids))

    df = pd.DataFrame(all_rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Saved Phase-3 ranking eval to: {OUTPUT_FILE}")

    recall_10 = df.groupby("query")["is_relevant"].max().mean()
    print(f"🔹 Phase-3 Recall@10: {recall_10:.3f}")


if __name__ == "__main__":
    main()
