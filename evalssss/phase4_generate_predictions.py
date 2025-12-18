import sys
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# =========================================================
# ADD PROJECT ROOT
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from retrieval.search import search
from reranking.reranker import rerank


# =========================================================
# CONFIG
# =========================================================
BASE_DIR = PROJECT_ROOT

# ⚠️ CHANGE SHEET NAME IF NEEDED
EXCEL_FILE = BASE_DIR / "data" / "train_test_data" / "Gen_AI Dataset (1).xlsx"
TEST_SHEET = "Test-Set"  # Unlabelled test queries

ASSESSMENTS_FILE = BASE_DIR / "data" / "processed" / "shl_assessments.json"

OUTPUT_CSV = BASE_DIR / "data" / "submission_predictions.csv"

FINAL_K = 10


# =========================================================
# LOAD CANONICAL ASSESSMENTS
# =========================================================
def load_assessments():
    with open(ASSESSMENTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {a["assessment_id"]: a for a in data}


# =========================================================
# LOAD TEST QUERIES
# =========================================================
def load_test_queries():
    df = pd.read_excel(EXCEL_FILE, sheet_name=TEST_SHEET)
    df.columns = [c.strip().lower() for c in df.columns]

    assert "query" in df.columns

    queries = df["query"].dropna().unique().tolist()
    print(f"🔹 Loaded {len(queries)} test queries")

    return queries


# =========================================================
# GENERATE PREDICTIONS
# =========================================================
def generate_predictions(queries, assessments):
    rows = []

    for query in tqdm(queries, desc="🔹 Generating predictions"):
        # Phase-2 retrieval
        retrieved = search(query)[:50]

        candidates = []
        for r in retrieved:
            aid = r.get("assessment_id")
            if aid in assessments:
                c = assessments[aid].copy()
                c["retrieval_score"] = r.get("retrieval_score", 0.0)
                candidates.append(c)


        if not candidates:
            continue

        # Phase-3 reranking
        reranked = rerank(query, candidates, final_k=FINAL_K)

        for a in reranked:
            rows.append({"Query": query, "Assessment_url": a["url"]})

    return pd.DataFrame(rows)


# =========================================================
# MAIN
# =========================================================
def main():
    assessments = load_assessments()
    queries = load_test_queries()

    df = generate_predictions(queries, assessments)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Saved submission CSV to: {OUTPUT_CSV}")
    print(f"📊 Total rows: {len(df)}")


if __name__ == "__main__":
    main()
