import sys
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


# =========================================================
# ADD PROJECT ROOT
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from retrieval.search import search


# =========================================================
# CONFIG
# =========================================================
BASE_DIR = PROJECT_ROOT

EXCEL_FILE = BASE_DIR / "data" / "train_test_data" / "Gen_AI Dataset (1).xlsx"
TRAIN_SHEET = "Train-Set"

ASSESSMENTS_FILE = BASE_DIR / "data" / "processed" / "shl_assessments.json"

OUTPUT_CSV = BASE_DIR / "data" / "retrieval_eval.csv"

# NEW: missing URLs output directory
MISSING_URLS_DIR = Path(__file__).resolve().parent / "data"
MISSING_URLS_FILE = MISSING_URLS_DIR / "missing_urls.csv"

K_VALUES = [10, 20, 30, 50]


# =========================================================
# URL NORMALIZATION
# =========================================================
def normalize_shl_url(url: str) -> str:
    url = url.strip().rstrip("/")
    url = url.replace("/solutions/products/", "/products/")
    return url


# =========================================================
# LOAD URL → ASSESSMENT_ID MAP
# =========================================================
def load_url_to_id_map():
    with open(ASSESSMENTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    url_to_id = {}
    for item in data:
        url = normalize_shl_url(item["url"])
        aid = item["assessment_id"]
        url_to_id[url] = aid

    print(f"🔹 Loaded {len(url_to_id)} URL → assessment_id mappings")
    return url_to_id


# =========================================================
# LOAD TRAIN DATA
# =========================================================
def load_training_data(url_to_id):
    print("🔹 Loading training dataset...")

    df = pd.read_excel(EXCEL_FILE, sheet_name=TRAIN_SHEET)
    df.columns = [c.strip().lower() for c in df.columns]

    assert "query" in df.columns
    assert "assessment_url" in df.columns

    query_to_ids = defaultdict(set)
    missing_urls = []

    for _, row in df.iterrows():
        query = row["query"].strip()
        raw_url = row["assessment_url"].strip()
        url = normalize_shl_url(raw_url)

        if url not in url_to_id:
            missing_urls.append(
                {"query": query, "assessment_url": raw_url, "normalized_url": url}
            )
            continue

        query_to_ids[query].add(url_to_id[url])

    print(f"🔹 Unique queries: {len(query_to_ids)}")
    print(f"⚠️  URLs not found in canonical data: {len(missing_urls)}")

    return list(query_to_ids.items()), missing_urls


# =========================================================
# SAVE MISSING URLS
# =========================================================
def save_missing_urls(missing_urls):
    if not missing_urls:
        print("✅ No missing URLs to save")
        return

    MISSING_URLS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(missing_urls)
    df.to_csv(MISSING_URLS_FILE, index=False)

    print(f"📄 Saved missing URLs to: {MISSING_URLS_FILE}")


# =========================================================
# EVALUATION
# =========================================================
def run_evaluation(queries):
    print("🔹 Running Phase-2 recall evaluation...")

    results = []

    for idx, (query, true_ids) in enumerate(tqdm(queries)):
        retrieved = search(query)

        retrieved_ids = [r["assessment_id"] for r in retrieved if "assessment_id" in r]

        # Debug first query
        if idx == 0:
            print("\n🔍 DEBUG SAMPLE")
            print("Query:", query)
            print("GT assessment_ids:", list(true_ids))
            print("Retrieved assessment_ids:", retrieved_ids[:10])

        row = {
            "query": query,
            "num_relevant": len(true_ids),
        }

        for k in K_VALUES:
            top_k = retrieved_ids[:k]
            hit = any(aid in top_k for aid in true_ids)
            row[f"recall@{k}"] = int(hit)

        results.append(row)

    return pd.DataFrame(results)


# =========================================================
# MAIN
# =========================================================
def main():
    url_to_id = load_url_to_id_map()

    queries, missing_urls = load_training_data(url_to_id)
    save_missing_urls(missing_urls)

    eval_df = run_evaluation(queries)

    print("\n🔹 Phase-2 Recall Summary")
    for k in K_VALUES:
        print(f"recall@{k}: {eval_df[f'recall@{k}'].mean():.3f}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Saved evaluation results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
