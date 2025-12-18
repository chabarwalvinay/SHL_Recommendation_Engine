import json
import pandas as pd
import os

# Ensure these imports point to your actual files
from crawl import crawl_shl_assessments
from clean import clean_records
from validate import validate_assessments


def save_json(data, filepath):
    """Helper to save JSON data safely."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            data, f, indent=4, ensure_ascii=False
        )  # Changed indent to 4 for better readability
    print(f" -> Saved: {filepath}")


def main():
    print("Starting Phase 1 ingestion...")

    # --- STEP 1: CRAWL ---
    # The crawler now has a timeout of 60s and random delays to prevent the 30s/it hang.
    records = crawl_shl_assessments()

    if not records:
        print("❌ CRITICAL: No records captured. Exiting ingestion.")
        return

    print(f"\nCrawl complete. Captured {len(records)} records.")

    # --- STEP 2: SAVE RAW DATA IMMEDIATELY ---
    # Always keep the raw scrape untouched.
    raw_path = "data/raw/shl_assessments_raw.json"
    print("Saving raw backup...")
    save_json(records, raw_path)

    # --- STEP 3: CLEAN & ASSIGN IDs ---
    print("Cleaning records...")
    try:
        # 1. Run the safe cleaning logic (Option 2 we discussed)
        records = clean_records(records)

        # 2. Assign stable IDs AFTER cleaning
        for i, r in enumerate(records):
            r["assessment_id"] = f"shl_{i:05d}"

    except Exception as e:
        print(f"⚠️ Warning: Cleaning routine failed: {e}")
        print("Continuing with raw records to ensure data preservation.")

    # --- STEP 4: SAVE PROCESSED DATA ---
    proc_path = "data/processed/shl_assessments.json"
    parquet_path = "data/processed/shl_assessments.parquet"

    print("Saving processed data...")
    save_json(records, proc_path)

    try:
        # Convert list of dicts to DataFrame for Parquet
        df = pd.DataFrame(records)

        # Parquet doesn't like lists (test_type) in some older versions.
        # Modern pandas handles it fine, but we'll wrap it in a try just in case.
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        print(f" -> Saved: {parquet_path}")
    except Exception as e:
        print(f"⚠️ Could not save Parquet: {e}")

    # --- STEP 5: VALIDATE (SAFE MODE) ---
    # Our new validation script flags issues without crashing the whole process.
    print("\nRunning Validation...")
    try:
        validate_assessments(records)
        print("\n✅ Ingestion process finished.")
    except Exception as e:
        print(f"\n❌ UNEXPECTED VALIDATION ERROR: {e}")


if __name__ == "__main__":
    main()
