import json


def validate_assessments(records: list[dict]) -> None:
    """
    Validates data quality without stopping execution or raising errors.
    Logs issues to the console for manual review.
    """
    total_count = len(records)

    # 1. BASELINE CHECK (Will not crash, just alert)
    if total_count == 0:
        print("❌ ERROR: The records list is completely empty!")
        return

    print(f"\n--- Starting Validation for {total_count} items ---")

    # 2. TRACKING COUNTERS
    missing_names = []
    bad_urls = []
    invalid_types = []
    invalid_metadata = []

    # All known SHL Test Type codes
    VALID_CODES = {"A", "B", "C", "D", "E", "K", "P", "S"}

    for i, r in enumerate(records):
        try:
            # Check Name
            name = r.get("name")
            if not name:
                missing_names.append(f"Row {i} (URL: {r.get('url')})")

            # Check URL
            url = r.get("url", "")
            if not url or not url.startswith("http"):
                bad_urls.append(f"Row {i} (Name: {name})")

            # Check Test Type (STRICT but non-crashing)
            tt = r.get("test_type")
            if not isinstance(tt, list):
                invalid_types.append(f"Row {i}: Type is {type(tt)}, expected list")
            elif len(tt) == 0:
                # Flagged because we removed the default 'K'
                invalid_types.append(f"Row {i}: No test types found (Empty List)")
            else:
                unrecognized = [code for code in tt if code not in VALID_CODES]
                if unrecognized:
                    invalid_types.append(f"Row {i}: Unrecognized codes {unrecognized}")

            # Check Metadata (Duration/Remote/Adaptive)
            if not isinstance(r.get("duration"), (int, float)):
                invalid_metadata.append(f"Row {i}: Duration is not a number")

            if r.get("remote_support") not in {"Yes", "No"}:
                invalid_metadata.append(
                    f"Row {i}: remote_support is '{r.get('remote_support')}'"
                )

        except Exception as e:
            # Catch-all for a specific row so the loop continues
            print(f"⚠️ Unexpected error validating row {i}: {e}")

    # 3. PRINT ISSUES (Instead of raising Assertions)
    if missing_names:
        print(f"🚩 MISSING NAMES: {len(missing_names)} items")

    if bad_urls:
        print(f"🚩 BAD URLs: {len(bad_urls)} items")

    if invalid_types:
        # Just print the first 5 to avoid flooding the console
        print(f"🚩 TEST TYPE ISSUES: {len(invalid_types)} items")
        for issue in invalid_types[:5]:
            print(f"   - {issue}")
        if len(invalid_types) > 5:
            print(f"   - ... and {len(invalid_types) - 5} more")

    # 4. STATS CALCULATION
    stats = {code: 0 for code in VALID_CODES}
    for r in records:
        for code in r.get("test_type", []):
            if code in stats:
                stats[code] += 1

    timed = sum(1 for r in records if r.get("duration", 0) > 0)
    remote_yes = sum(1 for r in records if r.get("remote_support") == "Yes")

    print("\n--------------------------------")
    print("VALIDATION SUMMARY (SAFE MODE)")
    print(f"Total Scraped:    {total_count}")
    print(f"Remote Ready:     {remote_yes}")
    print(f"Duration Found:   {timed}")
    print("\nCODE BREAKDOWN:")
    for code, count in sorted(stats.items()):
        print(f"  [{code}]: {count}")
    print("--------------------------------")
    print("✅ Validation complete. Data has been preserved.")
