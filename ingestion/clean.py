import re

# SAFE PATTERNS: We removed .* to ensure we don't delete actual data.
_JUNK_PATTERNS = [
    r"contact us for more information",
    r"learn more about this assessment",
    r"read our privacy policy",
    r"terms and conditions apply",
    r"all rights reserved",
    r"cookies help us deliver our services",
    r"click here to view",
]


def clean_text(text: str) -> str:
    if not text:
        return ""

    # We do NOT lowercase here to keep the data professional.
    # We use re.IGNORECASE inside re.sub to find matches regardless of caps.
    cleaned_text = text
    for pattern in _JUNK_PATTERNS:
        # Replace junk with a space to prevent words from sticking together
        cleaned_text = re.sub(pattern, " ", cleaned_text, flags=re.IGNORECASE)

    # CRITICAL: Normalize whitespace.
    # This removes the \n, \t, and triple spaces common in scraped HTML.
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def clean_records(records: list[dict]) -> list[dict]:
    cleaned = []
    for r in records:
        # Create a copy to keep your raw data safe during the process
        item = r.copy()

        # 1. Clean Name (Keep casing, just strip leading/trailing spaces)
        if "name" in item and item["name"]:
            item["name"] = item["name"].strip()

        # 2. Clean Description (The most important cleaning step)
        if "description" in item:
            item["description"] = clean_text(item["description"])

        # 3. Clean Test Type List (Ensures consistency: ['s'] -> ['S'])
        if "test_type" in item and isinstance(item["test_type"], list):
            item["test_type"] = [
                t.strip().upper() for t in item["test_type"] if t.strip()
            ]

        cleaned.append(item)

    return cleaned
