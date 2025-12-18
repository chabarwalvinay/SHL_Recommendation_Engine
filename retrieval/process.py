import re
import unicodedata
from typing import Optional


# =========================================================
# CONFIG
# =========================================================

# Hard cap to prevent embedding overflow
MAX_CHARS = 3500

# Minimum useful query length
MIN_CHARS = 5

# Common JD boilerplate patterns (CONSERVATIVE)
BOILERPLATE_PATTERNS = [
    r"\babout us\b",
    r"\bwho we are\b",
    r"\bour culture\b",
    r"\bdiversity[, ]+equity[, ]+and inclusion\b",
    r"\bequal opportunity employer\b",
    r"\bbenefits include\b",
    r"\bwhy join us\b",
    r"#\w+",  # hashtags
]


# =========================================================
# HELPERS
# =========================================================


def _normalize_unicode(text: str) -> str:
    """Normalize unicode characters (quotes, dashes, etc.)."""
    return unicodedata.normalize("NFKD", text)


def _remove_boilerplate(text: str) -> str:
    """Remove obvious non-signal boilerplate text."""
    lowered = text.lower()
    for pattern in BOILERPLATE_PATTERNS:
        lowered = re.sub(pattern, " ", lowered, flags=re.IGNORECASE)
    return lowered


def _cleanup_whitespace(text: str) -> str:
    """Collapse excessive whitespace and line breaks."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _remove_weird_symbols(text: str) -> str:
    """Remove emojis and non-informative symbols."""
    # Keep letters, numbers, punctuation commonly useful in skills
    return re.sub(r"[^\w\s.,:/+-]", " ", text)


# =========================================================
# MAIN API
# =========================================================


def preprocess_query(raw_query: Optional[str]) -> str:
    """
    Phase-2 query normalization.
    - Removes noise
    - Preserves semantic signal
    - NO intent extraction
    - NO filtering
    """

    if not raw_query or not isinstance(raw_query, str):
        return ""

    # Step 1: Unicode normalization
    text = _normalize_unicode(raw_query)

    # Step 2: Lowercase (embedding friendly)
    text = text.lower()

    # Step 3: Remove obvious boilerplate
    text = _remove_boilerplate(text)

    # Step 4: Remove emojis / strange symbols
    text = _remove_weird_symbols(text)

    # Step 5: Whitespace cleanup
    text = _cleanup_whitespace(text)

    # Step 6: Length guard (truncate safely)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    # Step 7: Minimum length guard
    if len(text) < MIN_CHARS:
        return ""

    return text
