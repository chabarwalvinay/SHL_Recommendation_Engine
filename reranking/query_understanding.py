import os
import json
from typing import Dict, Any

from groq import Groq
from dotenv import load_dotenv


# =========================================================
# ENV + CLIENT
# =========================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment")

client = Groq(api_key=GROQ_API_KEY)


# =========================================================
# DEFAULT FALLBACK (EDGE CASE SAFE)
# =========================================================
DEFAULT_INTENT = {
    "technical_skills": [],
    "behavioral_traits": [],
    "role_signals": [],
    "constraints": {"max_duration": None, "seniority": None},
}


# =========================================================
# PROMPT (STRICT + ROBUST)
# =========================================================
def _build_prompt(query: str) -> str:
    return f"""
You are an expert hiring assistant.

Extract structured hiring intent from the input text.
The input may be:
- a short query
- a long job description
- informal hiring language

Return ONLY valid JSON.
Do NOT add explanations or markdown.

JSON schema:
{{
  "technical_skills": [string],
  "behavioral_traits": [string],
  "role_signals": [string],
  "constraints": {{
    "max_duration": number | null,
    "seniority": string | null
  }}
}}

Rules:
- Normalize skills (e.g., "Java developer" → "Java")
- Behavioral traits are soft skills (communication, teamwork, leadership, etc.)
- role_signals include role type or domain (developer, sales, manager, analyst)
- max_duration is in minutes if mentioned, else null
- If information is missing, return empty lists or nulls

Input text:
\"\"\"{query}\"\"\"
"""


# =========================================================
# MAIN FUNCTION
# =========================================================
def extract_intent(query: str) -> Dict[str, Any]:
    """
    Extract structured hiring intent from a query or JD text.

    Always returns a valid dictionary.
    Never raises on LLM parsing issues.
    """

    if not query or not query.strip():
        return DEFAULT_INTENT.copy()

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": _build_prompt(query)}],
            temperature=0,
            max_tokens=400,
        )

        content = response.choices[0].message.content.strip()

        # Attempt strict JSON parse
        parsed = json.loads(content)

        # ---- Defensive validation ----
        intent = DEFAULT_INTENT.copy()

        intent["technical_skills"] = list(
            set(map(str, parsed.get("technical_skills", [])))
        )

        intent["behavioral_traits"] = list(
            set(map(str, parsed.get("behavioral_traits", [])))
        )

        intent["role_signals"] = list(set(map(str, parsed.get("role_signals", []))))

        constraints = parsed.get("constraints", {}) or {}
        intent["constraints"] = {
            "max_duration": constraints.get("max_duration"),
            "seniority": constraints.get("seniority"),
        }

        return intent

    except Exception as e:
        # 🚨 NEVER break Phase-3 because of LLM
        print(f"[WARN] Query understanding failed: {e}")
        return DEFAULT_INTENT.copy()
