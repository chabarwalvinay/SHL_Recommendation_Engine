from reranking.query_understanding import extract_intent
from reranking.scoring import compute_score
from reranking.balance import enforce_balance


def rerank(query: str, candidates: list, final_k: int = 10) -> list:
    """
    Full Phase-3 reranking pipeline.
    """

    # 1️⃣ Understand query (1 LLM call)
    intent = extract_intent(query)

    # 2️⃣ Score candidates
    for c in candidates:
        c["final_score"] = compute_score(c, intent)

    # 3️⃣ Sort by final score
    candidates = sorted(candidates, key=lambda x: x.get("final_score", 0), reverse=True)

    # 4️⃣ Enforce K/P balance
    return enforce_balance(candidates, final_k)
