def compute_score(candidate: dict, intent: dict) -> float:
    """
    Compute final relevance score for a candidate.
    Fully safe for missing or None fields.
    """

    score = 0.0

    # 1️⃣ Base similarity score (Phase-2)
    retrieval_score = candidate.get("retrieval_score")
    if retrieval_score is None:
        retrieval_score = 0.0

    score += 0.5 * float(retrieval_score)

    # Safe text aggregation
    text = (
        (candidate.get("name") or "") + " " + (candidate.get("description") or "")
    ).lower()

    # 2️⃣ Technical skill overlap
    for skill in intent.get("technical_skills", []):
        if skill and skill.lower() in text:
            score += 0.1

    # 3️⃣ Behavioral traits
    for trait in intent.get("behavioral_traits", []):
        if trait and trait.lower() in text:
            score += 0.05

    # 4️⃣ Duration constraint
    constraints = intent.get("constraints") or {}
    max_duration = constraints.get("max_duration")

    duration = candidate.get("duration")
    if max_duration is not None and duration is not None:
        try:
            if int(duration) <= int(max_duration):
                score += 0.05
        except (ValueError, TypeError):
            pass

    return round(score, 4)
