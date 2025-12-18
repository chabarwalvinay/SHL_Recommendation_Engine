def compute_score(candidate: dict, intent: dict) -> float:
    """
    Compute final relevance score for a candidate.
    Safe for missing fields.
    """

    score = 0.0

    # 1️⃣ Base similarity score (Phase-2)
    score += 0.5 * float(candidate.get("retrieval_score", 0.0))

    text = (
        (candidate.get("name") or "") + " " + (candidate.get("description") or "")
    ).lower()

    # 2️⃣ Technical skill overlap
    for skill in intent.get("technical_skills", []):
        if skill.lower() in text:
            score += 0.1

    # 3️⃣ Behavioral traits
    for trait in intent.get("behavioral_traits", []):
        if trait.lower() in text:
            score += 0.05

    # 4️⃣ Duration constraint
    max_duration = intent.get("constraints", {}).get("max_duration")
    duration = candidate.get("duration")

    if max_duration and duration and duration <= max_duration:
        score += 0.05

    return round(score, 4)
