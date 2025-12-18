def enforce_balance(candidates, final_k=10):
    """
    Enforce Knowledge / Personality balance.
    Deterministic, rule-based.
    """

    knowledge = []
    personality = []
    others = []

    for c in candidates:
        test_type = str(c.get("test_type", "")).lower()

        if "knowledge" in test_type:
            knowledge.append(c)
        elif "personality" in test_type:
            personality.append(c)
        else:
            others.append(c)

    result = []

    # Rule: at least 50% Knowledge, at least 20% Personality
    k_quota = max(1, int(0.5 * final_k))
    p_quota = max(1, int(0.2 * final_k))

    result.extend(knowledge[:k_quota])
    result.extend(personality[:p_quota])

    remaining = [c for c in candidates if c not in result]
    result.extend(remaining)

    return result[:final_k]
