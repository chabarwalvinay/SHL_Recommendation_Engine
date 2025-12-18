def normalize_test_type(test_types):
    """
    Normalize test_type strings to match Appendix-2.
    """
    mapping = {
        "knowledge and skills": "Knowledge & Skills",
        "knowledge & skills": "Knowledge & Skills",
        "personality and behavior": "Personality & Behaviour",
        "personality & behaviour": "Personality & Behaviour",
        "competencies": "Competencies",
        "ability and aptitude": "Ability & Aptitude",
    }

    normalized = []
    for t in test_types or []:
        key = t.lower()
        normalized.append(mapping.get(key, t))

    return normalized


def format_assessment(a: dict) -> dict:
    adaptive = a.get("adaptive_support")

    # Enforce Yes / No
    if adaptive not in ["Yes", "No"]:
        adaptive = "No"

    return {
        "url": a["url"],
        "name": a["name"],
        "adaptive_support": adaptive,
        "description": a.get("description", ""),
        "duration": int(a.get("duration") or 0),
        "remote_support": a.get("remote_support", "Yes"),
        "test_type": normalize_test_type(a.get("test_type", [])),
    }
