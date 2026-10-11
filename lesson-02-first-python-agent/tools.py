
def compare_skills(candidate_skills, required_skills):
    """Compare candidate skills with job requirements."""

    candidate = {
        skill.strip().lower()
        for skill in candidate_skills
    }

    required = {
        skill.strip().lower()
        for skill in required_skills
    }

    matched = sorted(candidate & required)
    missing = sorted(required - candidate)

    score = (
        round(len(matched) / len(required) * 100, 2)
        if required
        else 0
    )

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "match_percentage": score
    }
