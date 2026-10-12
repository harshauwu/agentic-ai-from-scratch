def compare_skills(candidate_skills, required_skills):
    candidate = {s.strip().lower() for s in candidate_skills}
    required = {s.strip().lower() for s in required_skills}
    matched = sorted(candidate & required)
    missing = sorted(required - candidate)
    percentage = round(len(matched) / len(required) * 100, 2) if required else 0.0
    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_match_percentage": percentage,
    }


def compare_experience(candidate_years, required_years):
    if candidate_years < 0 or required_years < 0:
        raise ValueError("Experience cannot be negative")
    percentage = (
        round(min(candidate_years / required_years, 1) * 100, 2)
        if required_years else 100.0
    )
    return {
        "candidate_years": candidate_years,
        "required_years": required_years,
        "experience_gap": max(required_years - candidate_years, 0),
        "experience_match_percentage": percentage,
    }


def calculate_match_score(skill_match_percentage, experience_match_percentage):
    if not (0 <= skill_match_percentage <= 100 and 0 <= experience_match_percentage <= 100):
        raise ValueError("Percentages must be between 0 and 100")
    score = skill_match_percentage * 0.7 + experience_match_percentage * 0.3
    return {
        "overall_match_score": round(score, 2),
        "skill_weight": 0.7,
        "experience_weight": 0.3,
    }


def generate_recommendations(missing_skills, experience_gap):
    if experience_gap < 0:
        raise ValueError("Experience gap cannot be negative")
    suggestions = [f"Consider developing your {skill} skills." for skill in missing_skills]
    if experience_gap > 0:
        suggestions.append(f"The job asks for {experience_gap:g} more year(s) of experience.")
    if not suggestions:
        suggestions.append("You meet the listed skills and experience requirements.")
    return {"recommendations": suggestions}
