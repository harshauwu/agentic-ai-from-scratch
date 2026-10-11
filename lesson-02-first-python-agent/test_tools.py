
from tools import compare_skills


def test_compare_skills():
    result = compare_skills(
        ["Python", "SQL", "Docker"],
        ["Python", "SQL", "Azure", "Docker"]
    )

    assert result["match_percentage"] == 75.0
    assert result["missing_skills"] == ["azure"]


def test_case_insensitive():
    result = compare_skills(
        ["PYTHON"],
        ["python"]
    )

    assert result["match_percentage"] == 100.0


def test_empty_requirements():
    result = compare_skills(
        ["Python"],
        []
    )

    assert result["match_percentage"] == 0


if __name__ == "__main__":
    test_compare_skills()
    test_case_insensitive()
    test_empty_requirements()

    print("All tests passed!")
