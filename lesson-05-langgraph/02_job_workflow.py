from typing_extensions import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from workflow_debug import show_graph, show_update


class JobState(TypedDict):
    candidate_skills: list[str]
    required_skills: list[str]
    missing_input: NotRequired[bool]
    matched_skills: NotRequired[list[str]]
    missing_skills: NotRequired[list[str]]
    match_percentage: NotRequired[float]
    report: NotRequired[str]


def validate_input(state: JobState) -> dict[str, bool]:
    missing = not state.get("candidate_skills") or not state.get("required_skills")
    print(f"[NODE ACTION] validate_input found missing data: {missing}")
    return {"missing_input": missing}


def route_after_validation(state: JobState) -> str:
    path = "request_details" if state["missing_input"] else "compare_skills"
    print(f"[CONDITIONAL EDGE] validate_input → {path}")
    return path


def request_details(state: JobState) -> dict[str, str]:
    print("[NODE ACTION] request_details asks for both skill lists")
    return {"report": "Please provide both your skills and the job requirements."}


def compare_skills_node(state: JobState) -> dict[str, object]:
    print("[NODE ACTION] compare_skills compares normalized skill sets")
    candidate = {skill.strip().lower() for skill in state["candidate_skills"]}
    required = {skill.strip().lower() for skill in state["required_skills"]}
    matched = sorted(candidate & required)
    missing = sorted(required - candidate)
    percentage = round(len(matched) / len(required) * 100, 2)
    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "match_percentage": percentage,
    }


def make_report(state: JobState) -> dict[str, str]:
    print("[NODE ACTION] make_report formats the comparison result")
    report = (
        f"Match: {state['match_percentage']}% | "
        f"Matched: {', '.join(state['matched_skills']) or 'none'} | "
        f"Missing: {', '.join(state['missing_skills']) or 'none'}"
    )
    return {"report": report}


builder = StateGraph(JobState)
builder.add_node("validate_input", validate_input)
builder.add_node("request_details", request_details)
builder.add_node("compare_skills", compare_skills_node)
builder.add_node("make_report", make_report)
builder.add_edge(START, "validate_input")
builder.add_conditional_edges(
    "validate_input",
    route_after_validation,
    {"request_details": "request_details", "compare_skills": "compare_skills"},
)
builder.add_edge("request_details", END)
builder.add_edge("compare_skills", "make_report")
builder.add_edge("make_report", END)
graph = builder.compile()


def run_example(title: str, example: JobState) -> None:
    print(f"\n\n### {title}")
    print("[INPUT STATE]", example)
    report = None
    for update in graph.stream(example, stream_mode="updates"):
        show_update(update)
        for values in update.values():
            if "report" in values:
                report = values["report"]
    print("[FINAL REPORT]", report)


if __name__ == "__main__":
    show_graph(graph, "DETERMINISTIC JOB WORKFLOW WITH CONDITIONAL ROUTING")
    run_example(
        "Complete requirements branch",
        {
            "candidate_skills": ["Python", "SQL", "Docker"],
            "required_skills": ["Python", "SQL", "Azure", "Docker"],
        },
    )
    run_example(
        "Missing requirements branch",
        {"candidate_skills": ["Python"], "required_skills": []},
    )
