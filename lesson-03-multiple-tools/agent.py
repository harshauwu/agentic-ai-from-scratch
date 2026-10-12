import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from debug_logger import debug_log, print_workflow_stage, print_workflow_summary
from tools import (
    calculate_match_score,
    compare_experience,
    compare_skills,
    generate_recommendations,
)

# 1) Connect to Gemini
load_dotenv(override=True)
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL")
if not API_KEY or not MODEL:
    raise ValueError("Set GEMINI_API_KEY and GEMINI_MODEL in .env")
client = genai.Client(api_key=API_KEY)

# 2) Allowlist: only these Python functions may be executed
AVAILABLE_TOOLS = {
    "compare_skills": compare_skills,
    "compare_experience": compare_experience,
    "calculate_match_score": calculate_match_score,
    "generate_recommendations": generate_recommendations,
}

# 3) Tell Gemini what each function does and which arguments it accepts
DECLARATIONS = [
    types.FunctionDeclaration(
        name="compare_skills",
        description="Compare candidate technical skills with job skills.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "candidate_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
                "required_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
            },
            "required": ["candidate_skills", "required_skills"],
        },
    ),
    types.FunctionDeclaration(
        name="compare_experience",
        description="Compare candidate and required years of experience.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "candidate_years": {"type": "NUMBER"},
                "required_years": {"type": "NUMBER"},
            },
            "required": ["candidate_years", "required_years"],
        },
    ),
    types.FunctionDeclaration(
        name="calculate_match_score",
        description="Calculate overall score from PREVIOUS tool results: skill and experience percentages.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "skill_match_percentage": {"type": "NUMBER"},
                "experience_match_percentage": {"type": "NUMBER"},
            },
            "required": ["skill_match_percentage", "experience_match_percentage"],
        },
    ),
    types.FunctionDeclaration(
        name="generate_recommendations",
        description="Recommend improvements using missing skills and experience gap from PREVIOUS tool results.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "missing_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
                "experience_gap": {"type": "NUMBER"},
            },
            "required": ["missing_skills", "experience_gap"],
        },
    ),
]

CONFIG = types.GenerateContentConfig(
    system_instruction=(
        "You are a job analysis assistant. Do not invent candidate or job details. "
        "Ask for missing information. Use comparison tools before calculating "
        "an overall score or recommendations. Use actual previous tool results "
        "for dependent tool calls. Avoid unnecessary repeated calls."
    ),
    tools=[types.Tool(function_declarations=DECLARATIONS)],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


def execute_tool(name, arguments):
    """Execute only registered tools; do not execute arbitrary model-generated code."""
    if name not in AVAILABLE_TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    return AVAILABLE_TOOLS[name](**arguments)


def run_agent(user_message, max_steps=8):
    print_workflow_summary(user_message)
    history = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
    ]

    for step in range(1, max_steps + 1):
        print_workflow_stage(
            "AGENT ITERATION",
            f"Step {step}: THINK / DECIDE",
            {
                "model": MODEL,
                "history_messages": len(history),
                "available_tools": sorted(AVAILABLE_TOOLS.keys()),
            },
        )

        response = client.models.generate_content(
            model=MODEL, contents=history, config=CONFIG
        )
        if not response.candidates or not response.candidates[0].content:
            raise RuntimeError("Gemini returned no usable response")

        assistant_content = response.candidates[0].content
        history.append(assistant_content)
        calls = [
            part.function_call
            for part in (assistant_content.parts or [])
            if part.function_call is not None
        ]
        debug_log("MODEL DECISION", f"Gemini requested {len(calls)} tool(s)", [call.name for call in calls])

        if not calls:
            answer = response.text or ""
            print_workflow_stage("FINAL ANSWER", "Agent finished without more tool calls.", answer)
            return answer

        observations = []
        for call in calls:
            name = call.name
            args = dict(call.args or {})
            print_workflow_stage("ACTION", f"Executing {name}", args)
            try:
                result = execute_tool(name, args)
            except (ValueError, TypeError) as exc:
                result = {"error": str(exc)}
            debug_log("OBSERVATION", f"Tool result for {name}", result)
            observations.append(
                types.Part.from_function_response(
                    name=name, response={"result": result}
                )
            )

        history.append(types.Content(role="user", parts=observations))
        print_workflow_stage(
            "FEEDBACK",
            "Tool observations were appended to the conversation history.",
            {"next_step": "Send observations back to Gemini for the next decision"},
        )

    raise RuntimeError("Agent stopped: maximum iterations reached")


if __name__ == "__main__":
    message = (
        "My skills are Python, SQL and Docker and I have 2 years of experience. "
        "The job requires Python, SQL, Azure and Docker and 3 years of experience. "
        "Compare my skills and experience, calculate my overall match score, "
        "and recommend improvements."
    )
    run_agent(message)
