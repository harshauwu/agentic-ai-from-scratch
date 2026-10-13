import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from debug_logger import debug_log, print_box, print_workflow_summary
from memory import AgentState
from tools import compare_skills

load_dotenv(override=True)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL")
if not API_KEY or not MODEL:
    raise RuntimeError("Set GEMINI_API_KEY and GEMINI_MODEL in .env")

client = genai.Client(api_key=API_KEY)

AVAILABLE_TOOLS = {"compare_skills": compare_skills}

COMPARE_SKILLS_DECLARATION = types.FunctionDeclaration(
    name="compare_skills",
    description="Compare candidate technical skills with job requirements.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "candidate_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
            "required_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "required": ["candidate_skills", "required_skills"],
    },
)

CONFIG = types.GenerateContentConfig(
    system_instruction=(
        "You are a job skills analysis assistant. "
        "Use information from earlier messages in the conversation. "
        "When asked to compare skills, call compare_skills if both lists are known. "
        "If either list is missing, ask for the missing information. "
        "Never invent candidate skills or job requirements."
    ),
    tools=[types.Tool(function_declarations=[COMPARE_SKILLS_DECLARATION])],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


def execute_tool(name: str, arguments: dict):
    if name not in AVAILABLE_TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    return AVAILABLE_TOOLS[name](**arguments)


def run_agent(user_message: str, state: AgentState, max_iterations: int = 6) -> str:
    """Process one user turn using the same AgentState across calls."""
    print_workflow_summary(user_message)
    state.add_user(user_message)
    debug_log("MEMORY", "User message stored in AgentState", {"history_length": len(state.history)})

    for iteration in range(1, max_iterations + 1):
        print_box(
            "AGENT ITERATION",
            f"Step {iteration}\nModel: {MODEL}\nHistory messages: {len(state.history)}",
        )
        print("[DECISION] Sending conversation history to Gemini")

        response = client.models.generate_content(
            model=MODEL,
            contents=state.history,
            config=CONFIG,
        )
        if not response.candidates or response.candidates[0].content is None:
            raise RuntimeError("Gemini returned no usable response")

        model_content = response.candidates[0].content
        state.history.append(model_content)

        calls = [
            part.function_call
            for part in (model_content.parts or [])
            if part.function_call is not None
        ]
        debug_log("MODEL DECISION", f"Gemini requested {len(calls)} tool call(s)", [call.name for call in calls])

        if not calls:
            final_text = response.text or ""
            print_box("FINAL ANSWER", final_text)
            state.add_assistant_text(final_text)
            return final_text

        tool_parts = []
        for call in calls:
            arguments = dict(call.args or {})
            print_box("ACTION", f"Tool: {call.name}\nArgs: {json.dumps(arguments, indent=2)}")
            try:
                result = execute_tool(call.name, arguments)
            except (ValueError, TypeError) as exc:
                result = {"error": str(exc)}
            debug_log("OBSERVATION", f"Tool result for {call.name}", result)
            print_box("OBSERVATION", json.dumps(result, indent=2))

            tool_parts.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": result},
                )
            )

        state.history.append(types.Content(role="user", parts=tool_parts))
        debug_log("MEMORY", "Tool results appended to history", {"history_length": len(state.history)})
        print("[MEMORY] Tool results appended to history")

    raise RuntimeError("Agent stopped after six iterations")


def main():
    state = AgentState.load()
    print("Job Skills Agent — Lesson 04")
    print("Commands: /save, /reset, /history, /exit")

    while True:
        user_message = input("\nYou: ").strip()
        if not user_message:
            continue
        if user_message == "/exit":
            break
        if user_message == "/save":
            state.save()
            print("Saved to conversation.json")
            continue
        if user_message == "/reset":
            state = AgentState()
            print("Started a new in-memory conversation")
            continue
        if user_message == "/history":
            print(json.dumps(state.transcript, indent=2))
            continue

        try:
            run_agent(user_message, state)
        except Exception as exc:
            print(f"[ERROR] {exc}")
            break


if __name__ == "__main__":
    main()
