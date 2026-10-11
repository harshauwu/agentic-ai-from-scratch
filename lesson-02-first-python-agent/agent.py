import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import compare_skills

from debug_logger import debug_log


# 1. Load configuration
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# 2. Define the tool schema
compare_skills_tool = types.FunctionDeclaration(
    name="compare_skills",
    description=(
        "Compare a candidate's technical skills "
        "with the required skills for a job."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "candidate_skills": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "required_skills": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            }
        },
        "required": [
            "candidate_skills",
            "required_skills"
        ]
    }
)


# 3. Register available Python functions
AVAILABLE_TOOLS = {
    "compare_skills": compare_skills
}


# 4. Execute the requested tool
def execute_tool(tool_name, arguments):

    if tool_name not in AVAILABLE_TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")

    function = AVAILABLE_TOOLS[tool_name]

    return function(**arguments)


# 5. Agent loop
def run_agent(user_message):

    debug_log(
        "USER INPUT",
        "New request received",
        user_message
    )

    history = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_message)
            ]
        )
    ]

    config = types.GenerateContentConfig(
        system_instruction=(
            "You are a job skills analysis assistant. "
            "Use compare_skills when the user requests "
            "a skills comparison. "
            "Do not invent skills. "
            "If either skills list is missing, "
            "ask the user for the missing information."
        ),
        tools=[
            types.Tool(
                function_declarations=[
                    compare_skills_tool
                ]
            )
        ],
        automatic_function_calling=(
            types.AutomaticFunctionCallingConfig(
                disable=True
            )
        )
    )

    for step in range(5):

        debug_log(
            "THINK / DECIDE",
            f"Agent iteration {step + 1}: sending context to Gemini",
            {
                "model": MODEL,
                "history_messages": len(history),
                "available_tools": list(AVAILABLE_TOOLS.keys())
            }
        )

        # 1. Ask Gemini to choose its next action
        response = client.models.generate_content(
            model=MODEL,
            contents=history,
            config=config
        )

        if not response.candidates:
            raise RuntimeError("Gemini returned no candidates.")

        assistant_content = response.candidates[0].content

        if assistant_content is None:
            raise RuntimeError("Gemini returned no content.")

        history.append(assistant_content)

        # 2. Extract function calls
        function_calls = [
            part.function_call
            for part in (assistant_content.parts or [])
            if part.function_call is not None
        ]

        debug_log(
            "MODEL DECISION",
            "Gemini returned its next action",
            {
                "tool_calls_requested": len(function_calls),
                "tool_names": [
                    call.name for call in function_calls
                ]
            }
        )

        # 3. If no tool calls, return the final answer
        if not function_calls:

            final_answer = response.text or ""

            debug_log(
                "FINAL ANSWER",
                "Agent completed successfully",
                final_answer
            )

            return final_answer

        tool_response_parts = []

        # 4. Execute each tool requested by Gemini
        for call in function_calls:

            tool_name = call.name
            arguments = dict(call.args or {})

            debug_log(
                "ACTION",
                f"Executing Python tool: {tool_name}",
                {
                    "tool": tool_name,
                    "arguments": arguments
                }
            )

            result = execute_tool(
                tool_name,
                arguments
            )

            debug_log(
                "OBSERVATION",
                f"Tool {tool_name} completed",
                result
            )

            tool_response_parts.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result}
                )
            )

        # 5. Add observations to conversation history
        history.append(
            types.Content(
                role="user",
                parts=tool_response_parts
            )
        )

        debug_log(
            "FEEDBACK",
            "Tool observations added to conversation history",
            {
                "history_messages": len(history),
                "next_step": "Send observations back to Gemini"
            }
        )

    raise RuntimeError(
        "Agent stopped: maximum iterations reached."
    )

# 6. Run the application
if __name__ == "__main__":

    user_message = (
        "My skills are Python, SQL and Docker. "
        "The job requires Python, SQL, Azure and Docker. "
        "Compare my skills with the job."
    )

    run_agent(user_message)