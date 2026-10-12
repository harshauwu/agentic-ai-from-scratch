# Lesson 03 — Build an AI Agent with Multiple Tools

**Stack:** Python 3.11+, Google Gemini (`google-genai`), `python-dotenv`  
**Framework:** None — we implement the agent loop ourselves.  
**Goal:** Learn how an LLM selects tools, Python executes them, and observations inform subsequent decisions.

## 1. What you will learn

By the end of this lesson, you will be able to:

1. Write several ordinary Python functions as agent tools.
2. Describe those functions to Gemini using function declarations.
3. Implement a registry that only permits approved tools.
4. Implement a manual **Decision → Action → Observation → Decision → Final Answer** loop.
5. Observe multiple tool calls and dependent tool execution in your terminal.
6. Explain the design in an AI Engineer interview.

> **Terminology:** “Think” in the terminal means **ask the model to decide what to do next**. It does **not** reveal the model's private internal reasoning.

## 2. Project scenario

A candidate has **Python, SQL, Docker** and **2 years** of experience. A job requires **Python, SQL, Azure, Docker** and **3 years** of experience.

The user asks:

> Compare my skills and experience, calculate my overall match score, and recommend improvements.

Our agent exposes four tools:

| Tool | Purpose | Depends on |
|---|---|---|
| `compare_skills` | Identify matched/missing skills and skill percentage | Candidate and job skills |
| `compare_experience` | Identify experience gap and percentage | Candidate and required years |
| `calculate_match_score` | Combine percentages with example 70/30 weights | Results of first two tools |
| `generate_recommendations` | Suggest improvements | Missing skills and experience gap |

The example score is **educational only**, not a validated hiring assessment.

## 3. How the agent works

```text
User request
     |
     v
Python agent sends conversation + tool definitions to Gemini
     |
     v
DECIDE: Gemini returns tool requests (or final answer)
     |
     +---- No tool requests ----> FINAL ANSWER
     |
     v
ACTION: Python validates the requested tool name and executes it
     |
     v
OBSERVATION: Python collects tool results
     |
     v
Append results to conversation history
     |
     +-----------------------> DECIDE again
```

**Key distinction:** Gemini requests a function call; **Python actually executes the function**. Gemini may request independent tools together, or dependent tools in later iterations. The example uses instructions to encourage dependency ordering; production code should enforce dependencies and validate tool arguments explicitly.

## 4. Create the project

```bash
mkdir -p lesson-03-multiple-tools
cd lesson-03-multiple-tools
python -m pip install -U google-genai python-dotenv
```

Directory layout:

```text
lesson-03-multiple-tools/
├── agent.py
├── tools.py
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Create `requirements.txt`:

```text
google-genai
python-dotenv
```

Create `.env.example`:

```dotenv
GEMINI_API_KEY=replace_with_your_key
GEMINI_MODEL=replace_with_available_function_calling_model
```

Copy it to `.env` and replace both placeholders. Get a key from [Google AI Studio](https://aistudio.google.com/apikey). Choose a model your account can access that supports function calling. Model availability and free-tier limits can change.

Create `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

**Never commit a real API key.**

## 5. Build four tools — `tools.py`

Create `tools.py`:

```python
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
```

### Why separate tools?

Each function performs one business operation and can be tested **without making an API call**. `compare_skills` uses exact, case-insensitive matching, not semantic equivalence.

Try a tool directly:

```bash
python -c 'from tools import compare_skills; print(compare_skills(["Python", "SQL", "Docker"], ["Python", "SQL", "Azure", "Docker"]))'
```

Expected skill match: **75%**; missing skill: **azure**.

## 6. Implement the agent — `agent.py`

Create `agent.py`:

```python
import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import (
    compare_skills,
    compare_experience,
    calculate_match_score,
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
    history = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
    ]

    for step in range(1, max_steps + 1):
        print(f"\n{'=' * 60}\nAGENT ITERATION {step}\n{'=' * 60}")
        print("[THINK / DECIDE] Sending context and tool descriptions to Gemini")

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
        print(f"[MODEL DECISION] Requested {len(calls)} tool(s)")

        if not calls:
            answer = response.text or ""
            print("\n[FINAL ANSWER]\n" + answer)
            return answer

        observations = []
        for call in calls:
            name = call.name
            args = dict(call.args or {})
            print(f"\n[ACTION] {name}")
            print("Arguments:", json.dumps(args, indent=2))
            try:
                result = execute_tool(name, args)
            except (ValueError, TypeError) as exc:
                result = {"error": str(exc)}
            print("[OBSERVATION]", json.dumps(result, indent=2))
            observations.append(
                types.Part.from_function_response(
                    name=name, response={"result": result}
                )
            )

        history.append(types.Content(role="user", parts=observations))
        print("[FEEDBACK] Observations added to history for the next decision")

    raise RuntimeError("Agent stopped: maximum iterations reached")


if __name__ == "__main__":
    message = (
        "My skills are Python, SQL and Docker and I have 2 years of experience. "
        "The job requires Python, SQL, Azure and Docker and 3 years of experience. "
        "Compare my skills and experience, calculate my overall match score, "
        "and recommend improvements."
    )
    run_agent(message)
```

### Important code concepts

- **`DECLARATIONS`** are schemas sent to Gemini; they do not execute Python.
- **`AVAILABLE_TOOLS`** maps allowed names to actual Python functions.
- **`execute_tool()`** dispatches a tool request to a registered function.
- **`history`** stores user input, model requests, and tool observations.
- **`automatic_function_calling.disable=True`** keeps execution under our control so we can see every step.
- **`max_steps=8`** prevents unlimited model-call iterations.

The implementation catches basic argument errors. It does not yet provide full schema validation, strict dependency enforcement, retries, or production-grade security controls.

## 7. Run the agent

```bash
python agent.py
```

You should see stages like:

```text
AGENT ITERATION 1
[THINK / DECIDE] Sending context and tool descriptions to Gemini
[MODEL DECISION] Requested 2 tool(s)
[ACTION] compare_skills
[OBSERVATION] { ... "skill_match_percentage": 75.0 }
[ACTION] compare_experience
[OBSERVATION] { ... "experience_match_percentage": 66.67 }
[FEEDBACK] Observations added to history for the next decision

AGENT ITERATION 2
[THINK / DECIDE] Sending context and tool descriptions to Gemini
[ACTION] calculate_match_score
[OBSERVATION] { ... "overall_match_score": 72.5 }
[ACTION] generate_recommendations
[OBSERVATION] { ... }

AGENT ITERATION 3
[FINAL ANSWER]
...
```

This is **illustrative**. Gemini may choose a different order, combine requests, or ask for clarification. The model's private reasoning is not printed.

### Understand the calculation

- Skill match: `3 / 4 × 100 = 75%`
- Experience match: `2 / 3 × 100 = 66.67%`
- Weighted score: `75 × 0.7 + 66.67 × 0.3 = 72.50%`

## 8. Practice exercises

Change the `message` in `agent.py` and run each test:

| Test | Example request | What to look for |
|---|---|---|
| Skills only | “I know Python and SQL; job needs Python, SQL, Docker. Compare skills.” | `compare_skills` |
| Experience only | “I have 2 years; job requires 4. Compare experience.” | `compare_experience` |
| Full evaluation | Provide skills, years, and request score + advice | Multiple tools and observations |
| Missing data | “I know Python. What's my overall job match?” | Clarifying question; no invented requirements |
| General question | “What is an AI agent?” | Direct answer without unnecessary tools |

**Challenge:** Add a fifth tool, `calculate_missing_skill_count(missing_skills)`, and observe whether Gemini selects it when asked “How many required skills am I missing?”

## 9. Common interview questions

**Q: How does an agent choose between multiple tools?**  
A: The application sends function declarations to the LLM. The LLM uses the request and context to return a structured tool request. Python validates and executes the allowed tool.

**Q: Does Gemini execute Python functions?**  
A: No. Gemini requests a tool call; our Python code executes it and returns an observation. Some SDKs can automate this orchestration, but this lesson deliberately implements it manually.

**Q: Can an LLM request multiple tools at once?**  
A: Yes. Independent tools can be requested in the same response. Our sample executes them one after another in Python; it does not run them concurrently.

**Q: What is a dependent tool call?**  
A: A call that requires a result from another tool. `calculate_match_score` needs the percentages from `compare_skills` and `compare_experience`.

**Q: How do you guarantee the correct tool order?**  
A: Prompt instructions can guide ordering but do not guarantee it. Production systems should enforce dependencies in application code or a workflow engine.

**Q: Why build this without LangGraph?**  
A: To understand tool schemas, model decisions, Python dispatch, observations, state, and stopping conditions before using an orchestration framework.

**Q: How would you make it production-ready?**  
A: Add strict argument validation, dependency checks, API retries/timeouts, structured logs, tests with mocked LLM responses, evaluation datasets, and safeguards for side-effecting tools.

## 10. GitHub commit

From the repository root:

```bash
git add lesson-03-multiple-tools/
git commit -m "feat: add multi-tool Gemini agent tutorial"
git push
```

## 11. Next lesson

**Lesson 04 — Agent Memory and State:** retain candidate skills across multiple user messages, then compare them with job requirements supplied later. Learn the difference between conversation history, in-memory state, and persistent memory.

## References

- [Google Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling)
- [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/)
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/)
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
