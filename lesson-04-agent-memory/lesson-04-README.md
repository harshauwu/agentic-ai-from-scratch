# Lesson 04 — AI Agent Memory and State Management

**Project:** Job Skills Analysis Agent · **Level:** Beginner · **Stack:** Python + Google Gemini (`google-genai`) · **Framework:** None

In Lesson 03, our agent could choose between multiple tools. But every time we called `run_agent()`, it started with a new conversation. In this lesson, we will make the agent **remember earlier messages within a session** and understand the difference between **conversation history**, **structured state**, and **persistent memory**.

> **What you will build:** An interactive terminal agent that remembers the skills and job requirements you mentioned earlier, calls tools when necessary, and optionally saves the conversation to a local JSON file so you can resume after restarting Python.

## 1. Learning objectives

By the end of this lesson, you will be able to:

1. Explain what agent memory means.
2. Distinguish short-term conversation history from persistent storage.
3. Keep conversation history between user turns.
4. Track useful structured state separately from chat history.
5. See **Decision → Action → Observation → Final Answer** in your terminal.
6. Save and reload conversation history safely for a local learning project.
7. Explain these concepts in an AI Engineer interview.

## 2. The problem: an agent without memory

**Turn 1:**

```text
User: My skills are Python, SQL and Docker.
Agent: What are the job requirements?
```

**Turn 2:**

```text
User: The job requires Python, Azure and Docker. Compare them.
```

If your application creates a brand-new `history` list for Turn 2, Gemini cannot see the skills provided in Turn 1. The model does not automatically remember previous API calls.

**Solution:** Keep the relevant conversation history in your Python application and send it again on the next API call.

```text
TURN 1                                 TURN 2
User message                           New user message
     |                                       |
     v                                       v
Append to history                    Append to SAME history
     |                                       |
     v                                       v
Gemini receives history              Gemini receives BOTH turns
     |                                       |
     v                                       v
Agent reply                           Tool call / final reply
     |                                       |
     +---- saved in history <---------------+
```

**Key idea:** The **application manages memory**. The LLM sees only the context supplied with each request.

## 3. Three concepts you must not confuse

| Concept | Meaning | Example | Survives restart? |
|---|---|---|---|
| **Conversation history** | Ordered messages and tool results in the current session | User said “I know Python” | No, unless saved |
| **Structured agent state** | Explicit variables representing important facts or workflow progress | `candidate_skills = ["Python", "SQL"]` | No, unless saved |
| **Persistent memory** | Data stored outside the running process | JSON file or database record | Yes |

### Does the LLM itself store our conversation?

Not in the way this tutorial uses the API. We explicitly pass `contents=history` on each request. The `history` variable is stored in Python memory. If the Python process ends, it disappears unless we save it.

### Is conversation history the same as structured state?

No. A transcript can contain facts, but extracting the current facts from it can be unreliable. Structured state is explicit and easier to validate. We will use **history as the model's context** and introduce a simple `AgentState` object to manage that context. A production system might also store validated fields such as skills, experience, and job requirements separately.

## 4. Project structure

```text
agentic-ai-from-scratch/
└── lesson-04-agent-memory/
    ├── README.md
    ├── agent.py
    ├── tools.py
    ├── memory.py
    ├── requirements.txt
    ├── .env.example
    └── .gitignore
```

Create the folder and activate your Python environment:

```bash
mkdir -p lesson-04-agent-memory
cd lesson-04-agent-memory
conda activate Agent-hugging-face   # Or activate your own virtual environment
python -m pip install -U google-genai python-dotenv
```

Create `requirements.txt`:

```text
google-genai
python-dotenv
```

Create `.env.example`:

```dotenv
GEMINI_API_KEY=replace_with_your_api_key
GEMINI_MODEL=replace_with_an_available_function_calling_model
```

Copy it to `.env` and set your actual values. Find an available function-calling model in your Google account rather than assuming a model name works everywhere.

Create `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
conversation.json
.DS_Store
```

**Never commit your API key or a conversation file containing personal information.**

## 5. Step 1 — Reuse a simple tool (`tools.py`)

We will use one tool first so we can focus on memory. You already learned multi-tool registration in Lesson 03; the same memory design works with all four tools.

```python
# tools.py

def compare_skills(candidate_skills, required_skills):
    candidate = {skill.strip().lower() for skill in candidate_skills}
    required = {skill.strip().lower() for skill in required_skills}

    matched = sorted(candidate & required)
    missing = sorted(required - candidate)
    percentage = round(len(matched) / len(required) * 100, 2) if required else 0.0

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_match_percentage": percentage,
    }
```

Test it independently:

```bash
python -c 'from tools import compare_skills; print(compare_skills(["Python", "SQL"], ["Python", "Azure"]))'
```

Expected result: one matched skill (`python`), one missing skill (`azure`), and a **50%** match.

## 6. Step 2 — Create a state object (`memory.py`)

This class keeps the conversation history across user turns. It also lets us save and restore text-based chat messages. To keep the lesson safe and simple, we will **not** persist raw SDK tool-call objects; we save a compact text transcript instead. After loading it, the agent can continue chatting, but previously executed tool calls are represented as text summaries rather than native function-call messages.

```python
# memory.py
import json
from dataclasses import dataclass, field
from pathlib import Path

from google.genai import types


@dataclass
class AgentState:
    # Live Gemini conversation, including function calls and observations.
    history: list = field(default_factory=list)

    # Simple text transcript that can be stored in JSON.
    transcript: list[dict[str, str]] = field(default_factory=list)

    def add_user(self, message: str):
        self.history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=message)])
        )
        self.transcript.append({"role": "user", "text": message})

    def add_assistant_text(self, message: str):
        # Only use this for a final assistant message, not a tool call.
        self.transcript.append({"role": "model", "text": message})

    def save(self, filename: str = "conversation.json"):
        Path(filename).write_text(
            json.dumps(self.transcript, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, filename: str = "conversation.json"):
        path = Path(filename)
        if not path.exists():
            return cls()

        transcript = json.loads(path.read_text(encoding="utf-8"))
        state = cls()
        for entry in transcript:
            role = entry["role"]
            text = entry["text"]
            if role not in ("user", "model") or not isinstance(text, str):
                raise ValueError("Invalid saved conversation entry")
            state.history.append(
                types.Content(role=role, parts=[types.Part.from_text(text=text)])
            )
            state.transcript.append({"role": role, "text": text})
        return state
```

### Explain the important lines

- `@dataclass` creates a convenient class for holding state.
- `field(default_factory=list)` gives each agent session its **own** lists.
- `history` contains the actual Gemini SDK message objects.
- `transcript` contains only serializable text messages.
- `save()` writes the transcript to a JSON file.
- `load()` rebuilds a conversation from saved text messages.

**Important limitation:** A text transcript is enough for our learning exercise, but it is **not a lossless serialization** of Gemini tool calls. Production-grade persistence should save typed messages and tool events, use schema validation, and enforce access controls.

## 7. Step 3 — Create the memory-enabled agent (`agent.py`)

This is a **complete runnable example**. It uses manual function calling so you can see every stage in the terminal.

```python
# agent.py
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

from memory import AgentState
from tools import compare_skills

load_dotenv(override=True)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL")
if not API_KEY or not MODEL:
    raise RuntimeError("Set GEMINI_API_KEY and GEMINI_MODEL in .env")

client = genai.Client(api_key=API_KEY)

# Tool registry: maps an allowed name to a Python function.
AVAILABLE_TOOLS = {"compare_skills": compare_skills}

# Tool declaration: tells Gemini what it may request.
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


def run_agent(user_message: str, state: AgentState) -> str:
    """Process one user turn using the SAME state across calls."""
    state.add_user(user_message)

    for iteration in range(1, 7):
        print(f"\n===== AGENT ITERATION {iteration} =====")
        print("[DECISION] Sending conversation history to Gemini")
        print(f"[MEMORY] History messages: {len(state.history)}")

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

        if not calls:
            final_text = response.text or ""
            print("[FINAL ANSWER]", final_text)
            state.add_assistant_text(final_text)
            return final_text

        tool_parts = []
        for call in calls:
            arguments = dict(call.args or {})
            print(f"[ACTION] {call.name}({json.dumps(arguments)})")
            try:
                result = execute_tool(call.name, arguments)
            except (ValueError, TypeError) as exc:
                result = {"error": str(exc)}
            print("[OBSERVATION]", json.dumps(result, indent=2))

            tool_parts.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": result},
                )
            )

        # Send the observations back on the next iteration.
        state.history.append(types.Content(role="user", parts=tool_parts))
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
            # This simple tutorial stops after an API or tool-loop failure,
            # avoiding a potentially inconsistent partial conversation.
            break


if __name__ == "__main__":
    main()
```

### Why is `state` passed into `run_agent()`?

In Lesson 02, the function created a new history list every time:

```python
# Old approach — forgets earlier turns

def run_agent(user_message):
    history = []
```

Now we use the **same** `AgentState` object for each turn:

```python
# New approach — keeps earlier turns
state = AgentState()
run_agent("My skills are Python and SQL", state)
run_agent("The job requires Python and Azure. Compare them.", state)
```

This is the core change that adds short-term memory.

## 8. Step 4 — Run and observe memory

Start the agent:

```bash
python agent.py
```

Type these messages **one at a time**:

```text
You: My skills are Python, SQL and Docker.

Agent: What skills does the job require?

You: The job requires Python, Azure and Docker. Compare my skills.

[DECISION] Sending conversation history to Gemini
[MEMORY] History messages: 3
[ACTION] compare_skills(...)
[OBSERVATION] { ... }
[FINAL ANSWER] Your matching skills are Python and Docker...
```

The exact model response and number of messages may vary. The important behavior is that **the second user message can use information from the first**.

### Try these commands

| Command | Behavior |
|---|---|
| `/history` | Prints the saved text transcript |
| `/save` | Saves the text transcript to `conversation.json` |
| `/reset` | Starts a new in-memory session (does not delete the saved file) |
| `/exit` | Ends the program |

**Persistence experiment:**

1. Start the program and enter your skills.
2. Type `/save`.
3. Type `/exit`.
4. Run `python agent.py` again.
5. Enter job requirements and ask for a comparison.
6. Observe that the earlier text transcript is available again.

The saved transcript contains user messages and final assistant text, **not full native tool-call events**. This is sufficient for the example, but it is a deliberate simplification.

## 9. Follow the workflow step by step

```text
USER TURN 1
  |
  v
AgentState.add_user()
  |
  v
history = [user_turn_1]
  |
  v
Gemini: asks for job requirements
  |
  v
history = [user_turn_1, assistant_turn_1]

USER TURN 2
  |
  v
AgentState.add_user()
  |
  v
history = [user_turn_1, assistant_turn_1, user_turn_2]
  |
  v
[DECISION] Gemini sees both turns
  |
  v
[ACTION] compare_skills()
  |
  v
[OBSERVATION] skill comparison result
  |
  v
Append tool response to history
  |
  v
[DECISION] Gemini receives updated history
  |
  v
[FINAL ANSWER] Explanation of skill match
```

**The key lesson:** The agent's apparent memory is the result of **state management and context reuse**, not magic inside the model.

## 10. What about structured state?

Our example uses conversation history. In a real JobHarbor system, we should also maintain structured, validated fields:

```python
candidate_profile = {
    "skills": ["Python", "SQL", "Docker"],
    "experience_years": 2,
}

job_requirements = {
    "skills": ["Python", "Azure", "Docker"],
    "experience_years": 3,
}
```

This makes it easier to:

- Validate required fields before running tools.
- Detect when a user updates a skill or changes jobs.
- Avoid repeatedly asking the LLM to infer facts from old messages.
- Store candidate and job data in PostgreSQL later.

**Production design principle:** Use the LLM to interpret language, but keep authoritative business data in explicit application state or a database.

## 11. Common memory problems

### Problem A: The agent forgets information

**Cause:** A new history list is created for each user turn.

**Fix:** Pass the same `AgentState` object into `run_agent()`.

### Problem B: The agent remembers an old job after `/reset`

**Cause:** The program may have loaded `conversation.json` again after restart.

**Fix:** `/reset` only clears in-memory state. Remove the local JSON file yourself if you also want to clear the persisted tutorial transcript.

### Problem C: Context becomes too large

**Cause:** Every previous message is sent on every request. This increases token usage and may exceed the model's context window.

**Future solutions:** Keep recent messages, summarize older conversations, or retrieve only relevant saved facts. Be careful not to discard information needed to complete an in-progress tool interaction.

### Problem D: The agent invents missing details

**Cause:** The model may infer unsupported information from vague context.

**Fix:** Require explicit data validation, ask clarifying questions, and use structured state for important facts.

### Problem E: Sensitive data is saved locally

**Cause:** `/save` writes the transcript as readable JSON.

**Fix:** Avoid saving sensitive information in this learning project. For production systems, use appropriate access controls, retention policies, and encryption.

## 12. Test scenarios

| Test | Input sequence | Expected behavior |
|---|---|---|
| 1. Memory across turns | “I know Python and SQL” → “Job needs Python and Azure” | Uses earlier skills |
| 2. Missing information | “Compare my skills” | Asks for missing details |
| 3. Direct answer | “What is an AI agent?” | Answers without calling `compare_skills` |
| 4. New session | `/reset` → “Compare them” | Requests missing context |
| 5. Save and resume | Enter skills → `/save` → restart → enter job | Uses saved text context |
| 6. Tool observation | Ask for complete comparison | Prints ACTION and OBSERVATION |

> These are expected behaviors to test, not guarantees of model output. Test the actual results and investigate failures.

## 13. Interview questions and sample answers

**Q1. What is memory in an LLM agent?**

Memory is the application-managed information made available to the model across steps or conversations. It can include message history, structured state, summaries, and retrieved persistent records.

**Q2. Does the Gemini API automatically remember previous requests?**

Not in this implementation. I explicitly maintain conversation history in Python and send it with each request.

**Q3. What is the difference between short-term and long-term memory?**

Short-term memory is context available during the current session. Long-term memory is information persisted externally, such as in a database, and retrieved for future sessions.

**Q4. What is agent state?**

Agent state is the data needed to continue a workflow, such as message history, candidate skills, current job requirements, tool results, and execution status.

**Q5. How did you implement memory in your project?**

I created an `AgentState` class that maintains the Gemini conversation history across user turns. Each new message is appended to the same state, and the model receives that updated history. I also added a simple JSON transcript save/load feature to demonstrate persistence.

**Q6. Why not send the entire conversation forever?**

Long histories increase latency, cost, and the risk of exceeding context limits. A scalable system uses context-window management, summaries, and retrieval of relevant information.

**Q7. What is the difference between memory and RAG?**

Memory refers to information retained or retrieved for an agent. Retrieval-Augmented Generation (RAG) is a pattern that retrieves relevant external information and includes it in the model's context. RAG can be used to implement some forms of long-term memory, but the two concepts are not identical.

**Q8. How would you improve this for production?**

I would use structured state, persist full typed tool events, separate users and sessions, validate data, implement context-window management, and apply security controls to stored information.

## 14. GitHub commit

From the repository root, after adding and testing the lesson files:

```bash
git add lesson-04-agent-memory/
git commit -m "feat: add agent conversation memory and state management tutorial"
git push
```

## 15. Summary

You have learned the key distinction between **LLM reasoning** and **application-managed memory**:

```text
User message
    |
    v
AgentState (history)
    |
    v
Gemini decision
    |
    +--> Tool request --> Python tool --> Observation --+
    |                                                    |
    +<---------------- updated history <-----------------+
    |
    v
Final answer
    |
    v
Save transcript (optional)
```

**Remember:** A model can only use information it receives in its current context. Your Python application decides what to keep, what to retrieve, and what to send.

### Next lesson: Lesson 05 — LangGraph

We will rebuild this agent using **LangGraph**, and compare our manual Python loop with framework-managed **state, nodes, edges, and tool execution**.

### Official references

- [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/)
- [Gemini Function Calling](https://ai.google.dev/gemini-api/docs/function-calling)
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/)
