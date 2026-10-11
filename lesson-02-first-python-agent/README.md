# Lesson 02: Your First Python Agent

Build a small tool-using AI agent with Python and Google's Gemini API—without an agent framework. This lesson focuses on the mechanics that frameworks usually hide: describing a tool, receiving a tool-call request, executing it in Python, and returning its result to the model.

## What you'll build

A job-skills assistant that compares a candidate's skills with the skills required for a job. Gemini decides whether to call the Python function `compare_skills`; the application runs that function and sends its result back to Gemini for a natural-language response.

```text
User request → Gemini decision → Python tool → tool result → Gemini response
```

The model does not execute Python. It requests a function call; this application controls which registered function can run.

## Learning goals

By the end of this lesson, you should be able to:

- Explain the difference between an LLM and an AI agent.
- Describe a tool with a function declaration and JSON-style parameter schema.
- Map permitted tool names to Python functions and execute a requested tool.
- Follow the decision → action → observation loop in a small agent.
- Use conversation history to pass tool results back to the model.
- Read logs that show each stage of a run.

## Project files

| File | Purpose |
|---|---|
| `agent.py` | Loads configuration, creates the Gemini client, declares the tool, and runs the agent loop. |
| `tools.py` | Implements the deterministic skill comparison. |
| `debug_logger.py` | Prints timestamped, formatted details about each stage. |
| `requirements.txt` | Lists the Python packages needed by this lesson. |
| `.env.example` | Template for the Gemini API key and optional model setting. |

## How the agent works

### 1. Describe the tool

`agent.py` creates a `compare_skills` function declaration. Its schema tells Gemini the tool's name, purpose, and two required string arrays: `candidate_skills` and `required_skills`. The declaration describes the function; it does not implement it.

### 2. Register and execute permitted tools

`AVAILABLE_TOOLS` maps the name `compare_skills` to the Python function imported from `tools.py`. `execute_tool()` rejects names that are not in this registry, then calls the registered function with the model-provided arguments. This allowlist is a key boundary: never execute arbitrary code supplied by a model.

### 3. Ask Gemini to decide

`run_agent()` places the user's request in a conversation history and sends it to Gemini with a system instruction and the tool declaration. Automatic function calling is disabled so this lesson can show the orchestration explicitly.

### 4. Run a tool, if requested

The agent inspects the model response for function calls. If it finds one, it executes each requested registered tool and builds a function-response part containing the result.

### 5. Return the observation to Gemini

The tool response is appended to the history. On the next loop iteration, Gemini can use that observation to formulate its answer or request another action. If the response contains no tool calls, the agent returns `response.text` as the final answer.

The loop allows at most five model decisions. Reaching the limit raises a `RuntimeError` rather than allowing an unbounded loop.

## Skill comparison behavior

`compare_skills()` in `tools.py`:

1. Trims whitespace and lowercases each skill.
2. Treats the normalized lists as sets, so duplicates are removed.
3. Returns the alphabetically sorted skills found in both sets and the required skills that are missing.
4. Calculates the match percentage from the number of matched unique required skills. If the required list is empty, the score is `0`.

For example, candidate skills `Python`, `SQL`, and `Docker` compared with requirements `Python`, `SQL`, `Azure`, and `Docker` produce three matches, one missing skill (`azure`), and a score of `75.0`.

This is an exact, case-insensitive comparison—not semantic matching. For example, it does not infer that `JS` and `JavaScript` mean the same thing.

## Setup

### 1. Use Python

Use Python 3.11 or newer. From this lesson's directory, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

The requirements are `google-genai` and `python-dotenv`.

### 3. Configure Gemini

Copy `.env.example` to `.env`, then set your API key. The model setting is optional; the code defaults to `gemini-2.5-flash` when `GEMINI_MODEL` is unset.

```env
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-2.5-flash
```

Use a model available to your account that supports function calling. Keep your real `.env` private and do not commit API keys.

### 4. Run the example

From the lesson directory:

```bash
python agent.py
```

The `__main__` block sends a sample request comparing Python, SQL, and Docker with a job requiring Python, SQL, Azure, and Docker. To try another request, edit `user_message` in that block and run the script again.

## Following the logs

`debug_logger.py` prints a timestamp, stage name, message, and optional data. During a run, `agent.py` logs:

- **USER INPUT** — the incoming request.
- **THINK / DECIDE** — the model, history size, and tools available for this iteration. This is an application status label; it does not expose the model's private reasoning.
- **MODEL DECISION** — whether the response contains tool calls and their names.
- **ACTION** — the requested tool and arguments.
- **OBSERVATION** — the Python tool's result.
- **FEEDBACK** — confirmation that the result was added to conversation history.
- **FINAL ANSWER** — the response returned when the model no longer requests a tool.

## Try these scenarios

Change the sample `user_message` and observe the logs:

1. **Comparison request:** include both skill lists and ask for a comparison. Gemini should have enough information to request the tool.
2. **Missing information:** omit one list. The system instruction asks Gemini to request the missing information rather than inventing skills.
3. **General question:** ask what an AI agent is. Gemini can answer without calling the tool.
4. **No overlapping skills:** compare disjoint lists; the tool should return no matches and a `0` score.
5. **All skills match:** use identical skill lists; the tool should return a `100` score (assuming the list is non-empty).

Model behavior and answer wording can vary. For reliable automated tests, test `compare_skills()` independently and mock the Gemini response when testing the agent loop.

## Troubleshooting

- **Missing or invalid API key:** check that `.env` is in the lesson directory, contains a valid `GEMINI_API_KEY`, and that the key is enabled for the Gemini API.
- **Model error:** verify `GEMINI_MODEL` is available to your account and supports function calling, or remove that setting to use the code's default.
- **Package import error:** activate the virtual environment and run `python -m pip install -r requirements.txt` again.
- **Unexpected tool name:** the executor only accepts names present in `AVAILABLE_TOOLS`.
- **Maximum iterations reached:** the model requested tools for all five iterations. Inspect the logs and tool responses; the loop is intentionally bounded.

## Current limitations

This is a teaching example, not a production-ready agent. It has one tool, no persistent memory, no retry or timeout policy, and limited validation of model-generated arguments. The tool performs exact skill matching only. Production systems should add robust input validation, error handling, tests, usage limits, secure secret management, and monitoring.

## Next lessons

Continue by adding more tools, then explore memory and state, LangGraph, MCP, and multi-agent designs. Start with the simplest design that solves the task; additional agent machinery adds complexity and should have a clear purpose.
