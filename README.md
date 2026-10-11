
# AI Agent Development from Scratch

### Learn Agentic AI with Python, Google Gemini, and Custom Tools

A beginner-friendly, hands-on tutorial for understanding and developing AI agents **without using agent frameworks such as LangChain or LangGraph**.

The goal of this project is to understand how AI agents work internally before learning agent frameworks.

---

# 📚 Table of Contents

1. Introduction to Agentic AI
2. What Is an AI Agent?
3. LLM vs AI Agent
4. Understanding the Agentic Workflow
5. Project Overview
6. System Architecture
7. Project Setup
8. Building Our First Tool
9. Connecting Google Gemini
10. Registering Tools
11. Building the Agent Loop
12. Understanding the Complete Workflow
13. Debugging the Agent
14. Testing Different Scenarios
15. Common Beginner Questions
16. Interview Questions and Answers
17. Future Improvements
18. Learning Roadmap

---

# 1. Introduction to Agentic AI

## What is Agentic AI?

Agentic AI refers to AI systems that can pursue goals by making decisions, using tools, observing results, and deciding what to do next.

Unlike a basic chatbot, an AI agent can interact with external systems and perform actions.

### Example

Imagine you are developing a job application system.

A user asks:

> I know Python, SQL, and Docker. A job requires Python, SQL, Azure, and Docker. Can you compare my skills with the job requirements?

A basic LLM can answer this question using its language capabilities.

However, an AI agent can decide to use a dedicated Python function to calculate the comparison.

The workflow becomes:

```text
User Request
     |
     v
AI Agent
     |
     v
LLM decides what to do
     |
     v
Call compare_skills()
     |
     v
Execute Python function
     |
     v
Receive comparison results
     |
     v
LLM generates final answer
```

This is a simple example of an AI agent using a tool.

---

# 2. What Is an AI Agent?

An AI agent is a software system that uses an AI model to make decisions and take actions toward a goal.

A typical LLM-based agent contains several components.

| Component | Responsibility |
|---|---|
| LLM | Interprets requests and selects actions |
| Instructions | Define the agent's responsibilities and constraints |
| Tools | Perform specific operations |
| Agent Loop | Coordinates decisions and actions |
| Observations | Provide feedback from executed actions |
| Memory/State | Maintains relevant context |
| Final Response | Communicates the outcome |

Not every agent needs all these components.

For example, our first agent uses conversation history as short-term state but does not have a persistent memory database.

## Important concept

**The LLM is not the entire agent.**

Our agent consists of:

```text
AI Agent
   |
   |--- LLM (Google Gemini)
   |
   |--- Python application
   |
   |--- Custom tools
   |
   |--- Agent loop
   |
   |--- Conversation history
```

The LLM helps decide what action to take.

The Python application controls the workflow and executes the actions.

---

# 3. LLM vs AI Agent

Understanding this difference is important for technical interviews.

## Large Language Model

A Large Language Model generates responses based on the input and context it receives.

Example:

```text
User:
What is Python?

LLM:
Python is a high-level programming language.
```

No external tools are required.

## AI Agent

An AI agent can use an LLM to select and execute actions through tools.

Example:

```text
User:
Compare my skills with a job.

Agent:
1. Understands the request.
2. Selects compare_skills.
3. Generates tool arguments.
4. Executes the Python function.
5. Receives the result.
6. Generates the final answer.
```

### Comparison

| Feature | Basic LLM Application | Tool-Using AI Agent |
|---|---|---|
| Generate text | Yes | Yes |
| Select tools | Not necessarily | Yes |
| Execute external actions | Not by itself | Through application code |
| Use observations | Not necessarily | Yes |
| Multi-step workflow | Requires orchestration | Supported by agent loop |
| Autonomous decisions | Limited to application design | Within configured boundaries |

**Interview tip:**

An agent is not simply an LLM with a prompt. It combines an LLM with orchestration logic, tools, state, and feedback to complete tasks.

---

# 4. Understanding the Agentic Workflow

One of the most important concepts in agent development is:

## Thought → Action → Observation

We will use these terms to describe the agent's workflow.

### Step 1: Thought / Decision

The agent sends the user's request and available tool descriptions to the LLM.

The LLM decides whether a tool is needed.

Example:

```text
User request:
Compare my skills with the job.

Available tool:
compare_skills
```

The model may decide to request `compare_skills`.

**Important:** The decision is observable through the model's response. The model's private internal reasoning is not necessarily visible.

### Step 2: Action

The LLM requests a tool call.

Example:

```json
{
  "name": "compare_skills",
  "arguments": {
    "candidate_skills": [
      "Python",
      "SQL",
      "Docker"
    ],
    "required_skills": [
      "Python",
      "SQL",
      "Azure",
      "Docker"
    ]
  }
}
```

The Python application receives this request and executes the corresponding function.

### Step 3: Observation

The tool returns its result.

Example:

```json
{
  "matched_skills": [
    "Python",
    "SQL",
    "Docker"
  ],
  "missing_skills": [
    "Azure"
  ],
  "match_percentage": 75.0
}
```

This is the observation.

### Step 4: Decision Again

The application sends the tool result back to the LLM.

The LLM can now decide whether to:

- Produce the final answer.
- Call another tool.
- Call the same tool again.

### Step 5: Final Answer

When the model returns a normal response without requesting additional tools, the agent finishes.

Example:

```text
Your skills match 75% of the job requirements.

Matched:
- Python
- SQL
- Docker

Missing:
- Azure

Recommendation:
Focus on learning Microsoft Azure.
```

## Complete workflow

```text
               START
                 |
                 v
            USER REQUEST
                 |
                 v
          PYTHON APPLICATION
                 |
                 v
          THINK / DECIDE
                 |
                 v
             GEMINI LLM
                 |
                 v
          Tool required?
             /      \
           YES       NO
            |         |
            v         v
          ACTION   FINAL ANSWER
            |         |
            v         v
       Execute Tool   END
            |
            v
        OBSERVATION
            |
            v
     Add result to history
            |
            |
            +----> THINK / DECIDE
```

This repeating process is called the **Agent Loop**.

---

# 5. Project Overview

## Project: Job Skills Comparison Agent

We will develop an AI agent that compares a candidate's technical skills against job requirements.

### User input

```text
My skills are Python, SQL, and Docker.

The job requires Python, SQL, Azure, and Docker.

Compare my skills with the job.
```

### Expected output

```text
Matched skills:
Python, SQL, Docker

Missing skills:
Azure

Match percentage:
75%
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11+ | Application development |
| Google Gemini | LLM |
| Google Gen AI SDK | Gemini API integration |
| python-dotenv | Environment configuration |
| Custom Python Functions | Agent tools |
| VS Code | Development and debugging |
| GitHub | Version control |

We will not use an agent framework.

This allows us to understand the underlying mechanics of agent development.

---

# 6. System Architecture

Our project has three main components.

```text
+--------------------------+
|           USER           |
+--------------------------+
             |
             v
+--------------------------+
|         agent.py         |
|                          |
| - Agent instructions     |
| - Tool declarations      |
| - Tool execution         |
| - Agent loop             |
| - Conversation history   |
+--------------------------+
             |
             v
+--------------------------+
|       GOOGLE GEMINI      |
|                          |
| - Understand request     |
| - Select tool            |
| - Generate arguments     |
| - Generate final answer  |
+--------------------------+
             |
             v
+--------------------------+
|         tools.py         |
|                          |
| - compare_skills()       |
+--------------------------+
```

### Responsibility separation

**agent.py**

Controls the agent workflow.

**tools.py**

Contains the actual business logic.

**Google Gemini**

Provides language understanding and tool-selection capabilities.

This separation is useful because the tools can be tested independently of the LLM.

---

# 7. Project Setup

## Step 1: Create a project directory

```bash
mkdir ai-agent-from-scratch

cd ai-agent-from-scratch
```

## Step 2: Create a Python environment

Using Conda:

```bash
conda create -n ai-agent python=3.11

conda activate ai-agent
```

Alternatively, use Python's built-in virtual environment.

## Step 3: Install dependencies

```bash
pip install google-genai python-dotenv
```

## Step 4: Create the project structure

```text
ai-agent-from-scratch/
│
├── agent.py
│
├── tools.py
│
├── debug_logger.py
│
├── .env
│
├── .env.example
│
├── .gitignore
│
├── requirements.txt
│
└── README.md
```

### File responsibilities

| File | Description |
|---|---|
| agent.py | Main agent implementation |
| tools.py | Custom Python tools |
| debug_logger.py | Terminal workflow logging |
| .env | Local API credentials |
| .env.example | Example configuration |
| .gitignore | Prevents sensitive files from being committed |
| requirements.txt | Python dependencies |
| README.md | Project tutorial |

## Step 5: Configure the API key

Create a Gemini API key using Google AI Studio.

Official website:

Google AI Studio — aistudio.google.com

Add the key to `.env`:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=your_available_gemini_model
```

Select a model that supports function calling and is available to your account.

Create `.env.example`:

```env
GEMINI_API_KEY=replace_with_your_key
GEMINI_MODEL=replace_with_available_model
```

## Step 6: Create `.gitignore`

```gitignore
.env
.venv/
__pycache__/
*.pyc
.DS_Store
```

Never upload your real API key to GitHub.

---

# 8. Building Our First Tool

Before building the agent, we need a Python function.

## What is a tool?

A tool is a function or external capability that an agent can request.

Examples include:

- Calculating skill matches.
- Searching a database.
- Reading a file.
- Calling a weather API.
- Searching job listings.
- Creating GitHub issues.

Our first tool will compare two lists of skills.

## Create `tools.py`

```python
def compare_skills(candidate_skills, required_skills):

    # Normalize skills for case-insensitive comparison
    candidate = {
        skill.strip().lower()
        for skill in candidate_skills
    }

    required = {
        skill.strip().lower()
        for skill in required_skills
    }

    # Find matching skills
    matched = sorted(candidate.intersection(required))

    # Find missing skills
    missing = sorted(required.difference(candidate))

    # Calculate match percentage
    if required:
        percentage = round(
            len(matched) / len(required) * 100,
            2
        )
    else:
        percentage = 0.0

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "match_percentage": percentage
    }
```

## Understanding the code

### Function parameters

```python
def compare_skills(candidate_skills, required_skills):
```

The function accepts two lists.

Example:

```python
candidate_skills = ["Python", "SQL", "Docker"]

required_skills = [
    "Python",
    "SQL",
    "Azure",
    "Docker"
]
```

### Normalization

```python
skill.strip().lower()
```

This removes surrounding spaces and converts skills to lowercase.

For example:

```text
" Python " → "python"
```

This helps avoid simple matching errors.

### Find matching skills

```python
candidate.intersection(required)
```

This finds skills present in both sets.

### Find missing skills

```python
required.difference(candidate)
```

This identifies required skills the candidate does not have.

### Calculate match percentage

```python
len(matched) / len(required) * 100
```

Example:

```text
Matched skills = 3

Required skills = 4

Match percentage = 3 / 4 × 100

Result = 75%
```

This is a basic exact-match comparison, not a semantic assessment of equivalent skills.

## Important interview concept

**The tool is a normal Python function.**

It does not need to contain AI.

The LLM decides when to request the tool, and Python executes it.

---

# 9. Connecting Google Gemini

Now we will connect our application to an LLM.

## Create `agent.py`

Start with:

```python
import os

from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing")

client = genai.Client(
    api_key=api_key
)

MODEL = os.environ["GEMINI_MODEL"]
```

## Explanation

### load_dotenv()

```python
load_dotenv(override=True)
```

Loads environment variables from `.env`.

### genai.Client()

```python
client = genai.Client(api_key=api_key)
```

Creates a client for communicating with Google's Gemini API.

### MODEL

```python
MODEL = os.environ["GEMINI_MODEL"]
```

Reads the selected model from the environment.

This allows us to change models without modifying the agent implementation.

---

# 10. Registering Tools

Creating a Python function is not enough.

The LLM must know that the function exists.

We provide a **tool declaration**.

Add the following imports to `agent.py`:

```python
from google.genai import types

from tools import compare_skills
```

## Tool declaration

```python
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
                "items": {
                    "type": "STRING"
                }
            },
            "required_skills": {
                "type": "ARRAY",
                "items": {
                    "type": "STRING"
                }
            }
        },
        "required": [
            "candidate_skills",
            "required_skills"
        ]
    }
)
```

## Why do we need a tool declaration?

Gemini does not automatically know about our Python functions.

The declaration tells Gemini:

1. The tool's name.
2. What the tool does.
3. Which parameters it accepts.
4. Which parameters are required.
5. The expected parameter types.

### Tool declaration vs tool implementation

| Tool Declaration | Tool Implementation |
|---|---|
| Describes the function | Executes the actual logic |
| Sent to Gemini | Runs in Python |
| Helps the LLM choose tools | Produces results |

## Tool registry

Next, register the function:

```python
AVAILABLE_TOOLS = {
    "compare_skills": compare_skills
}
```

The registry maps a tool name to a Python function.

## Tool executor

```python
def execute_tool(tool_name, arguments):

    if tool_name not in AVAILABLE_TOOLS:
        raise ValueError(
            f"Unknown tool: {tool_name}"
        )

    function = AVAILABLE_TOOLS[tool_name]

    return function(**arguments)
```

### Why is this necessary?

The LLM may request:

```text
compare_skills
```

But the LLM does not execute the function.

Our Python application:

1. Reads the requested tool name.
2. Checks whether the tool is permitted.
3. Finds the corresponding function.
4. Executes the function with the supplied arguments.
5. Returns the result.

**Security principle:** Never allow the LLM to execute arbitrary Python code. Only execute explicitly registered and validated tools.

---

# 11. Building the Agent Loop

This is the most important part of the tutorial.

An agent loop repeatedly:

1. Sends context to the LLM.
2. Reads the model's decision.
3. Executes requested tools.
4. Collects observations.
5. Sends observations back to the LLM.
6. Stops when a final answer is returned.

## Configure Gemini

Add this function to `agent.py`:

```python
def create_agent_config():

    return types.GenerateContentConfig(
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
```

## Why disable automatic function calling?

The Google SDK can automatically handle certain Python function calls.

However, we want to learn how the agent works internally.

Therefore:

```python
disable=True
```

allows us to manually inspect and execute function calls.

## Implement `run_agent()`

```python
def run_agent(user_message):

    history = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=user_message
                )
            ]
        )
    ]

    config = create_agent_config()

    for step in range(5):

        print(f"\n--- Agent step {step + 1} ---")

        # THINK / DECIDE
        print("[THINK / DECIDE] Asking Gemini...")

        response = client.models.generate_content(
            model=MODEL,
            contents=history,
            config=config
        )

        if not response.candidates:
            raise RuntimeError(
                "Gemini returned no response candidates."
            )

        assistant_content = response.candidates[0].content

        if assistant_content is None:
            raise RuntimeError(
                "Gemini returned no content."
            )

        # Preserve the model response
        history.append(assistant_content)

        # Extract tool calls
        function_calls = [
            part.function_call
            for part in (assistant_content.parts or [])
            if part.function_call is not None
        ]

        # FINAL ANSWER
        if not function_calls:

            final_answer = response.text or ""

            print("\n[FINAL ANSWER]")
            print(final_answer)

            return final_answer

        tool_response_parts = []

        for call in function_calls:

            tool_name = call.name
            arguments = dict(call.args or {})

            # ACTION
            print("\n[ACTION]")
            print("Tool:", tool_name)
            print("Arguments:", arguments)

            result = execute_tool(
                tool_name,
                arguments
            )

            # OBSERVATION
            print("\n[OBSERVATION]")
            print(result)

            tool_response_parts.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={
                        "result": result
                    }
                )
            )

        # FEEDBACK
        history.append(
            types.Content(
                role="user",
                parts=tool_response_parts
            )
        )

        print("\n[FEEDBACK]")
        print("Sending observations back to Gemini.")

    raise RuntimeError(
        "Maximum agent iterations reached."
    )
```

## Start the agent

Add this to the bottom of `agent.py`:

```python
if __name__ == "__main__":

    user_message = (
        "My skills are Python, SQL and Docker. "
        "The job requires Python, SQL, Azure and Docker. "
        "Compare my skills with the job."
    )

    run_agent(user_message)
```

## Why use `range(5)`?

```python
for step in range(5):
```

This limits the agent to five LLM decision cycles.

Without a limit, a poorly designed agent could repeatedly request tools and consume unnecessary API tokens.

In production systems, we should also consider:

- Token limits.
- Timeouts.
- Error handling.
- API usage costs.
- Tool execution limits.

---

# 12. Understanding the Complete Workflow

Let's examine what happens when the agent runs.

## User request

```text
My skills are Python, SQL and Docker.

The job requires Python, SQL, Azure and Docker.

Compare my skills with the job.
```

## Iteration 1: Decision

Python sends Gemini:

```text
System instructions

User request

Available tool:
compare_skills
```

Gemini may respond with a structured function call.

Example:

```json
{
  "name": "compare_skills",
  "args": {
    "candidate_skills": [
      "Python",
      "SQL",
      "Docker"
    ],
    "required_skills": [
      "Python",
      "SQL",
      "Azure",
      "Docker"
    ]
  }
}
```

## Iteration 1: Action

Python executes:

```python
compare_skills(
    candidate_skills=[
        "Python",
        "SQL",
        "Docker"
    ],
    required_skills=[
        "Python",
        "SQL",
        "Azure",
        "Docker"
    ]
)
```

## Iteration 1: Observation

The function returns:

```json
{
  "matched_skills": [
    "docker",
    "python",
    "sql"
  ],
  "missing_skills": [
    "azure"
  ],
  "match_percentage": 75.0
}
```

## Iteration 2: Decision

Python sends the observation back to Gemini.

Gemini now has the calculated comparison.

If no further tool is needed, it generates a final response.

## Iteration 2: Final Answer

```text
You match 75% of the job requirements.

Matched skills:
- Python
- SQL
- Docker

Missing skill:
- Azure
```

The exact response wording is generated by the LLM and may vary.

---

# 13. Debugging the Agent

Understanding the agent workflow is easier when we can see each step in the terminal.

## Create `debug_logger.py`

```python
from datetime import datetime
import json


def debug_log(stage, message, data=None):

    timestamp = datetime.now().strftime("%H:%M:%S")

    print("\n" + "=" * 60)
    print(f"[{timestamp}] {stage}")
    print("=" * 60)

    print(message)

    if data is not None:
        print(
            json.dumps(
                data,
                indent=2,
                default=str
            )
        )
```

## Use the logger

Import it into `agent.py`:

```python
from debug_logger import debug_log
```

For example, before calling Gemini:

```python
debug_log(
    "THINK / DECIDE",
    "Sending request to Gemini",
    {
        "model": MODEL,
        "history_messages": len(history)
    }
)
```

Before executing a tool:

```python
debug_log(
    "ACTION",
    "Executing tool",
    {
        "tool": tool_name,
        "arguments": arguments
    }
)
```

After executing the tool:

```python
debug_log(
    "OBSERVATION",
    "Tool execution completed",
    result
)
```

When the agent finishes:

```python
debug_log(
    "FINAL ANSWER",
    "Agent completed",
    final_answer
)
```

These logging calls can be added to the corresponding sections of `run_agent()`.

## Example terminal output

```text
============================================================
[10:30:01] THINK / DECIDE
============================================================

Sending request to Gemini

Model: Gemini Flash

============================================================
[10:30:03] ACTION
============================================================

Tool: compare_skills

Arguments:
candidate_skills = ["Python", "SQL", "Docker"]
required_skills = ["Python", "SQL", "Azure", "Docker"]

============================================================
[10:30:03] OBSERVATION
============================================================

matched_skills = ["python", "sql", "docker"]
missing_skills = ["azure"]
match_percentage = 75.0

============================================================
[10:30:05] FINAL ANSWER
============================================================

Your skills match 75% of the job requirements.
```

This output is illustrative.

### Important clarification

The THINK / DECIDE stage shows when the application requests a decision from Gemini.

It does not reveal Gemini's private chain of thought.

---

# 14. Testing Different Scenarios

A good AI engineer tests more than the successful scenario.

## Test 1: Complete information

Input:

```text
My skills are Python and SQL.

The job requires Python, SQL and Azure.

Compare my skills.
```

Expected behavior:

- Agent selects `compare_skills`.
- Python executes the tool.
- Gemini receives the result.
- Agent generates a final answer.

## Test 2: Missing job requirements

Input:

```text
I know Python, SQL and Docker.

Can you compare my skills with a job?
```

Expected behavior:

- Gemini identifies missing information.
- Gemini asks for the job requirements.
- The tool should not be called with invented requirements.

## Test 3: General question

Input:

```text
What is an AI agent?
```

Expected behavior:

- Gemini answers directly.
- No tool call is necessary.

## Test 4: No matching skills

Input:

```text
My skills are Java and Spring Boot.

The job requires Python and Azure.

Compare my skills.
```

Expected result:

```text
Match percentage: 0%
```

## Test 5: All skills match

Input:

```text
My skills are Python, SQL and Docker.

The job requires Python, SQL and Docker.

Compare my skills.
```

Expected result:

```text
Match percentage: 100%
```

## What should we validate?

| Test | Expected |
|---|---|
| Complete input | Tool called |
| Missing information | Clarification requested |
| General question | Direct answer |
| No matches | 0% |
| All matches | 100% |
| Unknown tool | Rejected |
| Maximum iterations | Agent stops |

For reliable testing, we should test the Python tool independently and use mocked LLM responses to test the agent workflow.

---

# 15. Common Beginner Questions

## Q1. Is Gemini the AI agent?

No.

Gemini is the LLM used by the agent.

The complete agent also includes Python orchestration, tools, and conversation history.

## Q2. Does Gemini execute Python functions?

No, not directly.

Gemini generates a structured tool-call request.

Our Python application executes the function.

Some SDKs provide automatic function-calling features that handle this process for developers.

## Q3. What is tool calling?

Tool calling is a mechanism that allows an LLM to request an operation using a defined interface.

Example:

```text
LLM requests:
compare_skills(...)
```

The application then executes the requested function.

## Q4. What is an observation?

An observation is feedback received after an action.

Examples:

- A Python function result.
- A database query result.
- An API response.
- A tool execution error.

## Q5. What is the agent loop?

The agent loop is the control structure that repeatedly coordinates LLM decisions and tool execution.

```text
Decision
   |
Action
   |
Observation
   |
Decision
   |
Final Answer
```

## Q6. Why do we need conversation history?

The model needs relevant previous context to understand the results of earlier actions.

For example:

```text
User request
     |
Tool call
     |
Tool result
     |
Final response
```

Without the tool result being returned to the model, Gemini may not know what the Python function calculated.

## Q7. Is this a single-agent system?

Yes.

We have one LLM-driven agent with one available tool.

Later, we can extend it to multiple tools or multiple specialized agents.

## Q8. Do we need LangGraph?

No.

An agent can be developed using ordinary Python and an LLM SDK.

Frameworks such as LangGraph help developers manage more complex workflows, state, branching, and execution.

## Q9. Is every LLM application an AI agent?

No.

A basic chatbot that only generates text is not necessarily an agent.

Agentic behavior generally involves goal-directed decisions and actions, often through tools and iterative feedback.

The distinction depends on the system's actual behavior, not simply whether it uses an LLM.

---

# 16. AI Engineer Interview Questions and Answers

## Question 1: What is Agentic AI?

**Answer:**

Agentic AI refers to AI systems that can make decisions, take actions, use tools, and respond to feedback to accomplish goals.

For example, I developed a job skills comparison agent using Python and Google Gemini. The agent selects a Python tool, executes it, receives the result, and generates a final response.

## Question 2: Explain the Thought → Action → Observation workflow.

**Answer:**

The workflow consists of three main stages.

**Thought or Decision:** The LLM determines the next action based on the user request and available context.

**Action:** The application executes the requested tool.

**Observation:** The tool result is returned to the LLM.

This process can repeat until the agent produces a final answer.

## Question 3: What is function calling?

**Answer:**

Function calling allows an LLM to request predefined functions using structured arguments.

The LLM does not directly execute the function. The application validates the request, executes the function, and returns the result.

## Question 4: How did you develop your AI agent?

**Answer:**

I developed a Python-based AI agent using the Google Gemini SDK without an agent framework.

First, I implemented a custom skill comparison function.

Next, I defined a function declaration so Gemini could understand the available tool.

Then, I created a tool registry and an agent loop.

The agent sends the user request to Gemini, receives a function-call request, executes the Python tool, and returns the observation to Gemini.

Finally, Gemini generates the response.

I also implemented debugging logs to observe the decision, action, observation, and final-answer stages.

## Question 5: Why did you avoid using LangGraph initially?

**Answer:**

I wanted to understand the underlying mechanics of AI agents before using a framework.

Building the agent manually helped me understand tool declarations, function calling, conversation history, observations, and agent-loop control.

This foundational knowledge makes it easier to understand what frameworks automate.

## Question 6: What is the difference between an AI workflow and an AI agent?

**Answer:**

An AI workflow usually follows predefined steps.

For example:

```text
Extract CV
    |
Compare Skills
    |
Generate Report
```

An AI agent allows an LLM to make certain decisions dynamically.

For example:

```text
User Request
     |
LLM decides next action
     |
Selects an appropriate tool
     |
Receives feedback
     |
Decides what to do next
```

Many production systems combine deterministic workflows with agentic components.

## Question 7: How do you prevent an AI agent from running indefinitely?

**Answer:**

I implemented a maximum iteration limit.

```python
for step in range(5):
```

This prevents unlimited agent execution.

In a production environment, I would also implement execution timeouts, token budgets, error handling, and tool-call limits.

## Question 8: What are the limitations of your current agent?

**Answer:**

The current implementation is a learning prototype.

Its limitations include:

- Only one custom tool.
- No persistent memory.
- Basic exact-match skill comparison.
- Limited error handling.
- No production authentication.
- No automated agent evaluation.
- No database integration.

Future improvements would address these limitations.

## Question 9: What is the difference between tool calling and MCP?

**Answer:**

Tool calling allows an LLM to request a tool through an application-defined interface.

Model Context Protocol (MCP) provides a standardized way for AI applications to discover and interact with tools and external systems.

MCP is not required to implement basic tool calling.

## Question 10: How would you improve this agent for production?

**Answer:**

I would introduce:

1. Input and tool-argument validation.
2. Structured logging.
3. Tool execution timeouts.
4. Error handling and retries.
5. Persistent state when needed.
6. Automated tests and evaluations.
7. Monitoring of token usage and latency.
8. Security controls for external actions.

---

# 17. Future Improvements

Our current project is intentionally simple.

We can gradually extend it.

## Improvement 1: Multiple Tools

Add:

```python
compare_skills()
```

```python
calculate_experience_match()
```

```python
generate_skill_gap_report()
```

The LLM can select tools based on the user's request.

## Improvement 2: Conversation Memory

Allow the agent to retain relevant information across user messages.

Example:

```text
User:
My skills are Python and SQL.

User:
Now compare them with a job requiring Python and Azure.
```

The agent should be able to use the earlier skills list.

## Improvement 3: Database Integration

Store:

- Candidate profiles.
- Technical skills.
- Job requirements.
- Comparison results.

A database enables the application to retrieve and update persistent information.

## Improvement 4: CV Analysis

Allow users to upload a CV.

The system can extract structured information such as:

```json
{
  "name": "Candidate Name",
  "skills": [
    "Python",
    "SQL",
    "Docker"
  ],
  "experience": [],
  "certifications": []
}
```

A structured extraction pipeline may be more appropriate than an autonomous agent for this particular task.

## Improvement 5: LangGraph

Rebuild the agent using LangGraph.

Compare:

```text
Pure Python Agent
```

with:

```text
LangGraph Agent
```

Understand how the framework handles state, nodes, edges, and execution.

## Improvement 6: MCP Integration

Connect the agent to external tools through MCP.

Potential integrations include:

- GitHub.
- Databases.
- File systems.
- Approved external APIs.

## Improvement 7: Multi-Agent Architecture

Develop specialized agents:

```text
             USER
               |
               v
        ORCHESTRATOR
               |
      +--------+--------+
      |        |        |
      v        v        v
   CV Agent  Job Agent  Match Agent
      |        |        |
      +--------+--------+
               |
               v
          FINAL REPORT
```

Each agent has a specific responsibility.

However, multiple agents should only be introduced when the additional complexity is justified.

---

# 18. Learning Roadmap

This repository is the starting point for a larger Agentic AI learning journey.

| Lesson | Topic | Goal |
|---|---|---|
| 01 | AI Agent Fundamentals | Understand agent components |
| 02 | First Python Agent | Build an agent without frameworks |
| 03 | Tool Calling | Understand function declarations |
| 04 | Agent Loop | Implement decision-action-observation |
| 05 | Debugging | Inspect agent execution |
| 06 | Multiple Tools | Allow dynamic tool selection |
| 07 | Memory and State | Maintain conversation context |
| 08 | Error Handling | Build reliable execution |
| 09 | LangGraph | Learn framework-based agents |
| 10 | MCP | Connect external tools |
| 11 | Multi-Agent Systems | Coordinate specialized agents |
| 12 | Production AI Agents | Testing, security, monitoring, deployment |

---

# 🚀 Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file using `.env.example` as a reference.

Configure a valid Gemini API key and an available model that supports function calling.

Run:

```bash
python agent.py
```

The terminal will display the agent's workflow.

---

# 📦 Requirements

Create `requirements.txt`:

```text
google-genai
python-dotenv
```

For a reproducible release, pin tested dependency versions.

---

# 🎯 Key Takeaways

After completing this tutorial, you should understand:

1. What an AI agent is.
2. The difference between an LLM and an agent.
3. How tool calling works.
4. Why tool declarations are necessary.
5. How Python executes agent tools.
6. What observations are.
7. How an agent loop works.
8. How conversation history is maintained.
9. How to debug an agent.
10. Why agent frameworks are optional for simple systems.

## Final Concept

```text
AI AGENT
   |
   v
LLM DECISION
   |
   v
TOOL ACTION
   |
   v
OBSERVATION
   |
   v
NEXT DECISION
   |
   v
FINAL ANSWER
```

**The LLM provides decision-making capabilities, while the application provides controlled execution.**

Understanding this distinction is the foundation of developing reliable AI agents.

---

## References

- Google Gemini API Documentation — ai.google.dev/gemini-api/docs
- Google Gen AI Python SDK — googleapis.github.io/python-genai/
- Hugging Face Agents Course — huggingface.co/learn/agents-course
- LangGraph Documentation — docs.langchain.com/oss/python/langgraph/overview
- Model Context Protocol — modelcontextprotocol.io

---

**Project:** AI Agent Development from Scratch  
**Language:** Python  
**LLM:** Google Gemini  
**Framework:** None  
**Learning Level:** Beginner → Intermediate