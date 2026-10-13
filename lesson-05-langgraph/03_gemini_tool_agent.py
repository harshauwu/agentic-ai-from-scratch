import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from workflow_debug import show_graph, show_update

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("Set GOOGLE_API_KEY in .env")
MODEL_NAME = os.getenv("GEMINI_MODEL")
if not MODEL_NAME:
    raise RuntimeError("Set GEMINI_MODEL in .env")


@tool
def compare_skills(candidate_skills: list[str], required_skills: list[str]) -> dict:
    """Compare candidate technical skills with job requirements."""
    print("\n[ACTION] Python is executing compare_skills")
    candidate = {skill.strip().lower() for skill in candidate_skills}
    required = {skill.strip().lower() for skill in required_skills}
    if not required:
        return {"error": "Job requirements cannot be empty."}

    result = {
        "matched_skills": sorted(candidate & required),
        "missing_skills": sorted(required - candidate),
        "match_percentage": round(len(candidate & required) / len(required) * 100, 2),
    }
    print("[OBSERVATION] Tool returned:", result)
    return result


llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
llm_with_tools = llm.bind_tools([compare_skills])
SYSTEM = SystemMessage(
    content=(
        "You are a job skills assistant. For skill comparisons, use compare_skills. "
        "Never invent candidate skills or job requirements. If either list is missing, "
        "ask the user to supply it. Use tool results when reporting a match percentage."
    )
)


def assistant(state: MessagesState) -> dict[str, list]:
    print(f"\n[THINK / DECIDE] Gemini receives {len(state['messages'])} message(s)")
    response = llm_with_tools.invoke([SYSTEM, *state["messages"]])
    print("[MODEL DECISION] Requested tools:", [call["name"] for call in response.tool_calls])
    return {"messages": [response]}


def route_after_assistant(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    route = "tools" if last_message.tool_calls else "end"
    print(f"[CONDITIONAL EDGE] assistant → {route}")
    return route


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode([compare_skills]))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", route_after_assistant, {"tools": "tools", "end": END})
builder.add_edge("tools", "assistant")
graph = builder.compile()


if __name__ == "__main__":
    show_graph(graph, "GEMINI TOOL AGENT: assistant → tools → assistant / END")
    question = (
        "My skills are Python, SQL and Docker. "
        "The job requires Python, SQL, Azure and Docker. Compare my skills."
    )
    print("\n[USER INPUT]", question)
    final_answer = None
    for update in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 10},
        stream_mode="updates",
    ):
        show_update(update)
        assistant_update = update.get("assistant", {})
        for message in assistant_update.get("messages", []):
            if not message.tool_calls:
                final_answer = message.content
    print("\n[FINAL ANSWER]\n", final_answer)
