import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from workflow_debug import show_graph

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("Set GOOGLE_API_KEY in .env")
MODEL_NAME = os.getenv("GEMINI_MODEL")
if not MODEL_NAME:
    raise RuntimeError("Set GEMINI_MODEL in .env")


@tool
def compare_skills(candidate_skills: list[str], required_skills: list[str]) -> dict:
    """Compare candidate technical skills with job requirements."""
    print("  Tool compare_skills: received candidate and required skill lists")
    candidate = {skill.strip().lower() for skill in candidate_skills}
    required = {skill.strip().lower() for skill in required_skills}
    if not required:
        print("  Tool result: job requirements are empty")
        return {"error": "Job requirements cannot be empty."}

    result = {
        "matched_skills": sorted(candidate & required),
        "missing_skills": sorted(required - candidate),
        "match_percentage": round(len(candidate & required) / len(required) * 100, 2),
    }
    print(f"  Tool result: {result}")
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
    print("\nNode: assistant")
    print(f"  Passing {len(state['messages'])} message(s) to Gemini")
    print("  Calling LLM...")
    response = llm_with_tools.invoke([SYSTEM, *state["messages"]])
    if response.tool_calls:
        print("  LLM returned a tool request:")
        for call in response.tool_calls:
            print(f"    {call['name']} with arguments {call['args']}")
    else:
        print("  LLM returned a final text response")
    return {"messages": [response]}


def route_after_assistant(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        print("\nEdge condition: Did Gemini request a tool? YES")
        print("  Route to next node: tools")
        print("  Node: tools will execute the requested function(s)")
        return "tools"

    print("\nEdge condition: Did Gemini request a tool? NO")
    print("  Route to next node: END")
    return "end"


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode([compare_skills]))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", route_after_assistant, {"tools": "tools", "end": END})
builder.add_edge("tools", "assistant")
graph = builder.compile()


if __name__ == "__main__":
    show_graph(graph, "assistant → (tools → assistant OR END)")
    question = (
        "My skills are Python, SQL and Docker. "
        "The job requires Python, SQL, Azure and Docker. Compare my skills."
    )
    print("\nSTART")
    print("Input message:", question)
    print("\nEdge: START → assistant")
    final_answer = None
    for update in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 10},
        stream_mode="updates",
    ):
        for node_name in update:
            print(f"\nNode completed: {node_name}")
            if node_name == "tools":
                print("Edge: tools → assistant (send tool result back to Gemini)")
        assistant_update = update.get("assistant", {})
        for message in assistant_update.get("messages", []):
            if not message.tool_calls:
                final_answer = message.content
    print("\nEND")
    print("Final answer:", final_answer)
