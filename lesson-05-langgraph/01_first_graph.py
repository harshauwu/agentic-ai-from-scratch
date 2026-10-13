from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from workflow_debug import show_graph, show_update


class State(TypedDict):
    message: str


def greet(state: State) -> dict[str, str]:
    print("[NODE ACTION] greet reads the current message")
    return {"message": state["message"] + " I am"}


def finish(state: State) -> dict[str, str]:
    print("[NODE ACTION] finish adds the final phrase")
    return {"message": state["message"] + " learning LangGraph!"}


builder = StateGraph(State)
builder.add_node("greet", greet)
builder.add_node("finish", finish)
builder.add_edge(START, "greet")
builder.add_edge("greet", "finish")
builder.add_edge("finish", END)
graph = builder.compile()


if __name__ == "__main__":
    show_graph(graph, "FIRST GRAPH: START → greet → finish → END")
    initial_state = {"message": "Hello."}
    print("\n[INPUT STATE]", initial_state)
    final_state = None
    for update in graph.stream(initial_state, stream_mode="updates"):
        show_update(update)
        for values in update.values():
            if "message" in values:
                final_state = values["message"]
    print("\n[FINAL]", {"message": final_state})
