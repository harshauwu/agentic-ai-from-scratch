from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from workflow_debug import reset_steps, show_graph, show_update


class State(TypedDict):
    message: str


def greet(state: State) -> dict[str, str]:
    return {"message": state["message"] + " I am"}


def finish(state: State) -> dict[str, str]:
    return {"message": state["message"] + " learning LangGraph!"}


builder = StateGraph(State)
builder.add_node("greet", greet)
builder.add_node("finish", finish)
builder.add_edge(START, "greet")
builder.add_edge("greet", "finish")
builder.add_edge("finish", END)
graph = builder.compile()


if __name__ == "__main__":
    show_graph(graph, "greet → finish")
    initial_state = {"message": "Hello."}
    print("Input:", initial_state)
    final_state = None
    reset_steps()
    for update in graph.stream(initial_state, stream_mode="updates"):
        show_update(update)
        for values in update.values():
            if "message" in values:
                final_state = values["message"]
    print("Final:", {"message": final_state})
