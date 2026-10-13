import json


def show_graph(graph, title):
    """Print Mermaid source that describes the compiled graph."""
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print("Paste this Mermaid text into https://mermaid.live to render a diagram:\n")
    print(graph.get_graph().draw_mermaid())
    print("=" * 72)


def show_update(update):
    """Display the node and state update produced by a streamed graph step."""
    for node_name, state_update in update.items():
        print(f"\n┌── NODE: {node_name} ──")
        try:
            print(json.dumps(state_update, indent=2, default=str, ensure_ascii=False))
        except (TypeError, ValueError):
            print(state_update)
        print("└" + "─" * 40)
