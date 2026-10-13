_step_number = 0


def show_graph(_graph, flow):
    """Print the execution path in a beginner-friendly, single-line format."""
    print(f"\nFlow: START → {flow} → END")


def reset_steps():
    global _step_number
    _step_number = 0


def show_update(update):
    """Print only which node ran and the values it returned."""
    global _step_number
    for node_name, state_update in update.items():
        _step_number += 1
        print(f"Step {_step_number}: node '{node_name}' ran")
        for key, value in state_update.items():
            print(f"  {key} = {value}")
