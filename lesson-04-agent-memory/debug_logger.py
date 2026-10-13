import json
from datetime import datetime


def debug_log(stage, message, data=None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print("\n" + "=" * 70)
    print(f"[{timestamp}] {stage}")
    print("=" * 70)
    print(message)

    if data is not None:
        if isinstance(data, (dict, list)):
            print(json.dumps(data, indent=2, default=str))
        else:
            print(data)

    print("=" * 70)


def print_box(title, body):
    lines = [str(x) for x in body.splitlines()] if isinstance(body, str) else [str(x) for x in body]
    width = max(len(title), max((len(line) for line in lines), default=0)) + 4
    print("\n╭" + "─" * width + "╮")
    print(f"│ {title:<{width - 2}} │")
    print("├" + "─" * width + "┤")
    for line in lines:
        print(f"│ {line:<{width - 2}} │")
    print("╰" + "─" * width + "╯")


def print_workflow_summary(user_message):
    print_box(
        "AGENT FLOW",
        "USER\n↓\nMEMORY\n↓\nDECIDE\n↓\nACTION\n↓\nOBSERVATION\n↓\nFINAL ANSWER",
    )
    print_box("INPUT", user_message)
