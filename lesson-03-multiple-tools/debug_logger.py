import json
from datetime import datetime


def format_box(title, lines):
    text_lines = [str(line) for line in lines]
    width = max(len(title), max((len(line) for line in text_lines), default=0)) + 2
    top = "╭" + "─" * width + "╮"
    bottom = "╰" + "─" * width + "╯"
    print(top)
    print(f"│ {title:<{width - 2}} │")
    print("├" + "─" * width + "┤")
    for line in text_lines:
        padded = line[: width - 4].ljust(width - 4)
        print(f"│ {padded} │")
    print(bottom)


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


def print_workflow_stage(stage, message, details=None):
    lines = [message]
    if details:
        if isinstance(details, (dict, list)):
            payload = json.dumps(details, indent=2, default=str)
            lines.extend(payload.splitlines())
        else:
            lines.extend(str(details).splitlines())
    format_box(stage, lines)


def print_workflow_summary(user_message):
    flow = [
        "USER REQUEST",
        "↓",
        "THINK / DECIDE",
        "↓",
        "TOOL ACTION",
        "↓",
        "OBSERVATION",
        "↓",
        "NEXT DECISION",
        "↓",
        "FINAL ANSWER",
    ]
    print("\n")
    format_box("AGENT WORKFLOW", flow)
    print_workflow_stage("INPUT", user_message)
