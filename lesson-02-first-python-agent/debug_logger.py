from datetime import datetime
import json


def debug_log(stage, message, data=None):

    timestamp = datetime.now().strftime("%H:%M:%S")

    print("\n" + "=" * 65)
    print(f"[{timestamp}] {stage}")
    print("=" * 65)

    print(message)

    if data is not None:

        if isinstance(data, (dict, list)):
            print(json.dumps(data, indent=2, default=str))
        else:
            print(data)

    print("=" * 65)