import json
from dataclasses import dataclass, field
from pathlib import Path

from google.genai import types


@dataclass
class AgentState:
    history: list = field(default_factory=list)
    transcript: list[dict[str, str]] = field(default_factory=list)

    def add_user(self, message: str):
        self.history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=message)])
        )
        self.transcript.append({"role": "user", "text": message})

    def add_assistant_text(self, message: str):
        self.transcript.append({"role": "model", "text": message})

    def save(self, filename: str = "conversation.json"):
        Path(filename).write_text(
            json.dumps(self.transcript, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, filename: str = "conversation.json"):
        path = Path(filename)
        if not path.exists():
            return cls()

        transcript = json.loads(path.read_text(encoding="utf-8"))
        state = cls()
        for entry in transcript:
            role = entry["role"]
            text = entry["text"]
            if role not in ("user", "model") or not isinstance(text, str):
                raise ValueError("Invalid saved conversation entry")
            state.history.append(
                types.Content(role=role, parts=[types.Part.from_text(text=text)])
            )
            state.transcript.append({"role": role, "text": text})
        return state
