from __future__ import annotations

from typing import Literal, TypedDict

Mode = Literal["offline", "online", "auto"]
UsedMode = Literal["offline", "online", "offline+online"]


class AgentState(TypedDict, total=False):
    mode: Mode
    question: str
    context_chunks: list[dict[str, str]]
    answer: str
    sources: list[str]
    used_mode: UsedMode
    stream: bool
