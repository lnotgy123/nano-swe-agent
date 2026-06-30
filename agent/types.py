from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from tools.executor import ToolResult


class LLM(Protocol):
    def generate(self, messages: list[dict[str, str]], **overrides: object) -> str:
        ...

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        ...


@dataclass
class AgentStep:
    thought: str
    tool: str
    args: dict
    result: ToolResult | None = None


@dataclass
class AgentRunResult:
    finished: bool
    summary: str
    task: str
    repo_context: ToolResult
    messages: list[dict[str, str]]
    steps: list[AgentStep] = field(default_factory=list)
