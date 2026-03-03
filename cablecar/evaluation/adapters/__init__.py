"""LLM adapter protocol and shared types for the model-agnostic benchmark harness.

Defines:
- :class:`ToolSchema` — tool definition sent to the LLM.
- :class:`ConversationMessage` — a single message in the tool-use transcript.
- :class:`AdapterResult` — summary returned after the tool-use loop completes.
- :class:`ToolDispatcher` — protocol for routing tool calls to local handlers.
- :class:`LLMAdapter` — abstract base class that each provider adapter implements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


class ToolSchema(BaseModel):
    """Schema for a single tool exposed to the LLM."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ConversationMessage(BaseModel):
    """A single message in the tool-use transcript."""

    role: str = Field(
        ..., description="One of 'system', 'user', 'assistant', 'tool'."
    )
    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdapterResult(BaseModel):
    """Result returned by an LLM adapter after the tool-use loop completes."""

    final_text: str = ""
    transcript: list[ConversationMessage] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    model: str = ""
    turns_used: int = 0
    terminated_reason: str = Field(
        default="",
        description="Why the loop ended: 'max_turns', 'end_turn', 'no_tool_use', 'error'.",
    )


# ---------------------------------------------------------------------------
# Dispatcher protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ToolDispatcher(Protocol):
    """Protocol for routing tool calls to local handlers."""

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute *tool_name* with *arguments* and return a JSON string."""
        ...


# ---------------------------------------------------------------------------
# LLM adapter ABC
# ---------------------------------------------------------------------------


class LLMAdapter(ABC):
    """Abstract base class for provider-specific LLM adapters.

    Each adapter implements a multi-turn tool-use loop:
    send system prompt + tools -> get response -> dispatch tool calls ->
    feed results back -> repeat until the LLM stops calling tools.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier (e.g. 'claude-sonnet-4-5-20250514')."""
        ...

    @abstractmethod
    def run_tool_loop(
        self,
        system_prompt: str,
        tools: list[ToolSchema],
        tool_dispatcher: ToolDispatcher,
        max_turns: int = 20,
    ) -> AdapterResult:
        """Run the multi-turn tool-use conversation.

        Parameters
        ----------
        system_prompt:
            The system prompt describing the agent's role and context.
        tools:
            Tool schemas to expose to the LLM.
        tool_dispatcher:
            Routes tool calls to local handlers and returns results.
        max_turns:
            Maximum number of LLM round-trips before stopping.

        Returns
        -------
        AdapterResult
            Contains the final text, full transcript, and token usage.
        """
        ...


__all__ = [
    "AdapterResult",
    "ConversationMessage",
    "LLMAdapter",
    "ToolDispatcher",
    "ToolSchema",
]
