"""Anthropic (Claude) adapter for the model-agnostic benchmark harness.

Implements :class:`AnthropicAdapter(LLMAdapter)` with full multi-turn
tool-use support via the Anthropic Python SDK.

Requires ``anthropic`` to be installed::

    pip install anthropic>=0.39.0

Set the ``ANTHROPIC_API_KEY`` environment variable before use.
"""

from __future__ import annotations

import json
from typing import Any

from cablecar.evaluation.adapters import (
    AdapterResult,
    ConversationMessage,
    LLMAdapter,
    ToolDispatcher,
    ToolSchema,
)


def _convert_tools(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    """Convert :class:`ToolSchema` list to Anthropic tool format."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        }
        for t in tools
    ]


class AnthropicAdapter(LLMAdapter):
    """LLM adapter for the Anthropic Messages API with tool use.

    Parameters
    ----------
    model:
        Model identifier (e.g. ``"claude-sonnet-4-5-20250514"``).
    max_tokens:
        Maximum tokens per response.
    temperature:
        Sampling temperature.
    api_key:
        Anthropic API key.  If ``None``, uses ``ANTHROPIC_API_KEY`` env var.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The Anthropic adapter requires the 'anthropic' package. "
                "Install it with: pip install 'anthropic>=0.39.0' "
                "or: pip install 'cablecar[anthropic]'"
            ) from exc

        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def run_tool_loop(
        self,
        system_prompt: str,
        tools: list[ToolSchema],
        tool_dispatcher: ToolDispatcher,
        max_turns: int = 20,
    ) -> AdapterResult:
        """Run the multi-turn tool-use conversation with the Anthropic API.

        The loop:
        1. Send the system prompt, tools, and messages.
        2. If the response contains ``tool_use`` blocks, dispatch each one
           via ``tool_dispatcher`` and append the results.
        3. Repeat until the model stops calling tools, emits ``end_turn``,
           or ``max_turns`` is reached.
        """
        anthropic_tools = _convert_tools(tools)
        messages: list[dict[str, Any]] = []
        transcript: list[ConversationMessage] = []

        total_input_tokens = 0
        total_output_tokens = 0
        turns_used = 0
        terminated_reason = ""
        final_text = ""

        for turn in range(max_turns):
            turns_used = turn + 1

            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_prompt,
                tools=anthropic_tools,
                messages=messages,
            )

            # Track tokens
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            # Extract text and tool_use blocks from the response
            text_parts: list[str] = []
            tool_use_blocks: list[dict[str, Any]] = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            assistant_text = "\n".join(text_parts)
            final_text = assistant_text

            # Record the assistant message in the transcript
            transcript.append(ConversationMessage(
                role="assistant",
                content=assistant_text,
                tool_calls=tool_use_blocks,
            ))

            # Append the full assistant content block to messages
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

            # If no tool calls, the model is done
            if not tool_use_blocks:
                terminated_reason = (
                    "end_turn" if response.stop_reason == "end_turn"
                    else "no_tool_use"
                )
                break

            # Dispatch each tool call and collect results
            tool_result_blocks: list[dict[str, Any]] = []
            tool_result_entries: list[dict[str, Any]] = []

            for tool_call in tool_use_blocks:
                try:
                    result_str = tool_dispatcher.dispatch(
                        tool_call["name"],
                        tool_call["input"],
                    )
                except Exception as exc:
                    result_str = json.dumps({
                        "error": str(exc),
                        "sanitized": True,
                    })

                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": result_str,
                })
                tool_result_entries.append({
                    "tool_use_id": tool_call["id"],
                    "tool_name": tool_call["name"],
                    "result": result_str,
                })

            # Record tool results in transcript
            transcript.append(ConversationMessage(
                role="tool",
                tool_results=tool_result_entries,
            ))

            # Append tool results to messages
            messages.append({
                "role": "user",
                "content": tool_result_blocks,
            })

            # Check stop reason
            if response.stop_reason == "end_turn":
                terminated_reason = "end_turn"
                break
        else:
            terminated_reason = "max_turns"

        return AdapterResult(
            final_text=final_text,
            transcript=transcript,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            model=self._model,
            turns_used=turns_used,
            terminated_reason=terminated_reason,
        )
