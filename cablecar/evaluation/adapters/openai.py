"""OpenAI (GPT) adapter stub for the model-agnostic benchmark harness.

Implements :class:`OpenAIAdapter(LLMAdapter)` with the interface defined but
not yet functional.  Documents the OpenAI tools/function conversion pattern.

Requires ``openai`` to be installed::

    pip install openai>=1.0.0

Conversion pattern for OpenAI tool use::

    # ToolSchema -> OpenAI tool format
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": schema.name,
                "description": schema.description,
                "parameters": schema.input_schema,
            },
        }
        for schema in tool_schemas
    ]

    # In the response, tool calls appear as:
    #   response.choices[0].message.tool_calls[i]
    # with .id, .function.name, .function.arguments (JSON string)

    # Tool results are sent back as messages with role="tool":
    #   {
    #       "role": "tool",
    #       "tool_call_id": call.id,
    #       "content": result_json_string,
    #   }

    # The loop terminates when:
    #   response.choices[0].finish_reason != "tool_calls"
"""

from __future__ import annotations

from cablecar.evaluation.adapters import (
    AdapterResult,
    LLMAdapter,
    ToolDispatcher,
    ToolSchema,
)


class OpenAIAdapter(LLMAdapter):
    """LLM adapter for OpenAI GPT models with tool use.

    Parameters
    ----------
    model:
        Model identifier (e.g. ``"gpt-4o"``, ``"gpt-4o-mini"``).
    api_key:
        OpenAI API key.  If ``None``, uses ``OPENAI_API_KEY`` env var.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key

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
        """Run the multi-turn tool-use loop with the OpenAI API.

        Not yet implemented.  See module docstring for the conversion pattern.
        """
        raise NotImplementedError(
            "OpenAIAdapter is not yet implemented. "
            "See cablecar/evaluation/adapters/openai.py docstring for "
            "the OpenAI tools/function conversion pattern."
        )
