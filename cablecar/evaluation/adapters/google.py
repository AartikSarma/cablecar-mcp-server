"""Google (Gemini / MedGemma) adapter stub for the model-agnostic benchmark harness.

Implements :class:`GoogleAdapter(LLMAdapter)` with the interface defined but
not yet functional.  Documents the Gemini function_calling conversion pattern.

Requires ``google-generativeai`` to be installed::

    pip install google-generativeai>=0.8.0

Conversion pattern for Gemini function calling::

    # ToolSchema -> Gemini FunctionDeclaration
    from google.generativeai.types import FunctionDeclaration, Tool

    declarations = [
        FunctionDeclaration(
            name=schema.name,
            description=schema.description,
            parameters=schema.input_schema,  # JSON Schema -> OpenAPI subset
        )
        for schema in tool_schemas
    ]
    gemini_tools = [Tool(function_declarations=declarations)]

    # In the response, function calls appear as:
    #   response.candidates[0].content.parts[i].function_call
    # with .name and .args attributes.

    # Tool results are sent back as:
    #   genai.protos.Part(
    #       function_response=genai.protos.FunctionResponse(
    #           name=call.name,
    #           response={"result": result_json},
    #       )
    #   )
"""

from __future__ import annotations

from cablecar.evaluation.adapters import (
    AdapterResult,
    LLMAdapter,
    ToolDispatcher,
    ToolSchema,
)


class GoogleAdapter(LLMAdapter):
    """LLM adapter for Google Gemini / MedGemma with function calling.

    Parameters
    ----------
    model:
        Model identifier (e.g. ``"gemini-2.0-flash"``, ``"medgemma"``).
    api_key:
        Google AI API key.  If ``None``, uses ``GOOGLE_API_KEY`` env var.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
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
        """Run the multi-turn function-calling loop with the Gemini API.

        Not yet implemented.  See module docstring for the conversion pattern.
        """
        raise NotImplementedError(
            "GoogleAdapter is not yet implemented. "
            "See cablecar/evaluation/adapters/google.py docstring for "
            "the Gemini function_calling conversion pattern."
        )
