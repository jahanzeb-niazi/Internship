"""
Step 3b: Google Gemini LLM (AI Studio)
---------------------------------------
Real tool selection via Gemini function calling.

We DISABLE automatic function calling so YOU still see each ReAct step
and approval gates run before tools execute.

Get a free key: https://aistudio.google.com/apikey
Set GEMINI_API_KEY in .env
"""

from __future__ import annotations

from typing import Any, Optional

from google import genai
from google.genai import types

from config import get_gemini_api_key, get_gemini_model
from llm_types import FinalAnswer, ToolCall
from tools import tools_for_gemini

SYSTEM_INSTRUCTION = """You are a helpful agent with tools for weather and tasks.

Rules:
- Use tools when the user asks about weather or todo tasks.
- Never invent weather or task data; call the appropriate tool first.
- For create_task and delete_task, propose the tool call; the user may approve or deny.
- When tool results are available, give a short, friendly final answer.
- If a tool was denied, explain that and suggest alternatives."""


class GeminiLLM:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        key = api_key or get_gemini_api_key()
        if not key:
            raise ValueError(
                "Missing GEMINI_API_KEY. Create .env from .env.example "
                "or export the variable from https://aistudio.google.com/apikey"
            )
        self.client = genai.Client(api_key=key)
        self.model = model or get_gemini_model()
        self.contents: list[Any] = []
        self._last_model_content: Any = None
        self._pending_call: Optional[dict] = None

    def reset(self, user_message: str) -> None:
        self.contents = [
            types.Content(role="user", parts=[types.Part(text=user_message)])
        ]
        self._last_model_content = None
        self._pending_call = None

    def decide(self) -> ToolCall | FinalAnswer:
        response = self.client.models.generate_content(
            model=self.model,
            contents=self.contents,
            config=self._generate_config(),
        )

        candidate = response.candidates[0].content
        self._last_model_content = candidate

        text_parts: list[str] = []
        function_calls = []
        for part in candidate.parts:
            if part.text:
                text_parts.append(part.text)
            if part.function_call:
                function_calls.append(part.function_call)

        thought = " ".join(text_parts).strip()
        if not thought:
            thought = "Gemini matched the request to a tool description."

        if function_calls:
            fc = function_calls[0]
            args = dict(fc.args) if fc.args else {}
            self._pending_call = {"name": fc.name, "id": getattr(fc, "id", None)}
            return ToolCall(
                name=fc.name,
                arguments=args,
                thought=f"{thought} -> selected tool: {fc.name}",
            )

        final_text = (response.text or thought or "").strip()
        if not final_text:
            final_text = "I could not produce an answer. Try rephrasing your question."
        return FinalAnswer(text=final_text, thought=thought)

    def record_tool_result(self, name: str, result: dict) -> None:
        self._append_function_response(name, {"result": result})

    def record_denial(self, name: str, reason: str) -> None:
        self._append_function_response(name, {"denied": True, "reason": reason})

    def _append_function_response(self, name: str, payload: dict) -> None:
        if self._last_model_content is not None:
            self.contents.append(self._last_model_content)

        call_id = self._pending_call.get("id") if self._pending_call else None
        if call_id:
            part = types.Part(
                function_response=types.FunctionResponse(
                    name=name,
                    response=payload,
                    id=call_id,
                )
            )
        else:
            part = types.Part.from_function_response(name=name, response=payload)

        self.contents.append(types.Content(role="user", parts=[part]))
        self._last_model_content = None
        self._pending_call = None

    def _generate_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=tools_for_gemini())],
            system_instruction=SYSTEM_INSTRUCTION,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
