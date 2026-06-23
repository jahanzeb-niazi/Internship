"""
LLM Service Layer — wraps Google Gemini with:
  - Schema validation (Pydantic)
  - Automatic retry on validation failure
  - Structured logging of every call
  - Graceful error handling (never crashes)
"""

from __future__ import annotations

import json
import time
import traceback
from typing import Type, TypeVar

import google.generativeai as genai
from pydantic import BaseModel, ValidationError
from google.api_core.exceptions import ResourceExhausted

from config import GOOGLE_API_KEY, GEMINI_MODEL, MAX_RETRIES, PROMPT_VERSIONS
from logger import log_llm_call
from schemas import LLMCallLog

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

T = TypeVar("T", bound=BaseModel)


def call_llm(
    prompt: str,
    function_name: str,
    response_model: Type[T] | None = None,
    prompt_version: str | None = None,
    temperature: float = 0.7,
    max_retries: int = MAX_RETRIES,
) -> T | str:
    """
    Call Google Gemini with structured output validation and retry logic.

    Args:
        prompt: The full prompt to send to the model.
        function_name: Name of the calling function (for logging).
        response_model: Pydantic model class to validate the response against.
                       If None, returns raw text.
        prompt_version: Version string for this prompt (defaults to config).
        temperature: Sampling temperature.
        max_retries: Number of retries on validation failure.

    Returns:
        Validated Pydantic model instance, or raw string if no response_model.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    if prompt_version is None:
        prompt_version = PROMPT_VERSIONS.get(function_name, "v1.0")

    model = genai.GenerativeModel(GEMINI_MODEL)
    last_error = None

    for attempt in range(1, max_retries + 1):
        start_time = time.time()
        success = False
        error_msg = None
        input_tokens = 0
        output_tokens = 0

        try:
            # Build the full prompt with JSON instruction if schema is needed
            full_prompt = prompt
            if response_model is not None and attempt == 1:
                schema_json = json.dumps(
                    response_model.model_json_schema(), indent=2
                )
                full_prompt += (
                    f"\n\n---\nIMPORTANT: You MUST respond with ONLY a valid JSON object "
                    f"matching this exact schema. No markdown, no explanation, no code fences.\n"
                    f"Schema:\n{schema_json}"
                )
            elif response_model is not None and attempt > 1:
                # On retry, include the previous error to help the model fix it
                full_prompt = (
                    f"{prompt}\n\n---\n"
                    f"IMPORTANT: Your previous response was invalid. Error: {last_error}\n"
                    f"You MUST respond with ONLY a valid JSON object matching this schema. "
                    f"No markdown, no explanation, no code fences.\n"
                    f"Schema:\n{json.dumps(response_model.model_json_schema(), indent=2)}"
                )

            # Call Gemini
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                ),
            )

            # Extract token counts from usage metadata
            usage = response.usage_metadata
            if usage:
                input_tokens = usage.prompt_token_count or 0
                output_tokens = usage.candidates_token_count or 0

            raw_text = response.text.strip()

            # If no schema validation needed, return raw text
            if response_model is None:
                success = True
                latency_ms = (time.time() - start_time) * 1000
                _log_call(
                    function_name, prompt_version, input_tokens,
                    output_tokens, latency_ms, success, None
                )
                return raw_text

            # Clean JSON from potential markdown code fences
            cleaned = _clean_json_response(raw_text)

            # Parse and validate against schema
            parsed = json.loads(cleaned)
            result = response_model.model_validate(parsed)
            success = True

            latency_ms = (time.time() - start_time) * 1000
            _log_call(
                function_name, prompt_version, input_tokens,
                output_tokens, latency_ms, success, None
            )
            return result

        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            error_msg = f"Attempt {attempt}/{max_retries}: {last_error}"
            latency_ms = (time.time() - start_time) * 1000
            _log_call(
                function_name, prompt_version, input_tokens,
                output_tokens, latency_ms, False, error_msg
            )
            if attempt == max_retries:
                raise RuntimeError(
                    f"Schema validation failed after {max_retries} attempts for "
                    f"'{function_name}'. Last error: {last_error}"
                )

        except ResourceExhausted as e:
            last_error = "Rate limit exceeded (429). Waiting 30s..."
            error_msg = f"Attempt {attempt}/{max_retries}: {last_error}"
            latency_ms = (time.time() - start_time) * 1000
            _log_call(
                function_name, prompt_version, input_tokens,
                output_tokens, latency_ms, False, error_msg
            )
            if attempt == max_retries:
                raise RuntimeError(
                    f"Rate limit failed after {max_retries} attempts for "
                    f"'{function_name}'."
                )
            print(f"  [yellow]Rate limit exceeded. Waiting 30s...[/]")
            time.sleep(30)

        except Exception as e:
            last_error = str(e)
            error_msg = f"Attempt {attempt}/{max_retries}: {traceback.format_exc()}"
            latency_ms = (time.time() - start_time) * 1000
            _log_call(
                function_name, prompt_version, input_tokens,
                output_tokens, latency_ms, False, error_msg
            )
            if attempt == max_retries:
                raise RuntimeError(
                    f"LLM call failed after {max_retries} attempts for "
                    f"'{function_name}'. Last error: {last_error}"
                )


def call_llm_raw(
    prompt: str,
    function_name: str,
    prompt_version: str | None = None,
    temperature: float = 0.7,
) -> str:
    """Convenience wrapper — call LLM without schema validation."""
    return call_llm(
        prompt=prompt,
        function_name=function_name,
        response_model=None,
        prompt_version=prompt_version,
        temperature=temperature,
    )


def _clean_json_response(text: str) -> str:
    """
    Strip markdown code fences and extra whitespace from LLM JSON responses.
    Handles ```json ... ``` and ``` ... ``` patterns.
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _log_call(
    function_name: str,
    prompt_version: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    success: bool,
    error: str | None,
) -> None:
    """Create and write a log entry."""
    entry = LLMCallLog(
        function_name=function_name,
        prompt_version=prompt_version,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        latency_ms=round(latency_ms, 2),
        model=GEMINI_MODEL,
        success=success,
        error=error,
    )
    log_llm_call(entry)
