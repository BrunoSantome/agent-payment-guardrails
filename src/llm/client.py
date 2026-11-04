"""Gemini access.

One method: send a conversation and a schema, get back a validated object.
Wrapping the SDK keeps it out of the rest of the system and gives the tests
something to stand in for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from google import genai
from google.genai import types
from pydantic import BaseModel

class LLMError(Exception):
    """No usable structured response came back from the provider."""


@dataclass(frozen=True)
class Message:
    """One turn of the conversation.

    Gemini needs every turn labelled: "user" for the person's requests,
    "model" for our side of the exchange. Without it the model cannot tell
    what it was told from what it said - which matters when the previous turn
    was a denial it now has to work around.
    """

    role: str
    text: str


class GeminiClient:
    def __init__(self, api_key: str, model: str, max_tokens: int) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def structured(
        self, *, instruction: str, messages: Sequence[Message], schema: type[BaseModel]
    ) -> BaseModel:
        """Return an instance of `schema`, or raise LLMError.

        Gemini is constrained to the schema, so there is no JSON parsing or
        retrying on malformed output anywhere in this project.
        """
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    types.Content(role=m.role, parts=[types.Part(text=m.text)])
                    for m in messages
                ],
                config=types.GenerateContentConfig(
                    system_instruction=instruction,
                    response_mime_type="application/json",
                    response_schema=schema,
                    max_output_tokens=self._max_tokens,
                    # Extraction, not brainstorming.
                    temperature=0.0,
                ),
            )
        except Exception as exc:
            # The SDK raises many transport and API error types; the caller
            # only needs to know the request did not work.
            raise LLMError(f"Gemini request failed: {exc}") from exc

        if response.parsed is None:
            raise LLMError(
                f"Gemini returned no usable JSON: {(response.text or '')[:200]!r}"
            )
        return response.parsed
