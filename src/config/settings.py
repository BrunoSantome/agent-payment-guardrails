"""Loads config.yaml and the environment into typed settings.

Everything that reads configuration goes through here, so a malformed config
fails once, at startup, with a message naming the problem - rather than as a
KeyError somewhere inside the policy engine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")

# Environment variable holding the API key, per provider.
API_KEY_VARS = {"gemini": "GEMINI_API_KEY"}

CUMULATIVE_SCOPES = ("beneficiary", "category", "global")


class ConfigError(Exception):
    """Configuration is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    max_tokens: int


@dataclass(frozen=True)
class CumulativeLimit:
    """A rolling-window spend cap, grouped by `scope`."""

    window_hours: int
    scope: str
    limit: Decimal  # major units

    @property
    def window(self) -> timedelta:
        return timedelta(hours=self.window_hours)


@dataclass(frozen=True)
class GuardrailSettings:
    # Major units. The policy engine converts these to minor units using the
    # currency of the payment being evaluated, so no exponent is assumed here.
    max_transaction_amount: Decimal
    auto_approval_threshold: Decimal
    cumulative: CumulativeLimit
    allowed_categories: tuple[str, ...]
    allowed_currencies: tuple[str, ...]


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings
    guardrails: GuardrailSettings


def load_settings(path: Path | None = None) -> Settings:
    config_path = path or DEFAULT_CONFIG_PATH
    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"No config file at {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {config_path}: {exc}") from exc

    return Settings(
        llm=_llm_settings(_section(raw, "llm")),
        guardrails=_guardrail_settings(_section(raw, "guardrails")),
    )


def load_api_key(provider: str) -> str:
    """Read the provider's API key from .env or the environment."""
    load_dotenv()
    var = API_KEY_VARS.get(provider)
    if var is None:
        raise ConfigError(f"No API key variable known for provider '{provider}'")
    key = (os.getenv(var) or "").strip()
    if not key:
        raise ConfigError(f"{var} is not set. Put it in a .env file at the repo root.")
    return key


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"config is missing a '{name}:' section")
    return section


def _llm_settings(raw: dict[str, Any]) -> LLMSettings:
    provider = str(raw.get("provider") or "").strip().lower()
    model = str(raw.get("model") or "").strip()
    if not provider:
        raise ConfigError("llm.provider is required")
    if not model:
        raise ConfigError("llm.model is required")
    if provider not in API_KEY_VARS:
        raise ConfigError(
            f"Unknown llm.provider '{provider}'. Known: {', '.join(API_KEY_VARS)}"
        )
    return LLMSettings(
        provider=provider,
        model=model,
        max_tokens=_positive_int(raw, "max_tokens", "llm.max_tokens", default=4096),
    )


def _guardrail_settings(raw: dict[str, Any]) -> GuardrailSettings:
    max_amount = _amount(raw, "max_transaction_amount", "guardrails.max_transaction_amount")
    threshold = _amount(raw, "auto_approval_threshold", "guardrails.auto_approval_threshold")
    if threshold > max_amount:
        # Otherwise nothing can ever be held: everything at or under the max
        # is auto-approved and everything above it is denied.
        raise ConfigError(
            "guardrails.auto_approval_threshold must not exceed "
            "guardrails.max_transaction_amount, or no payment is ever held "
            "for approval"
        )

    cumulative_raw = raw.get("cumulative")
    if not isinstance(cumulative_raw, dict):
        raise ConfigError("config is missing a 'guardrails.cumulative:' section")
    scope = str(cumulative_raw.get("scope") or "").strip().lower()
    if scope not in CUMULATIVE_SCOPES:
        raise ConfigError(
            f"guardrails.cumulative.scope must be one of "
            f"{', '.join(CUMULATIVE_SCOPES)}, got '{scope}'"
        )
    cumulative = CumulativeLimit(
        window_hours=_positive_int(cumulative_raw, "window_hours", "guardrails.cumulative.window_hours"),
        scope=scope,
        limit=_amount(cumulative_raw, "limit", "guardrails.cumulative.limit"),
    )
    if cumulative.limit < max_amount:
        raise ConfigError(
            "guardrails.cumulative.limit is below "
            "guardrails.max_transaction_amount, so a single payment at the "
            "maximum could never be made"
        )

    return GuardrailSettings(
        max_transaction_amount=max_amount,
        auto_approval_threshold=threshold,
        cumulative=cumulative,
        allowed_categories=_string_list(raw, "allowed_categories", "guardrails.allowed_categories", lower=True),
        allowed_currencies=_string_list(raw, "allowed_currencies", "guardrails.allowed_currencies", upper=True),
    )


def _amount(raw: dict[str, Any], key: str, label: str) -> Decimal:
    value = raw.get(key)
    if value is None:
        raise ConfigError(f"{label} is required")
    try:
        # via str() so a float in the YAML does not drag binary rounding in
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ConfigError(f"{label} is not a number: {value!r}") from exc
    if not amount.is_finite() or amount <= 0:
        raise ConfigError(f"{label} must be greater than zero, got {value!r}")
    return amount


def _positive_int(
    raw: dict[str, Any], key: str, label: str, *, default: int | None = None
) -> int:
    value = raw.get(key, default)
    if value is None:
        raise ConfigError(f"{label} is required")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} is not a whole number: {value!r}") from exc
    if number <= 0:
        raise ConfigError(f"{label} must be greater than zero, got {value!r}")
    return number


def _string_list(
    raw: dict[str, Any], key: str, label: str, *, lower: bool = False, upper: bool = False
) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{label} must be a non-empty list")
    items = []
    for item in value:
        text = str(item).strip()
        if not text:
            raise ConfigError(f"{label} contains an empty entry")
        items.append(text.lower() if lower else text.upper() if upper else text)
    return tuple(items)
