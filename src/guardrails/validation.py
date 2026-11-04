"""The first guardrail: is this a payment at all?

Structural checks only. Nothing here is configurable and nothing here needs
the ledger, so this is a pure function over whatever JSON the model produced.

Whether a payment is *allowed* is a different question, answered by
policy.py against the configuration. The two are separate because the remedy
differs: a validation failure means the agent must correct a field, a policy
denial means it must propose a different payment.

Lenient about format, strict about value. A currency symbol the model echoed
back is parsed rather than refused; an amount that is zero, negative, or finer
than the currency's smallest unit is refused rather than adjusted.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from collections.abc import Mapping
from typing import Any

from src.payment.models import PaymentIntent, UnsupportedCurrency, to_minor_units

REQUIRED_FIELDS = ("amount", "currency", "beneficiary", "category", "justification")

# These strings reach the ledger and the receipt, and they originate in a user
# prompt. Capping their length keeps a prompt-shaped blob out of the audit
# trail; it is not a defence against prompt injection, which is noted in the
# README.
MAX_LENGTHS = {"currency": 8, "beneficiary": 120, "category": 60, "justification": 500}

# Symbols and separators stripped from an amount before parsing. The agent is
# instructed to send bare digits; being tolerant here avoids refusing a real
# payment over punctuation.
_AMOUNT_NOISE = re.compile(r"[£$€,\s]")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class IntentValidationError(Exception):
    """The model returned usable JSON that is not a usable payment intent."""


def validate_intent(draft: Mapping[str, Any]) -> PaymentIntent:
    """Turn untrusted JSON into a PaymentIntent, or raise.

    Takes a plain mapping rather than the agent's schema class on purpose:
    this layer exists to distrust the shape it is handed.
    """
    missing = [field for field in REQUIRED_FIELDS if draft.get(field) is None]
    if missing:
        raise IntentValidationError(
            f"Missing required field(s): {', '.join(missing)}."
        )

    currency = _text(draft, "currency").upper()
    amount_minor = _amount(draft["amount"], currency)

    return PaymentIntent(
        amount_minor=amount_minor,
        currency=currency,
        beneficiary=_text(draft, "beneficiary"),
        category=_text(draft, "category").lower(),
        justification=_text(draft, "justification"),
    )


def _amount(raw: Any, currency: str) -> int:
    if isinstance(raw, bool):  # bool is an int subclass; never a valid amount
        raise IntentValidationError("Amount must be a number, got a boolean.")

    # via str() so a float from the model does not bring binary rounding with it
    text = _AMOUNT_NOISE.sub("", str(raw))
    if not text:
        raise IntentValidationError("Amount is empty.")

    try:
        amount = Decimal(text)
    except InvalidOperation:
        raise IntentValidationError(f"Amount '{raw}' is not a number.") from None
    if not amount.is_finite():
        raise IntentValidationError(f"Amount '{raw}' is not a finite number.")
    if amount <= 0:
        raise IntentValidationError(f"Amount must be greater than zero, got '{raw}'.")

    try:
        return to_minor_units(amount, currency)
    except UnsupportedCurrency as exc:
        raise IntentValidationError(str(exc)) from exc
    except ValueError:
        raise IntentValidationError(
            f"Amount '{raw}' is finer than the smallest unit of {currency}."
        ) from None


def _text(draft: Mapping[str, Any], field: str) -> str:
    raw = draft[field]
    if isinstance(raw, (dict, list, tuple, set)):
        raise IntentValidationError(f"Field '{field}' must be text, got {type(raw).__name__}.")

    # Collapse runs of whitespace and drop control characters, so a multi-line
    # blob cannot masquerade as a beneficiary name.
    value = " ".join(_CONTROL_CHARS.sub("", str(raw)).split())
    if not value:
        raise IntentValidationError(f"Field '{field}' is empty.")

    limit = MAX_LENGTHS[field]
    if len(value) > limit:
        raise IntentValidationError(
            f"Field '{field}' is longer than the {limit}-character limit."
        )
    return value
