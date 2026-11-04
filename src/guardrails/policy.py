"""The second guardrail: is this payment allowed?

Configuration-driven business rules, evaluated against a validated intent and
the spend ledger. This is the only place a payment decision is made: the LLM
proposes, this decides.

Rules run in a fixed order and the first one to object wins. Every deny rule
runs before the approval rule, so a payment that breaks a hard limit is never
put in front of a human - otherwise the approval queue becomes a way to get a
payment past a policy that already refused it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol

from src.config.settings import GuardrailSettings
from src.payment.models import (
    Decision,
    Evaluation,
    PaymentIntent,
    format_amount,
    to_minor_units,
)

# Rule identifiers. Callers branch on these rather than on the reason prose.
RULE_CURRENCY = "allowed_currencies"
RULE_CATEGORY = "allowed_categories"
RULE_MAX_AMOUNT = "max_transaction_amount"
RULE_CUMULATIVE = "cumulative_limit"
RULE_THRESHOLD = "auto_approval_threshold"
RULE_WITHIN_LIMITS = "within_limits"


class SpendLedger(Protocol):
    """What the policy engine needs from a store, and nothing more.

    Declared here so guardrails/ has no dependency on store/, and so policy
    tests can use a stub instead of a real ledger.
    """

    def total_spent_minor(
        self,
        *,
        since: datetime,
        currency: str,
        beneficiary: str | None = None,
        category: str | None = None,
    ) -> int:
        """Minor units executed since `since`, filtered by the given scope."""


@dataclass(frozen=True)
class _Context:
    ledger: SpendLedger
    now: datetime


Rule = Callable[[PaymentIntent, _Context], "Evaluation | None"]


class PolicyEngine:
    def __init__(self, guardrails: GuardrailSettings) -> None:
        self._guardrails = guardrails

    def evaluate(
        self,
        intent: PaymentIntent,
        ledger: SpendLedger,
        now: datetime | None = None,
    ) -> Evaluation:
        """Decide a single payment. `now` is injectable so the rolling window
        can be tested without waiting for real time to pass."""
        context = _Context(ledger=ledger, now=now or datetime.now(timezone.utc))
        for rule in self._rules():
            objection = rule(intent, context)
            if objection is not None:
                return objection
        return Evaluation(
            decision=Decision.APPROVED,
            reason="Within configured limits.",
            rule=RULE_WITHIN_LIMITS,
        )

    def _rules(self) -> tuple[Rule, ...]:
        # Order is the policy. Deny rules first, approval last.
        return (
            self._currency_is_allowed,
            self._category_is_allowed,
            self._within_single_payment_maximum,
            self._within_cumulative_limit,
            self._below_approval_threshold,
        )

    # --- deny rules --------------------------------------------------------

    def _currency_is_allowed(
        self, intent: PaymentIntent, context: _Context
    ) -> Evaluation | None:
        allowed = self._guardrails.allowed_currencies
        if intent.currency in allowed:
            return None
        return Evaluation(
            decision=Decision.DENIED,
            reason=(
                f"Payments in {intent.currency} are not permitted. "
                f"Allowed currencies: {', '.join(allowed)}."
            ),
            rule=RULE_CURRENCY,
        )

    def _category_is_allowed(
        self, intent: PaymentIntent, context: _Context
    ) -> Evaluation | None:
        allowed = self._guardrails.allowed_categories
        if intent.category in allowed:
            return None
        return Evaluation(
            decision=Decision.DENIED,
            reason=(
                f"Category '{intent.category}' is not permitted. "
                f"Allowed categories: {', '.join(allowed)}."
            ),
            rule=RULE_CATEGORY,
        )

    def _within_single_payment_maximum(
        self, intent: PaymentIntent, context: _Context
    ) -> Evaluation | None:
        limit = self._limit_in_minor_units(
            self._guardrails.max_transaction_amount, intent.currency
        )
        if intent.amount_minor <= limit:
            return None
        return Evaluation(
            decision=Decision.DENIED,
            reason=(
                f"{self._money(intent.amount_minor, intent.currency)} exceeds the "
                f"{self._money(limit, intent.currency)} maximum for a single payment."
            ),
            rule=RULE_MAX_AMOUNT,
        )

    def _within_cumulative_limit(
        self, intent: PaymentIntent, context: _Context
    ) -> Evaluation | None:
        cumulative = self._guardrails.cumulative
        limit = self._limit_in_minor_units(cumulative.limit, intent.currency)
        since = context.now - cumulative.window

        # Totals are filtered by currency: with no FX conversion in this
        # system, adding pence to cents would be meaningless. The rolling
        # limit is therefore per scope, per currency.
        spent = context.ledger.total_spent_minor(
            since=since, currency=intent.currency, **self._scope_filter(intent)
        )
        if spent + intent.amount_minor <= limit:
            return None
        return Evaluation(
            decision=Decision.DENIED,
            reason=(
                f"{self._money(intent.amount_minor, intent.currency)} would take "
                f"{cumulative.window_hours}-hour spend{self._scope_phrase(intent)} to "
                f"{self._money(spent + intent.amount_minor, intent.currency)}, over "
                f"the {self._money(limit, intent.currency)} limit "
                f"({self._money(spent, intent.currency)} already spent)."
            ),
            rule=RULE_CUMULATIVE,
        )

    # --- approval rule -----------------------------------------------------

    def _below_approval_threshold(
        self, intent: PaymentIntent, context: _Context
    ) -> Evaluation | None:
        threshold = self._limit_in_minor_units(
            self._guardrails.auto_approval_threshold, intent.currency
        )
        if intent.amount_minor <= threshold:
            return None
        return Evaluation(
            decision=Decision.REQUIRES_APPROVAL,
            reason=(
                f"{self._money(intent.amount_minor, intent.currency)} is above the "
                f"{self._money(threshold, intent.currency)} auto-approval threshold "
                f"and needs a human to approve it."
            ),
            rule=RULE_THRESHOLD,
        )

    # --- helpers -----------------------------------------------------------

    def _scope_filter(self, intent: PaymentIntent) -> dict[str, str]:
        scope = self._guardrails.cumulative.scope
        if scope == "beneficiary":
            return {"beneficiary": intent.beneficiary}
        if scope == "category":
            return {"category": intent.category}
        return {}  # global: every executed payment in the window counts

    def _scope_phrase(self, intent: PaymentIntent) -> str:
        scope = self._guardrails.cumulative.scope
        if scope == "beneficiary":
            return f" for {intent.beneficiary}"
        if scope == "category":
            return f" on {intent.category}"
        return ""

    @staticmethod
    def _limit_in_minor_units(limit, currency: str) -> int:
        # Limits are configured in major units; the currency is only known
        # here, which is why the conversion is not done at load time.
        return to_minor_units(limit, currency)

    @staticmethod
    def _money(amount_minor: int, currency: str) -> str:
        return format_amount(amount_minor, currency)
