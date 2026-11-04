"""The second guardrail: is this payment allowed?

Configuration-driven rules, checked against a validated intent and the ledger.
The only place a payment decision is made: the LLM proposes, this decides.

Rules are checked in order and the first objection wins. Every deny rule comes
before the approval rule, so a payment that breaks a hard limit is never put
in front of a human - otherwise the approval queue becomes a way to get a
payment past a policy that already refused it.
"""

from __future__ import annotations

from datetime import datetime, timezone

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


class PolicyEngine:
    def __init__(self, guardrails: GuardrailSettings) -> None:
        self._guardrails = guardrails

    def evaluate(
        self,
        intent: PaymentIntent,
        ledger,
        now: datetime | None = None,
    ) -> Evaluation:
        """Decide a single payment.

        `ledger` is anything that can answer total_spent_minor. `now` is
        injectable so the rolling window can be tested without waiting.
        """
        rules = self._guardrails
        now = now or datetime.now(timezone.utc)

        def money(amount_minor: int) -> str:
            return format_amount(amount_minor, intent.currency)

        def in_minor_units(major_amount) -> int:
            # Limits are configured in major units; the currency is only known
            # here, which is why they are not converted at load time.
            return to_minor_units(major_amount, intent.currency)

        if intent.currency not in rules.allowed_currencies:
            return Evaluation(
                Decision.DENIED,
                f"Payments in {intent.currency} are not permitted. Allowed "
                f"currencies: {', '.join(rules.allowed_currencies)}.",
                RULE_CURRENCY,
            )

        if intent.category not in rules.allowed_categories:
            return Evaluation(
                Decision.DENIED,
                f"Category '{intent.category}' is not permitted. Allowed "
                f"categories: {', '.join(rules.allowed_categories)}.",
                RULE_CATEGORY,
            )

        maximum = in_minor_units(rules.max_transaction_amount)
        if intent.amount_minor > maximum:
            return Evaluation(
                Decision.DENIED,
                f"{money(intent.amount_minor)} exceeds the {money(maximum)} "
                "maximum for a single payment.",
                RULE_MAX_AMOUNT,
            )

        # Totals are filtered by currency: with no FX conversion in this
        # system, adding pence to cents would be meaningless. The rolling
        # limit is therefore per scope, per currency.
        cumulative = in_minor_units(rules.cumulative.limit)
        spent = ledger.total_spent_minor(
            since=now - rules.cumulative.window,
            currency=intent.currency,
            **self._scope_filter(intent),
        )
        if spent + intent.amount_minor > cumulative:
            return Evaluation(
                Decision.DENIED,
                f"{money(intent.amount_minor)} would take "
                f"{rules.cumulative.window_hours}-hour spend"
                f"{self._scope_phrase(intent)} to "
                f"{money(spent + intent.amount_minor)}, over the "
                f"{money(cumulative)} limit ({money(spent)} already spent).",
                RULE_CUMULATIVE,
            )

        threshold = in_minor_units(rules.auto_approval_threshold)
        if intent.amount_minor > threshold:
            return Evaluation(
                Decision.REQUIRES_APPROVAL,
                f"{money(intent.amount_minor)} is above the {money(threshold)} "
                "auto-approval threshold and needs a human to approve it.",
                RULE_THRESHOLD,
            )

        return Evaluation(
            Decision.APPROVED, "Within configured limits.", RULE_WITHIN_LIMITS
        )

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
