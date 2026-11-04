"""Policy evaluation: the rules, their order, and their boundaries."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.config.settings import CumulativeLimit, GuardrailSettings
from src.guardrails.policy import (
    RULE_CATEGORY,
    RULE_CUMULATIVE,
    RULE_CURRENCY,
    RULE_MAX_AMOUNT,
    RULE_THRESHOLD,
    RULE_WITHIN_LIMITS,
    PolicyEngine,
)
from src.payment.models import Decision, PaymentIntent

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)


class StubLedger:
    """Returns a fixed total and records how it was queried."""

    def __init__(self, total_minor: int = 0) -> None:
        self.total_minor = total_minor
        self.queries: list[dict] = []

    def total_spent_minor(self, **kwargs) -> int:
        self.queries.append(kwargs)
        return self.total_minor


def guardrails(**overrides) -> GuardrailSettings:
    defaults = {
        "max_transaction_amount": Decimal("1000"),
        "auto_approval_threshold": Decimal("500"),
        "cumulative": CumulativeLimit(
            window_hours=24, scope="beneficiary", limit=Decimal("5000")
        ),
        "allowed_categories": ("equipment", "salary", "travel"),
        "allowed_currencies": ("GBP", "EUR", "USD"),
    }
    return GuardrailSettings(**{**defaults, **overrides})


def intent(amount_minor=40000, currency="GBP", beneficiary="Apple", category="equipment"):
    return PaymentIntent(
        amount_minor=amount_minor,
        currency=currency,
        beneficiary=beneficiary,
        category=category,
        justification="MacBook for the new starter",
    )


def evaluate(payment=None, *, spent=0, now=NOW, **config):
    engine = PolicyEngine(guardrails(**config))
    return engine.evaluate(payment or intent(), StubLedger(spent), now=now)


# --- the happy path --------------------------------------------------------


def test_a_payment_within_every_limit_is_approved():
    result = evaluate(intent(amount_minor=40000))
    assert result.decision is Decision.APPROVED
    assert result.rule == RULE_WITHIN_LIMITS


# --- single-payment maximum and approval threshold -------------------------


def test_exactly_at_the_threshold_is_auto_approved():
    assert evaluate(intent(amount_minor=50000)).decision is Decision.APPROVED


def test_a_penny_over_the_threshold_needs_a_human():
    result = evaluate(intent(amount_minor=50001))
    assert result.decision is Decision.REQUIRES_APPROVAL
    assert result.rule == RULE_THRESHOLD


def test_exactly_at_the_maximum_is_held_not_denied():
    result = evaluate(intent(amount_minor=100000))
    assert result.decision is Decision.REQUIRES_APPROVAL


def test_a_penny_over_the_maximum_is_denied():
    result = evaluate(intent(amount_minor=100001))
    assert result.decision is Decision.DENIED
    assert result.rule == RULE_MAX_AMOUNT


def test_the_denial_reason_names_both_amounts():
    reason = evaluate(intent(amount_minor=500000)).reason
    assert "5,000.00 GBP" in reason and "1,000.00 GBP" in reason


# --- allow-lists -----------------------------------------------------------


def test_a_disallowed_currency_is_denied():
    result = evaluate(intent(currency="EUR"), allowed_currencies=("GBP",))
    assert result.decision is Decision.DENIED
    assert result.rule == RULE_CURRENCY
    assert "EUR" in result.reason and "GBP" in result.reason


def test_a_disallowed_category_is_denied():
    result = evaluate(intent(category="audit"))
    assert result.decision is Decision.DENIED
    assert result.rule == RULE_CATEGORY
    assert "audit" in result.reason


# --- rule order ------------------------------------------------------------


def test_a_hard_limit_is_never_offered_to_a_human():
    """An over-maximum payment must not reach the approval queue."""
    assert evaluate(intent(amount_minor=100001)).decision is Decision.DENIED


def test_the_allow_list_is_reported_before_the_amount():
    """Both rules fail; the first in order is the one reported."""
    result = evaluate(intent(amount_minor=500000, category="audit"))
    assert result.rule == RULE_CATEGORY


def test_the_currency_is_reported_before_the_category():
    result = evaluate(
        intent(currency="USD", category="audit"), allowed_currencies=("GBP",)
    )
    assert result.rule == RULE_CURRENCY


# --- cumulative limit ------------------------------------------------------


def test_cumulative_spend_within_the_limit_passes():
    result = evaluate(intent(amount_minor=40000), spent=100000)
    assert result.decision is Decision.APPROVED


def test_exactly_at_the_cumulative_limit_passes():
    result = evaluate(intent(amount_minor=50000), spent=450000)
    assert result.decision is Decision.APPROVED


def test_a_penny_over_the_cumulative_limit_is_denied():
    result = evaluate(intent(amount_minor=50000), spent=450001)
    assert result.decision is Decision.DENIED
    assert result.rule == RULE_CUMULATIVE


def test_the_cumulative_reason_shows_the_arithmetic():
    result = evaluate(intent(amount_minor=50000), spent=490000)
    for fragment in ("500.00 GBP", "5,400.00 GBP", "5,000.00 GBP", "4,900.00 GBP"):
        assert fragment in result.reason
    assert "24-hour" in result.reason and "Apple" in result.reason


def test_the_cumulative_rule_is_checked_before_the_threshold():
    """A payment over the rolling limit is denied, not held."""
    result = evaluate(intent(amount_minor=90000), spent=450000)
    assert result.decision is Decision.DENIED
    assert result.rule == RULE_CUMULATIVE


# --- how the ledger is queried ---------------------------------------------


def test_the_window_start_is_now_minus_the_configured_hours():
    ledger = StubLedger()
    PolicyEngine(guardrails()).evaluate(intent(), ledger, now=NOW)
    assert ledger.queries[0]["since"] == NOW - timedelta(hours=24)


def test_a_different_window_moves_the_start():
    ledger = StubLedger()
    engine = PolicyEngine(
        guardrails(
            cumulative=CumulativeLimit(
                window_hours=168, scope="beneficiary", limit=Decimal("5000")
            )
        )
    )
    engine.evaluate(intent(), ledger, now=NOW)
    assert ledger.queries[0]["since"] == NOW - timedelta(hours=168)


def test_totals_are_isolated_per_currency():
    """No FX conversion, so pence and cents are never added together."""
    ledger = StubLedger()
    PolicyEngine(guardrails()).evaluate(intent(currency="EUR"), ledger, now=NOW)
    assert ledger.queries[0]["currency"] == "EUR"


def test_beneficiary_scope_filters_by_beneficiary():
    ledger = StubLedger()
    PolicyEngine(guardrails()).evaluate(intent(beneficiary="Apple"), ledger, now=NOW)
    assert ledger.queries[0] == {
        "since": NOW - timedelta(hours=24),
        "currency": "GBP",
        "beneficiary": "Apple",
    }


def test_category_scope_filters_by_category():
    ledger = StubLedger()
    engine = PolicyEngine(
        guardrails(
            cumulative=CumulativeLimit(
                window_hours=24, scope="category", limit=Decimal("5000")
            )
        )
    )
    engine.evaluate(intent(category="travel"), ledger, now=NOW)
    assert ledger.queries[0].get("category") == "travel"
    assert "beneficiary" not in ledger.queries[0]


def test_global_scope_filters_by_neither():
    ledger = StubLedger()
    engine = PolicyEngine(
        guardrails(
            cumulative=CumulativeLimit(
                window_hours=24, scope="global", limit=Decimal("5000")
            )
        )
    )
    engine.evaluate(intent(), ledger, now=NOW)
    assert "beneficiary" not in ledger.queries[0]
    assert "category" not in ledger.queries[0]


def test_the_ledger_is_not_queried_when_an_earlier_rule_already_denied():
    """Rules short-circuit: a disallowed category never hits the store."""
    ledger = StubLedger()
    PolicyEngine(guardrails()).evaluate(intent(category="audit"), ledger, now=NOW)
    assert ledger.queries == []


# --- limits are per currency, not per exponent -----------------------------


@pytest.mark.parametrize("currency", ["GBP", "EUR", "USD"])
def test_the_same_numeric_limit_applies_to_every_allowed_currency(currency):
    assert evaluate(intent(amount_minor=100001, currency=currency)).decision is (
        Decision.DENIED
    )
