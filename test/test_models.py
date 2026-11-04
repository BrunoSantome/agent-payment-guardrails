"""Money conversion, formatting, and the payment state machine."""

from decimal import Decimal

import pytest

from src.payment.models import (
    ALLOWED_TRANSITIONS,
    Decision,
    IllegalTransition,
    PaymentIntent,
    PaymentRecord,
    PaymentState,
    UnsupportedCurrency,
    format_amount,
    new_payment_id,
    new_transaction_id,
    to_major_units,
    to_minor_units,
)


def make_record(state: PaymentState) -> PaymentRecord:
    return PaymentRecord(
        intent=PaymentIntent(
            amount_minor=90000,
            currency="GBP",
            beneficiary="British Airways",
            category="travel",
            justification="Sales trip",
        ),
        decision=Decision.REQUIRES_APPROVAL,
        reason="Above the auto-approval threshold",
        state=state,
    )


# --- money -----------------------------------------------------------------


@pytest.mark.parametrize(
    "major, expected",
    [("49.99", 4999), ("1000", 100000), ("0.01", 1), ("2499.50", 249950)],
)
def test_major_to_minor(major, expected):
    assert to_minor_units(Decimal(major), "GBP") == expected


def test_sub_unit_amount_is_refused_not_rounded():
    """Silently truncating a third of a penny is how money goes missing."""
    with pytest.raises(ValueError, match="finer than the smallest unit"):
        to_minor_units(Decimal("10.001"), "GBP")


def test_round_trip_is_exact():
    assert to_major_units(to_minor_units(Decimal("1234.56"), "EUR"), "EUR") == Decimal(
        "1234.56"
    )


def test_unknown_currency_is_refused():
    with pytest.raises(UnsupportedCurrency, match="JPY"):
        to_minor_units(Decimal("1000"), "JPY")


def test_format_amount():
    assert format_amount(249950, "GBP") == "2,499.50 GBP"
    assert format_amount(1, "USD") == "0.01 USD"


def test_ids_are_prefixed_and_unique():
    assert new_payment_id().startswith("pay_")
    assert new_transaction_id().startswith("txn_")
    assert len({new_payment_id() for _ in range(100)}) == 100


def test_intent_renders_for_humans():
    intent = make_record(PaymentState.HELD).intent
    assert str(intent) == "900.00 GBP to British Airways (travel)"


def test_decision_values_match_the_api_contract():
    assert [d.value for d in Decision] == ["approved", "requires_approval", "denied"]


# --- state machine ---------------------------------------------------------


def test_approved_payment_can_execute():
    record = make_record(PaymentState.APPROVED)
    record.transition_to(PaymentState.EXECUTED)
    assert record.state is PaymentState.EXECUTED


def test_held_payment_cannot_execute_without_approval():
    """A held payment must go through a human first."""
    record = make_record(PaymentState.HELD)
    with pytest.raises(IllegalTransition, match="cannot become executed"):
        record.transition_to(PaymentState.EXECUTED)


def test_held_payment_executes_once_approved():
    record = make_record(PaymentState.HELD)
    record.transition_to(PaymentState.APPROVED)
    record.transition_to(PaymentState.EXECUTED)
    assert record.state is PaymentState.EXECUTED


def test_a_payment_cannot_execute_twice():
    record = make_record(PaymentState.APPROVED)
    record.transition_to(PaymentState.EXECUTED)
    with pytest.raises(IllegalTransition):
        record.transition_to(PaymentState.EXECUTED)


def test_a_rejected_payment_cannot_be_executed_afterwards():
    record = make_record(PaymentState.HELD)
    record.transition_to(PaymentState.DENIED)
    with pytest.raises(IllegalTransition):
        record.transition_to(PaymentState.EXECUTED)


def test_a_failed_payment_is_terminal():
    record = make_record(PaymentState.APPROVED)
    record.transition_to(PaymentState.FAILED)
    assert record.is_terminal
    with pytest.raises(IllegalTransition):
        record.transition_to(PaymentState.EXECUTED)


def test_transition_updates_the_timestamp():
    record = make_record(PaymentState.APPROVED)
    before = record.updated_at
    record.transition_to(PaymentState.EXECUTED)
    assert record.updated_at >= before


def test_every_state_has_a_transition_rule():
    """A state missing from the table would be silently terminal."""
    assert set(ALLOWED_TRANSITIONS) == set(PaymentState)


def test_only_executed_payments_count_towards_spend():
    counted = {
        state for state in PaymentState if make_record(state).counts_towards_spend()
    }
    assert counted == {PaymentState.EXECUTED}
