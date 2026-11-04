"""Mocked execution, and what happens when it fails."""

import pytest

from src.payment.executor import PaymentExecutionError, PaymentExecutor
from src.payment.models import Decision, Evaluation, PaymentIntent, PaymentState
from src.store.ledger import Ledger

INTENT = PaymentIntent(
    amount_minor=40000,
    currency="GBP",
    beneficiary="Apple",
    category="equipment",
    justification="MacBook for the new starter",
)


def record_for(decision=Decision.APPROVED):
    ledger = Ledger()
    return ledger.record(INTENT, Evaluation(decision, "because", "a_rule"))


def failing_gateway(record):
    raise ConnectionError("gateway unreachable")


def test_an_approved_payment_executes_and_returns_a_receipt():
    record = record_for()
    receipt = PaymentExecutor().execute(record)
    assert record.state is PaymentState.EXECUTED
    assert record.receipt is receipt
    assert receipt.payment_id == record.id
    assert (receipt.amount_minor, receipt.currency) == (40000, "GBP")


def test_a_held_payment_is_not_sent():
    record = record_for(Decision.REQUIRES_APPROVAL)
    with pytest.raises(PaymentExecutionError, match="nothing was sent"):
        PaymentExecutor().execute(record)
    assert record.state is PaymentState.HELD


def test_a_payment_cannot_be_sent_twice():
    record = record_for()
    PaymentExecutor().execute(record)
    with pytest.raises(PaymentExecutionError):
        PaymentExecutor().execute(record)


def test_a_held_payment_executes_once_a_human_approves():
    record = record_for(Decision.REQUIRES_APPROVAL)
    record.transition_to(PaymentState.APPROVED)
    assert PaymentExecutor().execute(record)
    assert record.state is PaymentState.EXECUTED


def test_a_gateway_failure_leaves_the_payment_failed():
    record = record_for()
    with pytest.raises(PaymentExecutionError, match="unreachable"):
        PaymentExecutor(gateway=failing_gateway).execute(record)
    assert record.state is PaymentState.FAILED
    assert record.failure_reason == "gateway unreachable"
    assert record.receipt is None
