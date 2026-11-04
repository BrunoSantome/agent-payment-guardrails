"""The whole flow, end to end, against a fake LLM."""

from decimal import Decimal

import pytest

from src.agent.intent_agent import AgentReply, IntentAgent, IntentDraft
from src.config.settings import CumulativeLimit, GuardrailSettings
from src.guardrails.policy import PolicyEngine
from src.llm.client import LLMError
from src.orchestrator import PaymentFlow
from src.payment.executor import PaymentExecutor
from src.payment.models import PaymentState
from src.store.ledger import Ledger

GUARDRAILS = GuardrailSettings(
    max_transaction_amount=Decimal("1000"),
    auto_approval_threshold=Decimal("500"),
    cumulative=CumulativeLimit(window_hours=24, scope="beneficiary", limit=Decimal("5000")),
    allowed_categories=("equipment", "travel"),
    allowed_currencies=("GBP",),
)


class ScriptedLLM:
    """Replies in order. `None` in the script raises, standing in for an outage."""

    def __init__(self, *replies) -> None:
        self._replies = list(replies)

    def structured(self, **kwargs):
        reply = self._replies.pop(0)
        if reply is None:
            raise LLMError("transport exploded")
        return reply


def propose(amount, beneficiary="Apple", category="equipment", currency="GBP") -> AgentReply:
    return AgentReply(
        action="propose",
        intent=IntentDraft(
            amount=amount,
            currency=currency,
            beneficiary=beneficiary,
            category=category,
            justification="Kit for the new starter",
        ),
        message=f"Paying {beneficiary} {amount} {currency}.",
    )


def flow(*replies, gateway=None) -> PaymentFlow:
    ledger = Ledger()
    executor = PaymentExecutor(gateway) if gateway else PaymentExecutor()
    return PaymentFlow(
        agent=IntentAgent(
            ScriptedLLM(*replies),
            categories=GUARDRAILS.allowed_categories,
            currencies=GUARDRAILS.allowed_currencies,
        ),
        policy=PolicyEngine(GUARDRAILS),
        ledger=ledger,
        executor=executor,
    )


# --- the three required paths ----------------------------------------------


def test_a_payment_within_limits_is_executed():
    result = flow(propose("400")).handle("Buy a monitor from Apple for £400")
    assert result.kind == "executed"
    assert result.record.state is PaymentState.EXECUTED
    assert result.record.receipt.transaction_id in result.message


def test_a_payment_over_the_maximum_is_denied_then_revised():
    """The feedback loop: the denial reason is in the agent's context."""
    payment_flow = flow(propose("5000"), propose("400"))

    denied = payment_flow.handle("Pay Apple £5,000 for MacBooks")
    assert denied.kind == "denied"
    assert denied.record.state is PaymentState.DENIED

    revised = payment_flow.handle("Just one laptop then")
    assert revised.kind == "executed"

    # The agent saw why the first attempt failed.
    assert any("Denied" in m.text for m in payment_flow.history)


def test_a_payment_above_the_threshold_is_held_and_not_executed():
    payment_flow = flow(propose("900"))
    result = payment_flow.handle("Pay Apple £900 for a laptop")
    assert result.kind == "held"
    assert result.record.state is PaymentState.HELD
    assert result.record.receipt is None


def test_a_held_payment_executes_once_a_human_approves():
    payment_flow = flow(propose("900"))
    held = payment_flow.handle("Pay Apple £900 for a laptop")
    approved = payment_flow.approve(held.record.id)
    assert approved.kind == "executed"
    assert held.record.state is PaymentState.EXECUTED


def test_a_held_payment_a_human_rejects_is_never_executed():
    payment_flow = flow(propose("900"))
    held = payment_flow.handle("Pay Apple £900 for a laptop")
    rejected = payment_flow.reject(held.record.id, "Not budgeted this quarter")
    assert rejected.kind == "denied"
    assert held.record.state is PaymentState.DENIED
    assert held.record.receipt is None


def test_a_human_rejection_keeps_both_reasons():
    """The human's reason must not destroy why policy held it."""
    payment_flow = flow(propose("900"))
    held = payment_flow.handle("Pay Apple £900 for a laptop")
    payment_flow.reject(held.record.id, "Not budgeted this quarter")
    assert held.record.rejection_reason == "Not budgeted this quarter"
    assert "auto-approval threshold" in held.record.reason


def test_the_rejection_reason_reaches_the_agent():
    """So a rejected payment can be revised, like a denied one."""
    payment_flow = flow(propose("900"))
    held = payment_flow.handle("Pay Apple £900 for a laptop")
    payment_flow.reject(held.record.id, "Not budgeted this quarter")
    assert any("Not budgeted this quarter" in m.text for m in payment_flow.history)


# --- the other outcomes ----------------------------------------------------


def test_an_ambiguous_request_asks_a_question_and_records_nothing():
    payment_flow = flow(AgentReply(action="clarify", message="Which beneficiary?"))
    result = payment_flow.handle("pay them the usual")
    assert result.kind == "clarify"
    assert result.record is None


def test_a_malformed_intent_is_reported_as_invalid_not_denied():
    """Malformed and denied are different failures with different remedies."""
    result = flow(propose("not-a-number")).handle("pay Apple something")
    assert result.kind == "invalid"
    assert result.record is None


def test_an_llm_outage_is_a_result_not_an_exception():
    result = flow(None).handle("Pay Apple £400")
    assert result.kind == "unavailable"


def test_a_gateway_failure_leaves_the_payment_failed():
    def broken(record):
        raise ConnectionError("gateway down")

    result = flow(propose("400"), gateway=broken).handle("Pay Apple £400")
    assert result.kind == "failed"
    assert result.record.state is PaymentState.FAILED


def test_approving_an_unknown_payment_raises():
    with pytest.raises(KeyError):
        flow().approve("pay_nope")


def test_approving_a_payment_that_is_not_held_raises():
    payment_flow = flow(propose("400"))
    executed = payment_flow.handle("Pay Apple £400")
    with pytest.raises(ValueError, match="not held"):
        payment_flow.approve(executed.record.id)
