"""One user turn in, one result out.

The only place the layers meet: agent -> validation -> policy -> ledger ->
executor. It contains no rules of its own.

Failures come back as results rather than exceptions, so a caller handles
every outcome the same way. The user-facing text here is written by code, not
by the model: the agent proposes payments, it never reports what happened to
them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.agent.intent_agent import IntentAgent
from src.guardrails.policy import PolicyEngine
from src.guardrails.validation import IntentValidationError, validate_intent
from src.llm.client import LLMError, Message
from src.payment.executor import PaymentExecutionError, PaymentExecutor
from src.payment.models import Decision, PaymentRecord, PaymentState, format_amount
from src.store.ledger import Ledger

Kind = Literal[
    "clarify",  # the agent needs something from the user
    "executed",  # approved and paid, receipt attached
    "held",  # waiting on a human
    "denied",  # refused by policy
    "invalid",  # the model's intent was malformed
    "failed",  # approved, but the gateway did not accept it
    "unavailable",  # the model could not be reached
]


@dataclass(frozen=True)
class TurnResult:
    kind: Kind
    message: str
    record: PaymentRecord | None = None


class PaymentFlow:
    def __init__(
        self,
        agent: IntentAgent,
        policy: PolicyEngine,
        ledger: Ledger,
        executor: PaymentExecutor,
    ) -> None:
        self._agent = agent
        self._policy = policy
        self._ledger = ledger
        self._executor = executor
        self._history: list[Message] = []

    @property
    def history(self) -> tuple[Message, ...]:
        return tuple(self._history)

    def handle(self, user_message: str) -> TurnResult:
        self._history.append(Message("user", user_message))

        try:
            reply = self._agent.respond(self._history)
        except LLMError as exc:
            # Nothing to record: the user's turn stays, so a retry keeps context.
            return TurnResult("unavailable", f"The agent is unavailable: {exc}")

        if reply.action == "clarify" or reply.intent is None:
            return self._respond("clarify", reply.message)

        try:
            intent = validate_intent(reply.intent.model_dump())
        except IntentValidationError as exc:
            return self._respond("invalid", f"That intent cannot be used: {exc}")

        evaluation = self._policy.evaluate(intent, self._ledger)
        record = self._ledger.record(intent, evaluation)

        if evaluation.decision is Decision.DENIED:
            return self._respond("denied", f"Denied: {evaluation.reason}", record)
        if evaluation.decision is Decision.REQUIRES_APPROVAL:
            return self._respond(
                "held",
                f"Held for approval ({record.id}): {evaluation.reason}",
                record,
            )
        return self._execute(record)

    def approve(self, payment_id: str) -> TurnResult:
        """A human approves a held payment, which then executes."""
        record = self._require_held(payment_id)
        record.transition_to(PaymentState.APPROVED)
        return self._execute(record)

    def reject(self, payment_id: str, reason: str) -> TurnResult:
        """A human refuses a held payment. The reason is required.

        It goes into the conversation, so the agent can revise against it the
        same way it revises against a policy denial. The policy's own reason
        for holding the payment is left intact.
        """
        record = self._require_held(payment_id)
        record.transition_to(PaymentState.DENIED)
        record.rejection_reason = reason
        return self._respond("denied", f"Denied by a human: {reason}", record)

    def _execute(self, record: PaymentRecord) -> TurnResult:
        try:
            receipt = self._executor.execute(record)
        except PaymentExecutionError as exc:
            return self._respond("failed", f"Payment failed: {exc}", record)
        return self._respond(
            "executed",
            f"Paid {format_amount(receipt.amount_minor, receipt.currency)} to "
            f"{receipt.beneficiary}. Receipt {receipt.transaction_id}.",
            record,
        )

    def _require_held(self, payment_id: str) -> PaymentRecord:
        record = self._ledger.get(payment_id)
        if record is None:
            raise KeyError(f"No payment {payment_id}")
        if record.state is not PaymentState.HELD:
            raise ValueError(f"{payment_id} is {record.state.value}, not held")
        return record

    def _respond(
        self, kind: Kind, message: str, record: PaymentRecord | None = None
    ) -> TurnResult:
        # The outcome goes into the conversation so the agent can act on it -
        # this is what lets a denied payment be revised on the next turn.
        self._history.append(Message("model", message))
        return TurnResult(kind, message, record)
