"""The ledger: every payment the system has considered, and spend queries over it.

Append-only - entries are never removed - and in-memory, which is enough for a
single agent acting on one payment at a time. Two consequences are deliberate
and noted in the README: a record's state mutates in place rather than being
an event log of transitions, and the store is not thread-safe.

The ledger stores and queries. It does not decide anything, and it never
assigns `state` directly: every change after the first goes through
PaymentRecord.transition_to, so the state machine cannot be routed around.
"""

from __future__ import annotations

from datetime import datetime

from src.payment.models import (
    Decision,
    Evaluation,
    PaymentIntent,
    PaymentRecord,
    PaymentState,
)

# The state a payment starts in, given what the policy engine decided.
INITIAL_STATE = {
    Decision.APPROVED: PaymentState.APPROVED,
    Decision.REQUIRES_APPROVAL: PaymentState.HELD,
    Decision.DENIED: PaymentState.DENIED,
}


class Ledger:
    def __init__(self) -> None:
        self._records: list[PaymentRecord] = []

    def record(self, intent: PaymentIntent, evaluation: Evaluation) -> PaymentRecord:
        """Append an entry for a payment that has just been evaluated.

        Denied payments are recorded too: a system that forgets what it refused
        cannot be audited.
        """
        record = PaymentRecord(
            intent=intent,
            decision=evaluation.decision,
            reason=evaluation.reason,
            state=INITIAL_STATE[evaluation.decision],
        )
        self._records.append(record)
        return record

    def get(self, payment_id: str) -> PaymentRecord | None:
        for record in self._records:
            if record.id == payment_id:
                return record
        return None

    def all(self) -> tuple[PaymentRecord, ...]:
        return self.records()

    def __len__(self) -> int:
        return len(self._records)

    def records(
        self,
        *,
        beneficiary: str | None = None,
        category: str | None = None,
        currency: str | None = None,
        state: PaymentState | None = None,
        decision: Decision | None = None,
        created_since: datetime | None = None,
    ) -> tuple[PaymentRecord, ...]:
        """Records matching every filter given, in the order they were recorded.

        One query rather than a getter per scope: a new dimension becomes a new
        keyword here instead of another near-identical loop.

        `created_since` filters on when a payment was *proposed*. That is a
        different question from total_spent_minor's `since`, which filters on
        when the money *left* - conflating the two is how a rolling limit gets
        miscounted.
        """
        return tuple(
            record
            for record in self._records
            if (beneficiary is None or record.intent.beneficiary == beneficiary)
            and (category is None or record.intent.category == category)
            and (currency is None or record.intent.currency == currency)
            and (state is None or record.state is state)
            and (decision is None or record.decision is decision)
            and (created_since is None or record.created_at >= created_since)
        )

    # Named cases, for readability at the call site.

    def for_beneficiary(self, beneficiary: str) -> tuple[PaymentRecord, ...]:
        return self.records(beneficiary=beneficiary)

    def for_category(self, category: str) -> tuple[PaymentRecord, ...]:
        return self.records(category=category)

    def by_state(self, state: PaymentState) -> tuple[PaymentRecord, ...]:
        return self.records(state=state)

    def by_decision(self, decision: Decision) -> tuple[PaymentRecord, ...]:
        return self.records(decision=decision)

    def held(self) -> tuple[PaymentRecord, ...]:
        """Payments waiting on a human."""
        return self.records(state=PaymentState.HELD)

    def beneficiaries(self) -> tuple[str, ...]:
        """Distinct beneficiaries seen, in first-seen order."""
        return tuple(dict.fromkeys(r.intent.beneficiary for r in self._records))

    def categories(self) -> tuple[str, ...]:
        """Distinct categories seen, in first-seen order."""
        return tuple(dict.fromkeys(r.intent.category for r in self._records))

    def total_spent_minor(
        self,
        *,
        since: datetime,
        currency: str,
        beneficiary: str | None = None,
        category: str | None = None,
    ) -> int:
        """Minor units that left the account since `since`, within the scope.

        Implements the SpendLedger protocol the policy engine depends on. Only
        executed payments count: proposing, holding or being refused a payment
        does not spend anything.
        """
        return sum(
            record.intent.amount_minor
            for record in self._records
            if self._counts(
                record,
                since=since,
                currency=currency,
                beneficiary=beneficiary,
                category=category,
            )
        )

    @staticmethod
    def _counts(
        record: PaymentRecord,
        *,
        since: datetime,
        currency: str,
        beneficiary: str | None,
        category: str | None,
    ) -> bool:
        if not record.counts_towards_spend():
            return False
        if record.spent_at is None:
            # Executed with no receipt is a bug in the executor. Raising is
            # the safe direction: silently skipping it would under-count spend
            # and let more money out than the limit allows.
            raise ValueError(
                f"{record.id} is {record.state.value} but has no receipt, so "
                "the time the money left is unknown"
            )
        if record.intent.currency != currency:
            return False
        if beneficiary is not None and record.intent.beneficiary != beneficiary:
            return False
        if category is not None and record.intent.category != category:
            return False
        return record.spent_at >= since
