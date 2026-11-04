"""Mocked payment execution.

No real payments API. The gateway is a seam: swap in a provider client and the
rest of the system does not change.
"""

from __future__ import annotations

from datetime import datetime, timezone
from collections.abc import Callable

from src.payment.models import (
    PaymentRecord,
    PaymentState,
    Receipt,
    new_transaction_id,
)


class PaymentExecutionError(Exception):
    """The payment did not go through. The record is left in a terminal state."""


def mock_gateway(record: PaymentRecord) -> str:
    """Stands in for a payment provider. Always succeeds."""
    return new_transaction_id()


class PaymentExecutor:
    def __init__(self, gateway: Callable[[PaymentRecord], str] = mock_gateway) -> None:
        self._gateway = gateway

    def execute(self, record: PaymentRecord) -> Receipt:
        """Send an approved payment and return its receipt.

        Anything not APPROVED is refused: a held payment needs a human first,
        and a denied or already-executed one must never be sent.
        """
        if record.state is not PaymentState.APPROVED:
            raise PaymentExecutionError(
                f"{record.id} is {record.state.value}, not approved - nothing was sent"
            )

        try:
            transaction_id = self._gateway(record)
        except Exception as exc:
            record.mark_failed(str(exc))
            raise PaymentExecutionError(f"{record.id} failed: {exc}") from exc

        receipt = Receipt(
            transaction_id=transaction_id,
            payment_id=record.id,
            amount_minor=record.intent.amount_minor,
            currency=record.intent.currency,
            beneficiary=record.intent.beneficiary,
            executed_at=datetime.now(timezone.utc),
        )
        record.mark_executed(receipt)
        return receipt
