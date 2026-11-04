"""The vocabulary shared by the guardrails, the store and the executor.

Nothing in here talks to an LLM, a policy file or a store. Keeping the types
in one place means those three layers depend on this module and not on each
other.

Money is always an integer count of minor units (pence, cents). Major-unit
decimals exist only at the edges: configuration coming in, and text going out.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

# Minor units per major unit, for the currencies this system can reason about.
# Anything absent is rejected rather than assumed to have two decimal places.
CURRENCY_EXPONENT = {"GBP": 2, "EUR": 2, "USD": 2}


class UnsupportedCurrency(Exception):
    """A currency with no known minor-unit exponent."""


class IllegalTransition(Exception):
    """An attempt to move a payment to a state it cannot reach from here.

    This is the guard against executing a held payment twice, or executing one
    a human has rejected.
    """


def exponent_for(currency: str) -> int:
    try:
        return CURRENCY_EXPONENT[currency]
    except KeyError:
        raise UnsupportedCurrency(
            f"Unsupported currency '{currency}'. "
            f"Supported: {', '.join(sorted(CURRENCY_EXPONENT))}."
        ) from None


def to_minor_units(amount: Decimal, currency: str) -> int:
    """Convert a major-unit amount to minor units, refusing to round.

    An amount finer than the currency's smallest unit is a caller error, not
    something to silently truncate.
    """
    minor = Decimal(amount).scaleb(exponent_for(currency))
    if minor != minor.to_integral_value():
        raise ValueError(
            f"{amount} {currency} is finer than the smallest unit of {currency}"
        )
    return int(minor)


def to_major_units(amount_minor: int, currency: str) -> Decimal:
    return Decimal(amount_minor).scaleb(-exponent_for(currency))


def format_amount(amount_minor: int, currency: str) -> str:
    exponent = exponent_for(currency)
    return f"{to_major_units(amount_minor, currency):,.{exponent}f} {currency}"


def new_payment_id() -> str:
    return f"pay_{uuid.uuid4().hex[:10]}"


def new_transaction_id() -> str:
    return f"txn_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class PaymentIntent:
    """A normalised payment request, ready to be evaluated.

    Being a valid intent says nothing about whether it is allowed: that is the
    policy engine's decision.
    """

    amount_minor: int
    currency: str
    beneficiary: str
    category: str
    justification: str


class Decision(str, Enum):
    """The three outcomes of policy evaluation, as the task names them."""

    APPROVED = "approved"
    REQUIRES_APPROVAL = "requires_approval"
    DENIED = "denied"


@dataclass(frozen=True)
class Evaluation:
    """What the policy engine returns.

    `reason` is prose for a human; `rule` names the constraint that fired, so
    callers can react to the decision without string-matching the prose.
    """

    decision: Decision
    reason: str
    rule: str


class PaymentState(str, Enum):
    """Where a payment has got to.

    APPROVED is distinct from EXECUTED on purpose: a payment that passed policy
    but has not yet been through the executor is not a payment that was made.
    """

    APPROVED = "approved"
    HELD = "held"
    DENIED = "denied"
    EXECUTED = "executed"
    FAILED = "failed"


# Every move a payment is allowed to make. Absent or empty means terminal.
ALLOWED_TRANSITIONS: dict[PaymentState, frozenset[PaymentState]] = {
    PaymentState.APPROVED: frozenset({PaymentState.EXECUTED, PaymentState.FAILED}),
    # A human either approves a held payment (it then goes to the executor) or
    # rejects it.
    PaymentState.HELD: frozenset({PaymentState.APPROVED, PaymentState.DENIED}),
    PaymentState.DENIED: frozenset(),
    PaymentState.EXECUTED: frozenset(),
    PaymentState.FAILED: frozenset(),
}

# States that count towards cumulative spend: money that actually left.
SPENT_STATES = frozenset({PaymentState.EXECUTED})


@dataclass(frozen=True)
class Receipt:
    """Proof that a (mocked) payment was made."""

    transaction_id: str
    payment_id: str
    amount_minor: int
    currency: str
    beneficiary: str
    executed_at: datetime


@dataclass
class PaymentRecord:
    """A ledger entry. Denied payments get one too, so refusals are auditable."""

    intent: PaymentIntent
    decision: Decision
    reason: str
    state: PaymentState
    id: str = field(default_factory=new_payment_id)
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    receipt: Receipt | None = None
    failure_reason: str | None = None

    def transition_to(self, state: PaymentState) -> None:
        """Move to `state`, or raise if that move is not allowed from here."""
        allowed = ALLOWED_TRANSITIONS.get(self.state, frozenset())
        if state not in allowed:
            permitted = ", ".join(sorted(s.value for s in allowed)) or "nothing"
            raise IllegalTransition(
                f"{self.id} is {self.state.value}; it cannot become "
                f"{state.value} (allowed: {permitted})"
            )
        self.state = state
        self.updated_at = _now()

    def mark_executed(self, receipt: Receipt) -> None:
        """Move to EXECUTED and attach the receipt: both, or neither.

        Transition first, so an illegal move leaves the record untouched.
        """
        self.transition_to(PaymentState.EXECUTED)
        self.receipt = receipt

    def mark_failed(self, reason: str) -> None:
        self.transition_to(PaymentState.FAILED)
        self.failure_reason = reason

    def counts_towards_spend(self) -> bool:
        return self.state in SPENT_STATES

    @property
    def spent_at(self) -> datetime | None:
        """When the money actually left, which is what a rolling window means.

        A payment proposed at 11:00 and approved by a human at 15:00 belongs to
        the 15:00 window, not the 11:00 one.
        """
        return self.receipt.executed_at if self.receipt else None
