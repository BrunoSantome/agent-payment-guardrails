"""The ledger: what it records, and what counts towards spend."""

from datetime import datetime, timedelta, timezone

import pytest

from src.payment.models import (
    Decision,
    Evaluation,
    IllegalTransition,
    PaymentIntent,
    PaymentState,
    Receipt,
    new_transaction_id,
)
from src.store.ledger import Ledger

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
WINDOW_START = NOW - timedelta(hours=24)


def intent(amount_minor=40000, currency="GBP", beneficiary="Apple", category="equipment"):
    return PaymentIntent(
        amount_minor=amount_minor,
        currency=currency,
        beneficiary=beneficiary,
        category=category,
        justification="MacBook for the new starter",
    )


def evaluation(decision=Decision.APPROVED):
    return Evaluation(decision=decision, reason="because", rule="a_rule")


def execute(ledger, record, *, at=NOW):
    """Move a record to EXECUTED with a receipt, the way the executor will."""
    record.receipt = Receipt(
        transaction_id=new_transaction_id(),
        payment_id=record.id,
        amount_minor=record.intent.amount_minor,
        currency=record.intent.currency,
        beneficiary=record.intent.beneficiary,
        executed_at=at,
    )
    record.transition_to(PaymentState.EXECUTED)
    return record


def spent(ledger, **overrides):
    query = {"since": WINDOW_START, "currency": "GBP"}
    return ledger.total_spent_minor(**{**query, **overrides})


# --- recording -------------------------------------------------------------


@pytest.mark.parametrize(
    "decision, state",
    [
        (Decision.APPROVED, PaymentState.APPROVED),
        (Decision.REQUIRES_APPROVAL, PaymentState.HELD),
        (Decision.DENIED, PaymentState.DENIED),
    ],
)
def test_the_initial_state_follows_the_decision(decision, state):
    ledger = Ledger()
    record = ledger.record(intent(), evaluation(decision))
    assert record.state is state


def test_an_approved_payment_is_not_yet_executed():
    """Passing policy is not the same as the money having moved."""
    ledger = Ledger()
    record = ledger.record(intent(), evaluation(Decision.APPROVED))
    assert record.state is PaymentState.APPROVED
    assert record.receipt is None


def test_denied_payments_are_recorded_for_audit():
    ledger = Ledger()
    ledger.record(intent(), evaluation(Decision.DENIED))
    assert len(ledger.all()) == 1
    assert ledger.all()[0].decision is Decision.DENIED


def test_the_reason_is_carried_onto_the_record():
    ledger = Ledger()
    record = ledger.record(
        intent(), Evaluation(Decision.DENIED, "over the maximum", "max_transaction_amount")
    )
    assert record.reason == "over the maximum"


def test_records_are_retrievable_by_id():
    ledger = Ledger()
    record = ledger.record(intent(), evaluation())
    assert ledger.get(record.id) is record
    assert ledger.get("pay_nonexistent") is None


def test_held_payments_are_listed_separately():
    ledger = Ledger()
    ledger.record(intent(), evaluation(Decision.APPROVED))
    held = ledger.record(intent(), evaluation(Decision.REQUIRES_APPROVAL))
    ledger.record(intent(), evaluation(Decision.DENIED))
    assert ledger.held() == (held,)


def test_entries_are_never_removed():
    ledger = Ledger()
    for _ in range(3):
        ledger.record(intent(), evaluation(Decision.DENIED))
    assert len(ledger.all()) == 3


def test_the_ledger_cannot_route_around_the_state_machine():
    """It only ever appends; transitions stay the record's business."""
    ledger = Ledger()
    record = ledger.record(intent(), evaluation(Decision.REQUIRES_APPROVAL))
    with pytest.raises(IllegalTransition):
        record.transition_to(PaymentState.EXECUTED)


# --- what counts towards spend ---------------------------------------------


def test_nothing_is_spent_by_an_empty_ledger():
    assert spent(Ledger()) == 0


def test_an_executed_payment_counts():
    ledger = Ledger()
    execute(ledger, ledger.record(intent(40000), evaluation()))
    assert spent(ledger, beneficiary="Apple") == 40000


@pytest.mark.parametrize(
    "decision", [Decision.APPROVED, Decision.REQUIRES_APPROVAL, Decision.DENIED]
)
def test_an_unexecuted_payment_spends_nothing(decision):
    """Proposing, holding or refusing a payment moves no money."""
    ledger = Ledger()
    ledger.record(intent(40000), evaluation(decision))
    assert spent(ledger) == 0


def test_a_held_payment_starts_counting_once_executed():
    ledger = Ledger()
    record = ledger.record(intent(90000), evaluation(Decision.REQUIRES_APPROVAL))
    assert spent(ledger) == 0
    record.transition_to(PaymentState.APPROVED)  # a human approves
    execute(ledger, record)
    assert spent(ledger) == 90000


def test_a_failed_payment_spends_nothing():
    ledger = Ledger()
    record = ledger.record(intent(40000), evaluation())
    record.transition_to(PaymentState.FAILED)
    assert spent(ledger) == 0


def test_totals_add_up_across_payments():
    ledger = Ledger()
    for amount in (10000, 25000, 5000):
        execute(ledger, ledger.record(intent(amount), evaluation()))
    assert spent(ledger) == 40000


# --- the window ------------------------------------------------------------


def test_spend_is_measured_from_when_the_money_left():
    """A payment held at 11:00 and executed at 15:00 belongs to 15:00."""
    ledger = Ledger()
    record = ledger.record(intent(40000), evaluation(Decision.REQUIRES_APPROVAL))
    record.created_at = NOW - timedelta(hours=30)  # proposed before the window
    record.transition_to(PaymentState.APPROVED)
    execute(ledger, record, at=NOW)
    assert spent(ledger) == 40000


def test_spend_outside_the_window_is_excluded():
    ledger = Ledger()
    execute(ledger, ledger.record(intent(40000), evaluation()), at=NOW - timedelta(hours=25))
    assert spent(ledger) == 0


def test_the_window_start_is_inclusive():
    ledger = Ledger()
    execute(ledger, ledger.record(intent(40000), evaluation()), at=WINDOW_START)
    assert spent(ledger) == 40000


# --- scope filters ---------------------------------------------------------


def test_totals_are_isolated_per_currency():
    ledger = Ledger()
    execute(ledger, ledger.record(intent(40000, currency="GBP"), evaluation()))
    execute(ledger, ledger.record(intent(90000, currency="EUR"), evaluation()))
    assert spent(ledger, currency="GBP") == 40000
    assert spent(ledger, currency="EUR") == 90000


def test_beneficiary_scope_excludes_other_beneficiaries():
    ledger = Ledger()
    execute(ledger, ledger.record(intent(40000, beneficiary="Apple"), evaluation()))
    execute(ledger, ledger.record(intent(90000, beneficiary="Dell"), evaluation()))
    assert spent(ledger, beneficiary="Apple") == 40000


def test_category_scope_excludes_other_categories():
    ledger = Ledger()
    execute(ledger, ledger.record(intent(40000, category="equipment"), evaluation()))
    execute(ledger, ledger.record(intent(90000, category="travel"), evaluation()))
    assert spent(ledger, category="travel") == 90000


def test_global_scope_counts_every_beneficiary():
    ledger = Ledger()
    execute(ledger, ledger.record(intent(40000, beneficiary="Apple"), evaluation()))
    execute(ledger, ledger.record(intent(90000, beneficiary="Dell"), evaluation()))
    assert spent(ledger) == 130000


def test_scopes_combine():
    ledger = Ledger()
    execute(ledger, ledger.record(intent(40000, beneficiary="Apple", category="equipment"), evaluation()))
    execute(ledger, ledger.record(intent(90000, beneficiary="Apple", category="travel"), evaluation()))
    assert spent(ledger, beneficiary="Apple", category="travel") == 90000


# --- queries and getters ---------------------------------------------------


def populated() -> Ledger:
    """Five payments across two beneficiaries, two categories, three decisions."""
    ledger = Ledger()
    ledger.record(intent(10000, beneficiary="Apple", category="equipment"), evaluation())
    ledger.record(
        intent(90000, beneficiary="Apple", category="travel"),
        evaluation(Decision.REQUIRES_APPROVAL),
    )
    ledger.record(
        intent(500000, beneficiary="Dell", category="equipment"),
        evaluation(Decision.DENIED),
    )
    ledger.record(intent(20000, beneficiary="Dell", category="travel"), evaluation())
    ledger.record(
        intent(30000, beneficiary="Apple", category="equipment", currency="EUR"),
        evaluation(),
    )
    return ledger


def test_length_is_the_number_of_records():
    assert len(populated()) == 5


def test_all_returns_every_record_in_order():
    ledger = populated()
    assert ledger.all() == ledger.records()
    assert [r.intent.amount_minor for r in ledger.all()] == [
        10000,
        90000,
        500000,
        20000,
        30000,
    ]


def test_filter_by_beneficiary():
    assert len(populated().for_beneficiary("Apple")) == 3


def test_filter_by_category():
    assert len(populated().for_category("travel")) == 2


def test_filter_by_state():
    assert len(populated().by_state(PaymentState.HELD)) == 1


def test_filter_by_decision():
    assert len(populated().by_decision(Decision.DENIED)) == 1


def test_filters_combine():
    ledger = populated()
    matches = ledger.records(beneficiary="Apple", category="equipment")
    assert [r.intent.currency for r in matches] == ["GBP", "EUR"]
    assert len(ledger.records(beneficiary="Apple", category="equipment", currency="GBP")) == 1


def test_no_filters_returns_everything():
    assert len(populated().records()) == 5


def test_an_unmatched_filter_returns_nothing():
    assert populated().for_beneficiary("Google") == ()


def test_created_since_filters_on_when_it_was_proposed():
    """Distinct from the spend window, which filters on when money left."""
    ledger = Ledger()
    old = ledger.record(intent(), evaluation())
    old.created_at = NOW - timedelta(hours=48)
    recent = ledger.record(intent(), evaluation())
    recent.created_at = NOW
    assert ledger.records(created_since=NOW - timedelta(hours=24)) == (recent,)


def test_distinct_beneficiaries_in_first_seen_order():
    assert populated().beneficiaries() == ("Apple", "Dell")


def test_distinct_categories_in_first_seen_order():
    assert populated().categories() == ("equipment", "travel")


def test_distinct_lists_are_empty_for_an_empty_ledger():
    assert Ledger().beneficiaries() == ()
    assert Ledger().categories() == ()
