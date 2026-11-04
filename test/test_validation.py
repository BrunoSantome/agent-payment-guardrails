"""Structural validation of model-produced intents."""

import pytest

from src.guardrails.validation import IntentValidationError, validate_intent
from src.payment.models import PaymentIntent

VALID_DRAFT = {
    "amount": "2499.50",
    "currency": "GBP",
    "beneficiary": "Apple",
    "category": "equipment",
    "justification": "MacBook for the new starter",
}


def draft(**overrides):
    return {**VALID_DRAFT, **overrides}


def test_a_valid_draft_becomes_an_intent():
    intent = validate_intent(VALID_DRAFT)
    assert intent == PaymentIntent(
        amount_minor=249950,
        currency="GBP",
        beneficiary="Apple",
        category="equipment",
        justification="MacBook for the new starter",
    )


# --- lenient about format --------------------------------------------------


@pytest.mark.parametrize(
    "amount, expected",
    [
        ("£5,000", 500000),
        ("5000", 500000),
        (" 5 000 ", 500000),
        ("$49.99", 4999),
        ("€1,234.56", 123456),
        (5000, 500000),
        (49.99, 4999),
    ],
)
def test_amount_formats_are_parsed_not_refused(amount, expected):
    """Refusing a real payment over a currency symbol is a bad failure."""
    assert validate_intent(draft(amount=amount)).amount_minor == expected


def test_currency_and_category_are_normalised():
    intent = validate_intent(draft(currency=" gbp ", category=" Equipment "))
    assert intent.currency == "GBP"
    assert intent.category == "equipment"


def test_whitespace_in_text_is_collapsed():
    intent = validate_intent(draft(beneficiary="  British   Airways \n"))
    assert intent.beneficiary == "British Airways"


def test_control_characters_are_stripped():
    intent = validate_intent(draft(beneficiary="Apple\x00\x1b[31m"))
    assert intent.beneficiary == "Apple[31m"


# --- strict about value ----------------------------------------------------


@pytest.mark.parametrize("amount", ["0", "0.00", "-5", "-0.01"])
def test_non_positive_amounts_are_refused(amount):
    with pytest.raises(IntentValidationError, match="greater than zero"):
        validate_intent(draft(amount=amount))


def test_sub_penny_amount_is_refused_not_rounded():
    with pytest.raises(IntentValidationError, match="smallest unit"):
        validate_intent(draft(amount="10.001"))


@pytest.mark.parametrize("amount", ["lots", "", "£", "12.3.4", "1e", "NaN", "Infinity"])
def test_unparseable_amounts_are_refused(amount):
    with pytest.raises(IntentValidationError, match="Amount"):
        validate_intent(draft(amount=amount))


def test_boolean_amount_is_refused():
    """bool is a subclass of int, so True would otherwise parse as 1."""
    with pytest.raises(IntentValidationError, match="boolean"):
        validate_intent(draft(amount=True))


def test_unknown_currency_is_refused():
    with pytest.raises(IntentValidationError, match="JPY"):
        validate_intent(draft(currency="JPY"))


# --- missing and malformed fields ------------------------------------------


@pytest.mark.parametrize("field", list(VALID_DRAFT))
def test_a_missing_field_names_itself(field):
    incomplete = {k: v for k, v in VALID_DRAFT.items() if k != field}
    with pytest.raises(IntentValidationError, match=field):
        validate_intent(incomplete)


@pytest.mark.parametrize("field", list(VALID_DRAFT))
def test_a_null_field_names_itself(field):
    with pytest.raises(IntentValidationError, match=field):
        validate_intent(draft(**{field: None}))


@pytest.mark.parametrize("field", ["beneficiary", "category", "justification"])
def test_a_whitespace_only_field_is_empty(field):
    with pytest.raises(IntentValidationError, match="empty"):
        validate_intent(draft(**{field: "   \n\t "}))


def test_a_structured_value_is_refused():
    with pytest.raises(IntentValidationError, match="must be text"):
        validate_intent(draft(beneficiary={"name": "Apple"}))


@pytest.mark.parametrize(
    "field, length", [("beneficiary", 121), ("category", 61), ("justification", 501)]
)
def test_overlong_text_is_refused(field, length):
    """These strings reach the ledger and the receipt."""
    with pytest.raises(IntentValidationError, match="limit"):
        validate_intent(draft(**{field: "x" * length}))


@pytest.mark.parametrize(
    "field, length", [("beneficiary", 120), ("category", 60), ("justification", 500)]
)
def test_text_at_the_limit_is_accepted(field, length):
    assert validate_intent(draft(**{field: "x" * length}))


def test_validation_needs_no_configuration():
    """It is a pure function: the same draft always gives the same intent."""
    assert validate_intent(VALID_DRAFT) == validate_intent(dict(VALID_DRAFT))
