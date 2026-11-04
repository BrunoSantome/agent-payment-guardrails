"""End-to-end demonstration, using the real LLM.

Three scenarios: a payment denied and then revised by the agent, a payment
held for a human and then approved, and a payment denied for being outside the
allowed categories.

Each scenario declares what it expects and the script prints expected against
actual rather than asserting. The policy decisions are deterministic, but the
agent may ask a clarifying question instead of proposing, and a demonstration
that crashes on that is less useful than one that shows it.

Run:  python scripted_run.py
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.agent.intent_agent import IntentAgent
from src.config.settings import Settings, load_api_key, load_settings
from src.guardrails.policy import PolicyEngine
from src.llm.client import GeminiClient
from src.orchestrator import PaymentFlow
from src.payment.executor import PaymentExecutor
from src.payment.models import format_amount
from src.store.ledger import Ledger

SCENARIOS = [
    (
        "Denied, then revised by the agent",
        [
            ("Pay Apple £5,000 for new MacBooks for the design team", "denied"),
            ("Too much. Just one MacBook, £400", "executed"),
        ],
    ),
    (
        "Held for a human, then approved",
        [("Pay British Airways £900 for the Berlin sales trip", "held")],
    ),
    (
        "Denied: category not allowed",
        [("Pay Deloitte £300 for the annual audit", "denied")],
    ),
]


def build(settings: Settings) -> tuple[PaymentFlow, Ledger]:
    client = GeminiClient(
        api_key=load_api_key(settings.llm.provider),
        model=settings.llm.model,
        max_tokens=settings.llm.max_tokens,
    )
    ledger = Ledger()
    flow = PaymentFlow(
        agent=IntentAgent(
            client,
            categories=settings.guardrails.allowed_categories,
            currencies=settings.guardrails.allowed_currencies,
        ),
        policy=PolicyEngine(settings.guardrails),
        ledger=ledger,
        executor=PaymentExecutor(),
    )
    return flow, ledger


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    settings = load_settings()
    flow, ledger = build(settings)
    print("Agentic payment flow - end-to-end demonstration")

    for title, turns in SCENARIOS:
        rule(title)
        for request, expected in turns:
            # Printed before the call: it takes a second or two, and a blank
            # screen looks like a hang.
            print(f"\n  user  > {request}", flush=True)
            result = flow.handle(request)
            note = "as expected" if result.kind == expected else f"EXPECTED {expected}"
            print(f"  system> [{result.kind}] {result.message}")
            print(f"          ({note})")

        # A human clears anything this scenario left waiting.
        for record in ledger.held():
            result = flow.approve(record.id)
            print(f"\n  human > approves {record.id}")
            print(f"  system> [{result.kind}] {result.message}")

    rule("Ledger")
    for record in ledger.all():
        print(
            f"  {record.id}  {record.state.value:<9} "
            f"{format_amount(record.intent.amount_minor, record.intent.currency):>14}  "
            f"{record.intent.beneficiary} ({record.intent.category})"
        )

    # The cumulative guardrail, asked the same way the policy engine asks it:
    # per beneficiary, per currency, over the configured rolling window.
    cumulative = settings.guardrails.cumulative
    window_start = datetime.now(timezone.utc) - cumulative.window
    rule(f"Spend in the last {cumulative.window_hours}h, per {cumulative.scope}")
    for beneficiary in ledger.beneficiaries():
        for currency in sorted(
            {r.intent.currency for r in ledger.for_beneficiary(beneficiary)}
        ):
            spent = ledger.total_spent_minor(
                since=window_start, currency=currency, beneficiary=beneficiary
            )
            print(f"  {beneficiary:<20} {format_amount(spent, currency):>14}")


if __name__ == "__main__":
    main()
