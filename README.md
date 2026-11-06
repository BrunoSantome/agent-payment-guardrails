# Safe Agentic Payment Flow

An LLM turns a request like *"Pay Apple £5,000 for new MacBooks"* into a
structured payment intent. Deterministic code decides whether the payment is
approved, held for a human or denied, and writes every message about what
happened. The model never approves a payment and never reports on one.

## Running it

Requires Python 3.12 and a Gemini API key.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your-key-here" > .env
```

**End-to-end demonstration** (calls Gemini):

```bash
python scripted_run.py
```

**Tests** (178 tests, no network or API key needed):

```bash
python -m pytest
```

The demonstration runs three scenarios:

| Request | Path |
|---|---|
| `Pay Apple £5,000 for new MacBooks`, then `Too much. Just one MacBook, £400` | denied, then revised and executed |
| `Pay British Airways £900 for the Berlin sales trip` | held, then approved (the script plays the approver) and executed |
| `Pay Deloitte £300 for the annual audit` | denied: category not allowed |

It prints the expected outcome next to the actual one, because the agent may
ask a clarifying question instead of proposing. If Gemini is busy (`503`/`504`),
change `llm.model` in `src/config/config.yaml`. Requests time out after 30 seconds.

## Architecture

```
request ─► IntentAgent        Gemini, constrained to a JSON schema    ─► clarify
        ─► validate_intent()  is it a well-formed payment?           ─► invalid
        ─► PolicyEngine       is it allowed?
        ─► Ledger.record()    every decision, including denials      ─► denied / held
        ─► PaymentExecutor    mocked gateway                         ─► executed / failed
```

| Module | Responsibility |
|---|---|
| `src/agent/intent_agent.py` | Prompt and response schema. Proposes an intent or asks a question |
| `src/llm/client.py` | Gemini behind one method, with schema-constrained output |
| `src/guardrails/validation.py` | Structural checks on the intent. No config, no ledger |
| `src/guardrails/policy.py` | The payment decision, from rules in config |
| `src/payment/models.py` | Shared types, money conversion, the payment state machine |
| `src/store/ledger.py` | In-memory record of every evaluated payment, and spend queries |
| `src/payment/executor.py` | Mocked execution behind a swappable gateway |
| `src/orchestrator.py` | Runs one user turn through the steps above |
| `src/config/config.yaml` | Limits, threshold and allow-lists |

- **The policy engine is the only place a decision is made.** It reads the
  config and the ledger, never the LLM, so every rule is tested without a model.
- **One entry point.** `PaymentFlow.handle(text)` returns a
  `TurnResult(kind, message, record)`, where `kind` is `executed`, `held`,
  `denied`, `invalid`, `failed`, `clarify` or `unavailable`. Held payments are
  resolved with `approve(id)` or `reject(id, reason)`.
- **Feedback loop.** Every outcome is added to the conversation as text written
  by the code. On the next turn the agent sees why a payment was denied and can
  propose a revised one. `Too much. Just one MacBook, £400` works because the
  agent carries `Apple`, `equipment` and `GBP` forward from the earlier turns.
- **State machine.** `APPROVED → EXECUTED | FAILED`, `HELD → APPROVED | DENIED`,
  and `DENIED`, `EXECUTED`, `FAILED` are final. Every change goes through
  `PaymentRecord.transition_to`, so a held payment cannot execute without
  approval and nothing executes twice.

## Guardrails

Rules run in order and the first objection wins.

| # | Rule | Config | Outcome |
|---|---|---|---|
| 1 | Currency not allowed | `allowed_currencies: [GBP, EUR, USD]` | denied |
| 2 | Category not allowed | `allowed_categories: [equipment, salary, travel]` | denied |
| 3 | Over the single-payment maximum | `max_transaction_amount: 1000` | denied |
| 4 | Over the rolling cumulative limit | `cumulative: {window_hours: 24, scope: beneficiary, limit: 5000}` | denied |
| 5 | Over the auto-approval threshold | `auto_approval_threshold: 500` | held for approval |
| 6 | Otherwise | | approved and executed |

All deny rules run before the hold rule, so a payment over a hard limit never
reaches a human. Startup rejects a threshold above the maximum and a cumulative
limit below the maximum, since either would quietly disable a rule.

**Cumulative limit:** rolling 24 hours, per beneficiary, per currency. Only
executed payments count, from the time the money left. I chose per beneficiary
because the likeliest agent mistake is over-paying one counterparty. The scope
(`beneficiary`, `category`, `global`) and the window are configurable.

## Assumptions

- Money is stored as integer minor units (pence, cents). The LLM returns amounts
  as strings, which are parsed with `Decimal`. Amounts finer than a penny are
  refused, not rounded.
- No FX conversion. The same limits apply in each currency and totals are kept
  per currency, so a beneficiary's cumulative cap applies once per currency.
- Validation and policy are separate. A malformed intent needs a field
  corrected; a denied one needs a different payment.
- One agent, so no `agent_id`. A per-agent limit is the same as a global one.
- Denied payments are recorded in the ledger, so refusals can be audited.
- After a denial the agent revises only when the user replies. It never retries
  on its own, so it cannot work its way around the policy.
- Approval is a function call, and the demo approves straight away. Who may
  approve is out of scope. Held payments do not block new requests and are not
  re-checked when approved.
- Everything is in memory and single-threaded.

## Extra time: never act on a request that failed

**The problem.** While Gemini was returning `503`, the demo paid Apple £400 when
asked to *"Pay Deloitte £300"*. When the model call failed, `handle()` kept the
user's message in the conversation with no reply. After a few failures in a
row, the model answered an older request instead of the latest one.

**Why this one.** Moving money for a request the user was told had failed is the
worst thing a payment agent can do, and provider outages are routine.

**The fix.** When the model call fails, the request is removed from the
conversation and the user is asked to send it again. Two regression tests cover
it in `test/test_orchestrator.py`.

**The trade-off.** The user has to restate the request after an outage.

## What I would do next

I went past the suggested time, so these are listed rather than built.

1. **An idempotency key on execution.** If the provider times out after paying
   and the payment is retried, it is paid twice.
2. **Reserve budget for held payments, and re-check the policy on approval.**
   Otherwise a payment approved hours later can take a beneficiary over the limit.
3. **Reject duplicate holds** for the same beneficiary, amount and currency.
4. **A lock or database transaction around check-and-record** once more than one
   worker processes payments.
5. **Tighter input handling.** `"1,50"` is read as £150, and `Apple` and `apple`
   count as separate beneficiaries.
