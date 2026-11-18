# Agent Payment Guardrails

**An AI agent that can propose payments, and a deterministic system that decides
whether they happen.**

Letting an LLM move money is risky: it can misread a request, invent an amount,
or tell a user a payment went through when it did not. This project keeps the
model to the one thing it is good at, turning natural language into a structured
payment request, and puts every decision behind rules written in code.

- The LLM **proposes**. It never approves, executes or refuses a payment.
- A **policy engine** decides: approve, hold for a human, or deny.
- Every message about a payment is **written by code**, so the model cannot
  misreport what happened.
- A denial goes back into the conversation, so the agent can **revise** the
  payment on the next turn.

Python 3.12 · Google Gemini (structured output) · Pydantic · pytest (178 tests)

## See it work

A real run of `python scripted_run.py` against Gemini:

```
user  > Pay Apple £5,000 for new MacBooks for the design team
system> [denied] Denied: 5,000.00 GBP exceeds the 1,000.00 GBP maximum for a single payment.

user  > Too much. Just one MacBook, £400
system> [executed] Paid 400.00 GBP to Apple. Receipt txn_2d11c1a8beb7.

user  > Pay British Airways £900 for the Berlin sales trip
system> [held] Held for approval (pay_bba04e3d23): 900.00 GBP is above the 500.00 GBP
        auto-approval threshold and needs a human to approve it.

human > approves pay_bba04e3d23
system> [executed] Paid 900.00 GBP to British Airways. Receipt txn_d00f763ae6d1.

user  > Pay Deloitte £300 for the annual audit
system> [denied] Denied: Category 'professional services' is not permitted.
        Allowed categories: equipment, salary, travel.
```

Three things to notice:

1. **The agent revises after a denial.** *"Too much. Just one MacBook, £400"*
   names no payee, category or currency. The agent carries Apple, equipment and
   GBP forward from the conversation, including the denial it just received.
2. **A payment above £500 waits for a human** and does not execute until it is
   approved.
3. **Every message is produced by code**, from the policy's decision and the
   payment record, not written by the model.

## How it works

```
request ─► IntentAgent        Gemini, constrained to a JSON schema    ─► clarify
        ─► validate_intent()  is it a well-formed payment?           ─► invalid
        ─► PolicyEngine       is it allowed?
        ─► Ledger.record()    every decision, including denials      ─► denied / held
        ─► PaymentExecutor    payment gateway (mocked)               ─► executed / failed
```

| Module | Responsibility |
|---|---|
| `src/agent/intent_agent.py` | Prompt and response schema. Proposes a payment or asks a question |
| `src/llm/client.py` | Gemini behind one method, with schema-constrained output |
| `src/guardrails/validation.py` | Structural checks on the proposed payment |
| `src/guardrails/policy.py` | The payment decision, from rules in config |
| `src/payment/models.py` | Shared types, money conversion, the payment state machine |
| `src/store/ledger.py` | Record of every evaluated payment, and spend queries |
| `src/payment/executor.py` | Execution behind a swappable gateway (mocked) |
| `src/orchestrator.py` | Runs one user turn through the steps above |
| `src/config/config.yaml` | Limits, threshold and allow-lists |

`PaymentFlow.handle(text)` is the single entry point. It returns a
`TurnResult(kind, message, record)`, where `kind` is `executed`, `held`,
`denied`, `invalid`, `failed`, `clarify` or `unavailable`, and it never raises
on the user path. Held payments are resolved with `approve(id)` or
`reject(id, reason)`.

## Guardrails

Rules run in order and the first objection wins.

| # | Rule | Config | Outcome |
|---|---|---|---|
| 1 | Currency not allowed | `allowed_currencies: [GBP, EUR, USD]` | denied |
| 2 | Category not allowed | `allowed_categories: [equipment, salary, travel]` | denied |
| 3 | Over the single-payment maximum | `max_transaction_amount: 1000` | denied |
| 4 | Over the rolling cumulative limit | `cumulative: {window_hours: 24, scope: beneficiary, limit: 5000}` | denied |
| 5 | Over the auto-approval threshold | `auto_approval_threshold: 500` | held for a human |
| 6 | Otherwise | | approved and executed |

All deny rules run before the hold rule, so a payment that breaks a hard limit
never reaches a human: the approval step cannot become a way around the policy.

The cumulative limit is a rolling 24 hours, per beneficiary, per currency, and
counts only money that actually left. Its scope (`beneficiary`, `category`,
`global`) and window are configurable, and the config is validated at startup:
a threshold above the maximum, or a cumulative limit below it, is rejected
because either would quietly disable a rule.

## Design decisions

- **The model never decides and never narrates.** A model that writes outcome
  messages will eventually say "paid" for a payment that is only held. Taking it
  off that path removes the failure instead of prompting against it.
- **Schema-constrained output.** Gemini must answer in a Pydantic schema, so
  there is no JSON parsing or repair code. Amounts come back as strings and are
  parsed with `Decimal`; the model never does arithmetic.
- **Money as integer minor units.** Pence and cents, never floats, so totals
  and limit checks are exact. An amount finer than a penny is refused, not
  rounded.
- **Validation is separate from policy.** A malformed proposal needs a field
  fixed; a denied one needs a different payment. Validation is a pure function
  with no config.
- **A state machine on the payment record.** `APPROVED → EXECUTED | FAILED`,
  `HELD → APPROVED | DENIED`, everything else final. A held payment cannot
  execute without approval, and nothing executes twice.
- **No autonomous retries.** After a denial, the agent revises only when the
  user replies. An agent that retries on its own learns the policy's edges, such
  as splitting a payment to stay under a limit.
- **Refusals are recorded.** Denied payments stay in the ledger for audit, with
  the policy's, the approver's and the gateway's reasons kept apart.

## A bug found in a live run

Running the demo while Gemini was returning `503` errors, the system **paid
Apple £400 when asked to "Pay Deloitte £300"**.

When a model call failed, the user's message stayed in the conversation with no
reply. After a few failures in a row, the model answered an older request
instead of the latest one. Every guardrail held; the payment was within policy.
The fault was in how the conversation handled a failure.

**Fix:** a request whose model call fails is removed from the conversation, and
the user is asked to send it again. Two regression tests reproduce the original
sequence. **Trade-off:** after an outage the user has to restate the request,
which is better than a payment they believe never happened.

## Running it

Requires Python 3.12 and a free Gemini API key.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your-key-here" > .env

python scripted_run.py             # end-to-end demo, calls Gemini
python -m pytest                   # 178 tests, no network or API key needed
```

If Gemini is busy (`503`/`504`), change `llm.model` in `src/config/config.yaml`.
The demo prints the expected outcome next to the actual one, because the agent
may ask a clarifying question instead of proposing.

## Limitations and next steps

The system is a single agent in one process, with an in-memory ledger and a
mocked payment gateway. To run it for real, I would add:

1. **An idempotency key on execution**, so a retry after a gateway timeout
   cannot pay twice.
2. **Budget reservations for held payments, and a policy re-check on approval**,
   so a payment approved hours later cannot push a beneficiary over the limit.
3. **Rejection of duplicate holds** for the same payee and amount.
4. **A database transaction around check-and-record**, for multiple workers.
5. **FX conversion**, so one limit applies across currencies. Today each
   currency has its own limits and totals.
6. **Payee ids instead of names**, so `Apple` and `apple` share one budget.
