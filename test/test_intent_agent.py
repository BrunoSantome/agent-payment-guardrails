"""The intent agent, against a fake LLM: no network, no API key."""

from src.agent.intent_agent import AgentReply, IntentAgent, IntentDraft
from src.llm.client import Message


class FakeLLM:
    """Returns scripted replies and records how it was called."""

    def __init__(self, *replies: AgentReply) -> None:
        self._replies = list(replies)
        self.calls: list[dict] = []

    def structured(self, *, instruction, messages, schema):
        self.calls.append(
            {"instruction": instruction, "messages": list(messages), "schema": schema}
        )
        return self._replies.pop(0)


def agent(*replies) -> tuple[IntentAgent, FakeLLM]:
    client = FakeLLM(*replies)
    return (
        IntentAgent(client, categories=("equipment", "travel"), currencies=("GBP",)),
        client,
    )


def proposal(**overrides) -> AgentReply:
    draft = {
        "amount": "2499",
        "currency": "GBP",
        "beneficiary": "Apple",
        "category": "equipment",
        "justification": "MacBook for the new starter",
    }
    return AgentReply(
        action="propose",
        intent=IntentDraft(**{**draft, **overrides}),
        message="Paying Apple 2499 GBP for a MacBook.",
    )


def test_a_proposal_comes_back_as_a_draft():
    intent_agent, _ = agent(proposal())
    reply = intent_agent.respond([Message("user", "Buy a MacBook from Apple, £2499")])
    assert reply.action == "propose"
    assert reply.intent.amount == "2499"
    assert reply.intent.beneficiary == "Apple"


def test_an_ambiguous_request_comes_back_as_a_question():
    intent_agent, _ = agent(
        AgentReply(action="clarify", message="Which beneficiary did you mean?")
    )
    reply = intent_agent.respond([Message("user", "pay them the usual")])
    assert reply.action == "clarify"
    assert reply.intent is None
    assert "beneficiary" in reply.message


def test_the_whole_conversation_is_sent_as_context():
    """A revised intent depends on the agent seeing why the last one failed."""
    intent_agent, client = agent(proposal(amount="400"))
    history = [
        Message("user", "Pay Apple £5000 for MacBooks"),
        Message("model", "Denied: exceeds the 1,000.00 GBP maximum"),
        Message("user", "just one then"),
    ]
    intent_agent.respond(history)
    assert client.calls[0]["messages"] == history


def test_the_instruction_carries_the_configured_categories():
    """The prompt hints at the allow-list; the policy engine still enforces it."""
    _, client = agent(proposal())
    intent_agent, client = agent(proposal())
    intent_agent.respond([Message("user", "anything")])
    instruction = client.calls[0]["instruction"]
    assert "equipment, travel" in instruction
    assert "GBP" in instruction


def test_the_agent_is_told_it_does_not_decide():
    intent_agent, client = agent(proposal())
    intent_agent.respond([Message("user", "anything")])
    assert "do not approve" in client.calls[0]["instruction"]
