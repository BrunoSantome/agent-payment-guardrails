"""Configuration loading and its consistency checks."""

from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from src.config import settings as settings_module
from src.config.settings import ConfigError, load_api_key, load_settings

REPO_CONFIG = Path(__file__).resolve().parents[1] / "src" / "config" / "config.yaml"

# A valid config, deep-copied and mutated per test.
VALID = {
    "llm": {"provider": "gemini", "model": "gemini-3.8-flash", "max_tokens": 4096},
    "guardrails": {
        "max_transaction_amount": 1000,
        "auto_approval_threshold": 500,
        "cumulative": {"window_hours": 24, "scope": "beneficiary", "limit": 5000},
        "allowed_categories": ["equipment", "salary", "travel"],
        "allowed_currencies": ["GBP", "EUR", "USD"],
    },
}


def write_config(tmp_path, **overrides) -> Path:
    """Write VALID to a temp file, with top-level sections merged from overrides."""
    config = {
        "llm": {**VALID["llm"], **overrides.get("llm", {})},
        "guardrails": {**VALID["guardrails"], **overrides.get("guardrails", {})},
    }
    for section in overrides:
        if section not in ("llm", "guardrails"):
            config[section] = overrides[section]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_repo_config_loads():
    """The config the system actually ships with must be valid."""
    settings = load_settings(REPO_CONFIG)
    assert settings.llm.provider == "gemini"
    assert settings.guardrails.max_transaction_amount == Decimal("1000")
    assert settings.guardrails.auto_approval_threshold == Decimal("500")
    assert settings.guardrails.cumulative.scope == "beneficiary"
    assert settings.guardrails.cumulative.window_hours == 24
    assert settings.guardrails.allowed_categories == ("equipment", "salary", "travel")
    assert settings.guardrails.allowed_currencies == ("GBP", "EUR", "USD")


def test_window_is_exposed_as_timedelta():
    settings = load_settings(REPO_CONFIG)
    assert settings.guardrails.cumulative.window.total_seconds() == 24 * 3600


def test_allow_lists_are_normalised(tmp_path):
    path = write_config(
        tmp_path,
        guardrails={
            "allowed_categories": [" Equipment ", "TRAVEL"],
            "allowed_currencies": ["gbp", " eur"],
        },
    )
    guardrails = load_settings(path).guardrails
    assert guardrails.allowed_categories == ("equipment", "travel")
    assert guardrails.allowed_currencies == ("GBP", "EUR")


def test_decimal_amount_keeps_exact_value(tmp_path):
    """Money must not pick up binary float rounding on the way in."""
    path = write_config(tmp_path, guardrails={"auto_approval_threshold": 500.50})
    threshold = load_settings(path).guardrails.auto_approval_threshold
    assert threshold == Decimal("500.50")


def test_threshold_above_max_is_rejected(tmp_path):
    """Otherwise no payment could ever be held for approval."""
    path = write_config(
        tmp_path,
        guardrails={"max_transaction_amount": 500, "auto_approval_threshold": 900},
    )
    with pytest.raises(ConfigError, match="auto_approval_threshold"):
        load_settings(path)


def test_cumulative_limit_below_max_is_rejected(tmp_path):
    """Otherwise a single payment at the maximum could never be made."""
    path = write_config(
        tmp_path,
        guardrails={
            "max_transaction_amount": 1000,
            "cumulative": {"window_hours": 24, "scope": "beneficiary", "limit": 500},
        },
    )
    with pytest.raises(ConfigError, match="cumulative.limit"):
        load_settings(path)


def test_unknown_cumulative_scope_is_rejected(tmp_path):
    path = write_config(
        tmp_path,
        guardrails={
            "cumulative": {"window_hours": 24, "scope": "department", "limit": 5000}
        },
    )
    with pytest.raises(ConfigError, match="scope"):
        load_settings(path)


@pytest.mark.parametrize("scope", ["beneficiary", "category", "global"])
def test_supported_scopes_load(tmp_path, scope):
    path = write_config(
        tmp_path,
        guardrails={"cumulative": {"window_hours": 24, "scope": scope, "limit": 5000}},
    )
    assert load_settings(path).guardrails.cumulative.scope == scope


def test_unknown_provider_is_rejected(tmp_path):
    path = write_config(tmp_path, llm={"provider": "skynet"})
    with pytest.raises(ConfigError, match="provider"):
        load_settings(path)


@pytest.mark.parametrize("amount", [0, -5, "many"])
def test_invalid_amounts_are_rejected(tmp_path, amount):
    path = write_config(tmp_path, guardrails={"auto_approval_threshold": amount})
    with pytest.raises(ConfigError, match="auto_approval_threshold"):
        load_settings(path)


@pytest.mark.parametrize("categories", [[], "equipment", None])
def test_allow_list_must_be_a_non_empty_list(tmp_path, categories):
    path = write_config(tmp_path, guardrails={"allowed_categories": categories})
    with pytest.raises(ConfigError, match="allowed_categories"):
        load_settings(path)


def test_missing_section_is_reported(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"llm": VALID["llm"]}))
    with pytest.raises(ConfigError, match="guardrails"):
        load_settings(path)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="No config file"):
        load_settings(tmp_path / "nope.yaml")


def test_malformed_yaml_is_reported(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("llm: [unclosed")
    with pytest.raises(ConfigError, match="Could not parse"):
        load_settings(path)


def test_api_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setattr(settings_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GEMINI_API_KEY", "  test-key  ")
    assert load_api_key("gemini") == "test-key"


def test_missing_api_key_names_the_variable(monkeypatch):
    monkeypatch.setattr(settings_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        load_api_key("gemini")


def test_api_key_for_unknown_provider(monkeypatch):
    monkeypatch.setattr(settings_module, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(ConfigError, match="skynet"):
        load_api_key("skynet")
