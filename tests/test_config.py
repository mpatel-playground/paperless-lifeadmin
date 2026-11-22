import dataclasses

import pytest

from lifeadmin.config import (
    DEFAULT_BIND_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    ConfigError,
    load_settings,
)

# Synthetic values only; never a real token.
FAKE_TOKEN = "fake-token-0123456789"
MINIMAL_ENV = {"PAPERLESS_URL": "http://paperless.test:8000", "PAPERLESS_TOKEN": FAKE_TOKEN}


def test_minimal_env_uses_defaults():
    settings = load_settings(MINIMAL_ENV)

    assert settings.paperless_url == "http://paperless.test:8000"
    assert settings.paperless_token == FAKE_TOKEN
    assert settings.ollama_url == DEFAULT_OLLAMA_URL
    assert settings.ollama_model == DEFAULT_OLLAMA_MODEL
    assert settings.bind_host == DEFAULT_BIND_HOST


def test_binds_to_localhost_by_default():
    assert load_settings(MINIMAL_ENV).bind_host == "127.0.0.1"


def test_overrides_are_read():
    env = {
        **MINIMAL_ENV,
        "OLLAMA_URL": "http://host.docker.internal:11434",
        "OLLAMA_MODEL": "some-model:7b",
        "BIND_HOST": "0.0.0.0",
    }
    settings = load_settings(env)

    assert settings.ollama_url == "http://host.docker.internal:11434"
    assert settings.ollama_model == "some-model:7b"
    assert settings.bind_host == "0.0.0.0"


@pytest.mark.parametrize("missing", ["PAPERLESS_URL", "PAPERLESS_TOKEN"])
def test_missing_required_var_is_named(missing):
    env = {k: v for k, v in MINIMAL_ENV.items() if k != missing}

    with pytest.raises(ConfigError, match=missing):
        load_settings(env)


def test_all_missing_vars_reported_together():
    with pytest.raises(ConfigError, match="PAPERLESS_URL, PAPERLESS_TOKEN"):
        load_settings({})


def test_blank_value_counts_as_missing():
    with pytest.raises(ConfigError, match="PAPERLESS_TOKEN"):
        load_settings({**MINIMAL_ENV, "PAPERLESS_TOKEN": "   "})


def test_blank_optional_falls_back_to_default():
    settings = load_settings({**MINIMAL_ENV, "OLLAMA_URL": "", "BIND_HOST": " "})

    assert settings.ollama_url == DEFAULT_OLLAMA_URL
    assert settings.bind_host == DEFAULT_BIND_HOST


def test_values_are_stripped():
    settings = load_settings({**MINIMAL_ENV, "PAPERLESS_TOKEN": f"  {FAKE_TOKEN}\n"})

    assert settings.paperless_token == FAKE_TOKEN


@pytest.mark.parametrize("name", ["PAPERLESS_URL", "OLLAMA_URL"])
def test_trailing_slashes_are_removed(name):
    settings = load_settings({**MINIMAL_ENV, name: "http://example.test:1234//"})

    assert getattr(settings, name.lower()) == "http://example.test:1234"


@pytest.mark.parametrize("name", ["PAPERLESS_URL", "OLLAMA_URL"])
def test_url_without_scheme_is_rejected(name):
    with pytest.raises(ConfigError, match=name):
        load_settings({**MINIMAL_ENV, name: "localhost:8000"})


def test_token_not_in_repr():
    settings = load_settings(MINIMAL_ENV)

    assert FAKE_TOKEN not in repr(settings)
    assert FAKE_TOKEN not in str(settings)


def test_token_not_in_error_message():
    with pytest.raises(ConfigError) as excinfo:
        load_settings({"PAPERLESS_TOKEN": FAKE_TOKEN})

    assert FAKE_TOKEN not in str(excinfo.value)


def test_settings_are_immutable():
    settings = load_settings(MINIMAL_ENV)

    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.paperless_url = "http://other.test"  # type: ignore[misc]


def test_reads_os_environ_by_default(monkeypatch):
    for name in ("OLLAMA_URL", "OLLAMA_MODEL", "BIND_HOST"):
        monkeypatch.delenv(name, raising=False)
    for name, value in MINIMAL_ENV.items():
        monkeypatch.setenv(name, value)

    assert load_settings().paperless_url == "http://paperless.test:8000"
