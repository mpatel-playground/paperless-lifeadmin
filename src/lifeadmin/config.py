"""Settings loaded from environment variables.

Everything the add-on needs to know about its surroundings comes from here, so the rest
of the code never reads ``os.environ`` directly. ``load_settings`` takes the environment
as an argument, which lets tests pass a plain dict instead of touching the real one.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
DEFAULT_BIND_HOST = "127.0.0.1"

REQUIRED = ("PAPERLESS_URL", "PAPERLESS_TOKEN")


class ConfigError(ValueError):
    """Raised when required settings are missing or invalid."""


@dataclass(frozen=True)
class Settings:
    paperless_url: str
    # repr=False keeps the token out of logs, tracebacks and debug output.
    paperless_token: str = field(repr=False)
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    bind_host: str = DEFAULT_BIND_HOST


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build ``Settings`` from ``env`` (defaults to ``os.environ``).

    Blank values count as missing. Every missing variable is reported at once, by name
    only, so the error never echoes a secret.
    """
    if env is None:
        env = os.environ

    def get(name: str) -> str:
        return env.get(name, "").strip()

    missing = [name for name in REQUIRED if not get(name)]
    if missing:
        raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

    return Settings(
        paperless_url=_normalise_url("PAPERLESS_URL", get("PAPERLESS_URL")),
        paperless_token=get("PAPERLESS_TOKEN"),
        ollama_url=_normalise_url("OLLAMA_URL", get("OLLAMA_URL") or DEFAULT_OLLAMA_URL),
        ollama_model=get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL,
        bind_host=get("BIND_HOST") or DEFAULT_BIND_HOST,
    )


def _normalise_url(name: str, value: str) -> str:
    """Require an http(s) scheme and drop trailing slashes, so callers can append paths."""
    if not value.startswith(("http://", "https://")):
        raise ConfigError(f"{name} must start with http:// or https://")
    return value.rstrip("/")
