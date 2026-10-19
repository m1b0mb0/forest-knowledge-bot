"""Environment validation without real credentials."""

import pytest

from bot.config import load_llm_settings


def test_groq_model_defaults_and_secret_is_not_in_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "  example-secret  ")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    settings = load_llm_settings()

    assert settings.api_key == "example-secret"
    assert settings.model == "openai/gpt-oss-20b"
    assert "example-secret" not in repr(settings)


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_groq_key_is_required(
    monkeypatch: pytest.MonkeyPatch, api_key: str | None
) -> None:
    if api_key is None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GROQ_API_KEY", api_key)

    with pytest.raises(ValueError, match="GROQ_API_KEY is missing"):
        load_llm_settings()


@pytest.mark.parametrize(
    "model", ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
)
def test_both_supported_groq_models_can_be_configured(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "example-secret")
    monkeypatch.setenv("GROQ_MODEL", f" {model} ")

    assert load_llm_settings().model == model


@pytest.mark.parametrize("model", ["", "unknown-model", "secret-in-model"])
def test_unsupported_model_is_rejected_without_echoing_its_value(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "example-secret")
    monkeypatch.setenv("GROQ_MODEL", model)

    with pytest.raises(ValueError, match="GROQ_MODEL must be") as error:
        load_llm_settings()

    assert "example-secret" not in str(error.value)
    if model:
        assert model not in str(error.value)
