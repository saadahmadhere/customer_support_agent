"""Single place that builds the chat model used across the app."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from customer_support_agent.core.settings import Settings


def build_chat_model(settings: Settings, temperature: float | None = None) -> BaseChatModel:
    """Return the configured chat LLM (currently OpenAI)."""
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it in .env before generating drafts.")

    return ChatOpenAI(
        model=settings.openai_model,
        api_key=SecretStr(settings.openai_api_key),
        temperature=settings.llm_temperature if temperature is None else temperature,
    )
