"""
Multi-provider LLM abstraction layer.
Selects provider based on AI_PROVIDER env var,
falls back gracefully if a provider's API key is missing.
"""
import os


def get_llm(provider: str = None, temperature: float = 0.3):
    """
    Returns a LangChain chat model for the requested provider.
    Falls back to the next available provider if the requested
    one has no API key configured.
    """
    provider = provider or os.environ.get("AI_PROVIDER", "anthropic")

    providers_available = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers_available.append("anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        providers_available.append("openai")
    if os.environ.get("GOOGLE_API_KEY"):
        providers_available.append("google")

    if not providers_available:
        raise RuntimeError(
            "No AI provider API keys configured. Set one of: "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY"
        )

    if provider not in providers_available:
        provider = providers_available[0]

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            temperature=temperature,
            max_tokens=1024,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            temperature=temperature,
            max_tokens=1024,
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.environ.get("GOOGLE_MODEL", "gemini-2.5-pro"),
            temperature=temperature,
        )

    raise RuntimeError(f"Unknown provider: {provider}")


def get_current_provider() -> str:
    """Returns which provider is actually active right now."""
    provider = os.environ.get("AI_PROVIDER", "anthropic")
    if provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GOOGLE_API_KEY"):
        return "google"
    return "none"
