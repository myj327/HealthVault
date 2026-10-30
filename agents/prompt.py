"""Minimal fallback; HealthVault supplies its own domain prompt."""


def build_system_prompt() -> str:
    return ("You are a helpful assistant. Use only the tools supplied by this application. "
            "Do not invent tool results or sources.")
