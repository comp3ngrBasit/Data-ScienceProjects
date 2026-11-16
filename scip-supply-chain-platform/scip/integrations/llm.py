"""LLM provider abstraction for the Copilot and report narratives.

Providers: Anthropic Claude (default) or OpenAI, selected by LLM_PROVIDER. Keys are read only from
environment variables (ANTHROPIC_API_KEY / OPENAI_API_KEY). If no key is configured, or the call fails,
`generate` returns None and callers fall back to retrieval-only answers.
"""
from __future__ import annotations

from scip.core.config import get_settings
from scip.core.logging import get_logger

log = get_logger("llm")


def provider_status() -> dict:
    s = get_settings()
    configured = {"anthropic": bool(s.anthropic_api_key), "openai": bool(s.openai_api_key)}
    active = s.llm_provider if s.llm_provider in configured and configured[s.llm_provider] else None
    if active is None:
        active = next((p for p, ok in configured.items() if ok), None) if s.llm_provider != "none" else None
    return {"selected": s.llm_provider, "active": active, "configured": configured,
            "model": {"anthropic": s.anthropic_model, "openai": s.openai_model}.get(active or "", None)}


def generate(system: str, prompt: str, max_tokens: int = 700) -> tuple[str | None, str]:
    """Returns (text or None, provider label)."""
    s = get_settings()
    active = provider_status()["active"]
    try:
        if active == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=s.anthropic_api_key, timeout=s.http_timeout_seconds * 3)
            msg = client.messages.create(model=s.anthropic_model, max_tokens=max_tokens, system=system,
                                         messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text"), f"anthropic:{s.anthropic_model}"
        if active == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=s.openai_api_key, timeout=s.http_timeout_seconds * 3)
            r = client.chat.completions.create(model=s.openai_model, max_tokens=max_tokens,
                                               messages=[{"role": "system", "content": system},
                                                         {"role": "user", "content": prompt}])
            return r.choices[0].message.content, f"openai:{s.openai_model}"
    except Exception as exc:   # network, auth, quota — degrade gracefully
        log.warning("LLM call failed (%s): %s", active, exc)
        return None, f"{active}:error"
    return None, "none"
