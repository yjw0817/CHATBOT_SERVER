"""Unified LLM client supporting multiple providers and local/remote mode."""
import os
import json
import re
from typing import Optional
from pathlib import Path

# Ensure .env is loaded
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_result = load_dotenv(env_path, override=True)
print(f"[LLM_CLIENT] Loaded .env from {env_path}: {load_result}")
print(f"[LLM_CLIENT] LLM_ENABLED={os.getenv('LLM_ENABLED')}, LLM_PROVIDER={os.getenv('LLM_PROVIDER')}, LLM_MODE={os.getenv('LLM_MODE')}")

# Runtime mode override (None = use env, "local"/"remote" = override)
_runtime_mode: Optional[str] = None
# Runtime model override (None = use env)
_runtime_model: Optional[str] = None


def set_llm_mode(mode: str):
    """Set LLM mode at runtime. 'local' or 'remote'."""
    global _runtime_mode
    if mode not in ("local", "remote"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'local' or 'remote'.")
    _runtime_mode = mode
    print(f"[LLM_CLIENT] Mode switched to: {mode}")


def get_llm_mode() -> str:
    """Get current LLM mode."""
    if _runtime_mode:
        return _runtime_mode
    return os.getenv("LLM_MODE", "remote").lower()


def set_llm_model(model: str):
    """Set LLM model at runtime."""
    global _runtime_model
    _runtime_model = model
    print(f"[LLM_CLIENT] Model switched to: {model}")


def get_llm_model() -> str:
    """Get current LLM model."""
    if _runtime_model:
        return _runtime_model
    return os.getenv("LLM_MODEL", "gpt-4o-mini")


def _get_config():
    """Get LLM config from environment (re-read each time)."""
    mode = get_llm_mode()
    if mode == "local":
        base_url = os.getenv("LLM_LOCAL_URL", "http://127.0.0.1:11434")
        api_key = ""  # local Ollama doesn't need API key
    else:
        base_url = os.getenv("LLM_REMOTE_URL", os.getenv("LLM_BASE_URL", ""))
        api_key = os.getenv("LLM_API_KEY", "")

    return {
        "api_key": api_key,
        "provider": os.getenv("LLM_PROVIDER", "openai").lower(),
        "model": get_llm_model(),
        "enabled": os.getenv("LLM_ENABLED", "false").lower() == "true",
        "base_url": base_url,
        "mode": mode,
    }


# Export for backward compatibility
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def call_llm(prompt: str, temperature: float = 0.3, model: Optional[str] = None) -> Optional[str]:
    """Call LLM with unified interface. Returns response text or None on failure.
    If model is provided, it overrides the global/runtime model for this call only."""
    config = _get_config()
    if model:
        config["model"] = model

    if not config["enabled"]:
        print(f"LLM disabled: enabled={config['enabled']}")
        return None

    # For non-ollama providers, require api_key
    if config["provider"] not in ("ollama", "local") and not config["api_key"]:
        print(f"LLM disabled: no api_key for provider={config['provider']}")
        return None

    try:
        if config["provider"] == "openai":
            return _call_openai(prompt, temperature, config)
        elif config["provider"] == "gemini":
            return _call_gemini(prompt, temperature, config)
        elif config["provider"] == "claude":
            return _call_claude(prompt, temperature, config)
        elif config["provider"] in ("ollama", "local"):
            return _call_ollama(prompt, temperature, config)
        else:
            print(f"Unknown LLM provider: {config['provider']}")
            return None
    except Exception as e:
        print(f"LLM call failed [{config['mode']}]: {e}")
        return None


def _call_openai(prompt: str, temperature: float, config: dict) -> Optional[str]:
    """OpenAI API call."""
    from openai import OpenAI
    client = OpenAI(api_key=config["api_key"])
    response = client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content


_GEMINI_MODEL = None

def _call_gemini(prompt: str, temperature: float, config: dict) -> Optional[str]:
    """Google Gemini API call."""
    import google.generativeai as genai
    global _GEMINI_MODEL

    if _GEMINI_MODEL is None:
        genai.configure(api_key=config["api_key"])
        _GEMINI_MODEL = genai.GenerativeModel(config["model"])

    response = _GEMINI_MODEL.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=temperature)
    )
    return response.text


def _call_claude(prompt: str, temperature: float, config: dict) -> Optional[str]:
    """Anthropic Claude API call."""
    import anthropic
    client = anthropic.Anthropic(api_key=config["api_key"])
    response = client.messages.create(
        model=config["model"],
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def _call_ollama(prompt: str, temperature: float, config: dict) -> Optional[str]:
    """Ollama via OpenAI-compatible API. Works for both local and remote."""
    import requests as _req
    base_url = (config["base_url"] or "http://127.0.0.1:11434").rstrip("/")
    # Support both /v1 suffix and bare URL
    if not base_url.endswith("/v1"):
        url = f"{base_url}/v1/chat/completions"
    else:
        url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config["api_key"] and config["api_key"] != "unused":
        headers["Authorization"] = f"Bearer {config['api_key']}"
    payload = {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 65536,
        "options": {"num_ctx": 131072},
    }
    mode_label = config.get("mode", "?")
    print(f"[LLM_OLLAMA] {mode_label} → {url} model={config['model']}")
    resp = _req.post(url, json=payload, headers=headers, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    # Fallback: if content is empty, check thinking/reasoning fields
    if not content.strip():
        for fallback_field in ("thinking", "reasoning", "reasoning_content"):
            fallback = msg.get(fallback_field)
            if fallback and fallback.strip():
                print(f"[LLM_OLLAMA] content empty, using '{fallback_field}' field")
                content = fallback
                break
    return content


def is_llm_available() -> bool:
    """Check if LLM is configured and available."""
    config = _get_config()
    if not config["enabled"]:
        return False
    # Ollama/local doesn't need api_key
    if config["provider"] in ("ollama", "local"):
        return True
    return bool(config["api_key"])


def get_llm_info() -> dict:
    """Get current LLM configuration info."""
    config = _get_config()
    return {
        "enabled": config["enabled"],
        "provider": config["provider"],
        "model": config["model"],
        "mode": config["mode"],
        "base_url": config["base_url"],
        "available": is_llm_available()
    }
