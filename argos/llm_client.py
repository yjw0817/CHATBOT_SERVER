"""Unified LLM client supporting multiple providers."""
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
print(f"[LLM_CLIENT] LLM_ENABLED={os.getenv('LLM_ENABLED')}, LLM_PROVIDER={os.getenv('LLM_PROVIDER')}")


def _get_config():
    """Get LLM config from environment (re-read each time)."""
    return {
        "api_key": os.getenv("LLM_API_KEY", ""),
        "provider": os.getenv("LLM_PROVIDER", "openai").lower(),
        "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "enabled": os.getenv("LLM_ENABLED", "false").lower() == "true",
        "base_url": os.getenv("LLM_BASE_URL", "")
    }


# Export for backward compatibility
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def call_llm(prompt: str, temperature: float = 0.3) -> Optional[str]:
    """Call LLM with unified interface. Returns response text or None on failure."""
    config = _get_config()
    
    if not config["enabled"] or not config["api_key"]:
        print(f"LLM disabled: enabled={config['enabled']}, has_key={bool(config['api_key'])}")
        return None
    
    try:
        if config["provider"] == "openai":
            return _call_openai(prompt, temperature, config)
        elif config["provider"] == "gemini":
            return _call_gemini(prompt, temperature, config)
        elif config["provider"] == "claude":
            return _call_claude(prompt, temperature, config)
        elif config["provider"] in ("ollama", "local"):
            return _call_local(prompt, temperature, config)
        else:
            print(f"Unknown LLM provider: {config['provider']}")
            return None
    except Exception as e:
        print(f"LLM call failed: {e}")
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


def _call_local(prompt: str, temperature: float, config: dict) -> Optional[str]:
    """Local LLM (Ollama) via OpenAI-compatible API."""
    from openai import OpenAI
    base_url = config["base_url"] or "http://localhost:11434/v1"
    client = OpenAI(base_url=base_url, api_key="ollama")
    response = client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content


def is_llm_available() -> bool:
    """Check if LLM is configured and available."""
    config = _get_config()
    return config["enabled"] and bool(config["api_key"])


def get_llm_info() -> dict:
    """Get current LLM configuration info."""
    config = _get_config()
    return {
        "enabled": config["enabled"],
        "provider": config["provider"],
        "model": config["model"],
        "available": is_llm_available()
    }
