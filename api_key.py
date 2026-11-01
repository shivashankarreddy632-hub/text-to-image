"""
API Key Manager Module
=======================
Handles reading, validating, and saving API keys from:
1. `api_key.txt` (direct single-key file)
2. `.env` file
3. Environment variables (os.environ)

Supports Google Gemini / Imagen, OpenAI, Stability AI, and Hugging Face.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Load .env file
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# Provider key environment variable mappings
PROVIDER_ENV_VARS = {
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "stability": ["STABILITY_API_KEY", "STABILITY_KEY"],
    "huggingface": ["HF_TOKEN", "HUGGINGFACE_API_KEY", "HF_API_KEY"],
}

# Provider display names
PROVIDER_NAMES = {
    "openai": "OpenAI (GPT-Image / DALL-E)",
    "gemini": "Google Gemini / Imagen 3",
    "stability": "Stability AI (Stable Diffusion)",
    "huggingface": "Hugging Face (FLUX / SDXL)",
}


def read_raw_api_key_file() -> Optional[str]:
    """
    Reads the content of `api_key.txt` if present.
    Ignores comment lines starting with '#' and placeholder text.
    """
    key_file = BASE_DIR / "api_key.txt"
    if not key_file.exists():
        return None

    try:
        with open(key_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "YOUR_API_KEY_HERE" not in line and len(line) > 10:
                        return line
    except Exception as e:
        print(f"[Warning] Failed reading api_key.txt: {e}")
    return None


def detect_provider_from_key(key: str) -> str:
    """
    Guesses the provider based on the key prefix.
    """
    key = key.strip()
    if key.startswith("AIzaSy"):
        return "gemini"
    elif key.startswith("hf_"):
        return "huggingface"
    elif key.startswith("sk-"):
        # Could be OpenAI or Stability; default to OpenAI
        return "openai"
    return "gemini"


def get_api_key(provider: Optional[str] = None) -> Tuple[Optional[str], str]:
    """
    Retrieves the API key and active provider.
    Priority:
    1. Check specific provider env var in .env or os.environ
    2. Check api_key.txt (auto-detects provider if not specified)
    3. Check default provider from .env

    Returns:
        (api_key, detected_or_requested_provider)
    """
    # Reload .env in case it was updated during runtime
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

    if provider:
        provider = provider.lower()
    else:
        provider = os.getenv("DEFAULT_PROVIDER", "gemini").lower()

    # 1. Check provider-specific environment variables
    env_keys = PROVIDER_ENV_VARS.get(provider, [])
    for var in env_keys:
        val = os.getenv(var, "").strip()
        if val and "Your" not in val and len(val) > 8:
            return val, provider

    # 2. Check api_key.txt
    raw_key = read_raw_api_key_file()
    if raw_key:
        detected = detect_provider_from_key(raw_key)
        if provider:
            if provider == detected:
                return raw_key, provider
        else:
            return raw_key, detected

    # 3. Check any other provider env var (only if no specific provider requested)
    if not provider:
        for p, vars_list in PROVIDER_ENV_VARS.items():
            for var in vars_list:
                val = os.getenv(var, "").strip()
                if val and "Your" not in val and len(val) > 8:
                    return val, p

    return None, provider


def save_api_key(key: str, provider: str = "gemini", target: str = "both") -> bool:
    """
    Saves the API key to api_key.txt and/or .env file.
    
    Args:
        key: The API key string.
        provider: 'gemini', 'openai', 'stability', or 'huggingface'
        target: 'file' (api_key.txt), 'env' (.env), or 'both'
    """
    key = key.strip()
    provider = provider.lower()
    success = True

    if target in ("file", "both"):
        try:
            txt_path = BASE_DIR / "api_key.txt"
            content = (
                f"# ==============================================================================\n"
                f"# API KEY FILE ({PROVIDER_NAMES.get(provider, provider)})\n"
                f"# ==============================================================================\n"
                f"{key}\n"
            )
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"[Error] Failed saving to api_key.txt: {e}")
            success = False

    if target in ("env", "both"):
        try:
            env_path = BASE_DIR / ".env"
            var_name = PROVIDER_ENV_VARS.get(provider, [f"{provider.upper()}_API_KEY"])[0]

            lines = []
            updated = False
            if env_path.exists():
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

            new_lines = []
            for line in lines:
                if line.strip().startswith(f"{var_name}="):
                    new_lines.append(f"{var_name}={key}\n")
                    updated = True
                else:
                    new_lines.append(line)

            if not updated:
                new_lines.append(f"{var_name}={key}\n")

            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            # Update live process environment
            os.environ[var_name] = key
        except Exception as e:
            print(f"[Error] Failed saving to .env: {e}")
            success = False

    return success


def get_status_info() -> dict:
    """
    Returns a dictionary of API key statuses across all providers.
    """
    key, active_p = get_api_key()
    statuses = {}
    for p in PROVIDER_ENV_VARS:
        p_key, _ = get_api_key(p)
        statuses[p] = {
            "name": PROVIDER_NAMES.get(p, p),
            "has_key": p_key is not None,
            "preview": f"{p_key[:6]}...{p_key[-4:]}" if p_key else "Not provided yet",
        }
    return {
        "active_provider": active_p,
        "has_any_key": key is not None,
        "providers": statuses,
    }


if __name__ == "__main__":
    status = get_status_info()
    print("API Key Status:")
    print(f"Active Provider: {status['active_provider']}")
    print(f"Any Key Present: {status['has_any_key']}")
    for p, details in status["providers"].items():
        state = "READY" if details["has_key"] else "MISSING"
        print(f" - [{state}] {details['name']}: {details['preview']}")

