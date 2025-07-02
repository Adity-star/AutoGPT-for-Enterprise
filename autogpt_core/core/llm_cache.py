# core/llm_cache.py

import os
import json
import hashlib
from pathlib import Path
from typing import Optional

CACHE_DIR = Path("llm_cache")
CACHE_DIR.mkdir(exist_ok=True)

def cache_key(prompt: str, model: str = "") -> str:
    """
    Hash the prompt and model together for uniqueness.
    """
    key_str = f"{model}::{prompt}"
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def get_cached_response(prompt: str, model: str = "") -> Optional[str]:
    key = cache_key(prompt, model)
    file_path = CACHE_DIR / f"{key}.json"

    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data.get("response")
            except json.JSONDecodeError:
                return None
    return None


def save_response(prompt: str, response: str, model: str = ""):
    key = cache_key(prompt, model)
    file_path = CACHE_DIR / f"{key}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({"response": response}, f, ensure_ascii=False)
