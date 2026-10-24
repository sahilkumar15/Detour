#!/usr/bin/env python3
"""List the model ids an API key can reach.

    python list_models.py                 # OpenAI
    python list_models.py groq            # Groq (free tier, open-weight models)
    python list_models.py together        # Together

Standalone because `python -c` does not run run_audit.py and so never
loads .env.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
    ),
}

provider = (sys.argv[1] if len(sys.argv) > 1 else "openai").lower()
if provider not in PROVIDERS:
    raise SystemExit(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")

base_url, key_env = PROVIDERS[provider]
key = os.environ.get(key_env, "").strip()
if not key:
    raise SystemExit(f"{key_env} is not set in .env")

for model_id in sorted(m.id for m in OpenAI(api_key=key, base_url=base_url).models.list()):
    print(model_id)