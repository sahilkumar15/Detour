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
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    # Hugging Face Inference Providers (a router). Listing it shows which
    # routed models your token can actually reach.
    "hf": ("https://router.huggingface.co/v1", "HF_TOKEN,HUGGINGFACE_API_KEY"),
    # Local keyless servers; empty key name means no key is required.
    "ollama": ("http://localhost:11434/v1", ""),
    "vllm": ("http://localhost:8000/v1", ""),
    "local": ("http://localhost:8000/v1", ""),
}

provider = (sys.argv[1] if len(sys.argv) > 1 else "openai").lower()
if provider not in PROVIDERS:
    raise SystemExit(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")

base_url, key_env = PROVIDERS[provider]
# Accept any of the accepted names for this provider's key.
candidates = [v.strip() for v in key_env.split(",") if v.strip()]
key = next((os.environ[n].strip() for n in candidates if os.environ.get(n, "").strip()), "")
if not key and candidates:
    raise SystemExit(f"{' or '.join(candidates)} is not set in .env")
key = key or "local"  # keyless local server

for model_id in sorted(m.id for m in OpenAI(api_key=key, base_url=base_url).models.list()):
    print(model_id)