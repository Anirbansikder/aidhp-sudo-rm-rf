# src/utils/openai_util.py

import os
import openai

def get_openai_client(api_key: str = None, base_url: str = None):
    """
    Initialize the openai module with an API key and a base URL.
    If not provided, it falls back to environment variables:
      - OPENAI_API_KEY
      - OPENAI_BASE_URL (defaults to "https://api.openai.com/v1" if not set)
    Returns the openai module with the configuration applied.
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    openai.api_key = api_key
    openai.api_base = base_url
    
    return openai
