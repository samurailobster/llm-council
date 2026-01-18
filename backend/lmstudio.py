"""LM Studio local LLM client for making requests to locally hosted models."""

import httpx
from typing import List, Dict, Any, Optional
from .config import LM_STUDIO_BASE_URL


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a locally hosted model via LM Studio's OpenAI-compatible API.

    Args:
        model: Model identifier with 'lmstudio/' prefix (e.g., "lmstudio/mistral-7b")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    # Strip the 'lmstudio/' prefix to get the local model name
    if not model.startswith("lmstudio/"):
        print(f"Warning: Model '{model}' does not use 'lmstudio/' prefix. Skipping LM Studio client.")
        return None

    local_model_name = model[len("lmstudio/"):]  # Remove prefix

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "model": local_model_name,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": -1,  # Unlimited
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{LM_STUDIO_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            message = data['choices'][0]['message']

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details')
            }

    except Exception as e:
        print(f"Error querying LM Studio model {model}: {e}")
        return None
