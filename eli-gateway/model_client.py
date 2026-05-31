import httpx
from gateway import config as _config

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=120.0)
    return _client


async def chat_completion(
    messages: list[dict],
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = False,
) -> str:
    """调用 LLM API，返回回复文本。支持 OpenAI 兼容格式。"""
    provider = provider or _config.DEFAULT_PROVIDER
    conf = _config.LLM_PROVIDERS.get(provider, _config.LLM_PROVIDERS.get("openai", {}))

    base_url = conf.get("base_url", "").rstrip("/")
    api_key = conf.get("api_key", "")
    model = model or conf.get("default_model", "gpt-4o")

    client = _get_client()

    if provider == "claude":
        return await _claude_chat(client, base_url, api_key, model, messages, temperature, max_tokens)

    # OpenAI 兼容格式（也适用于各种转发服务）
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    resp = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    return data["choices"][0]["message"]["content"]


async def _claude_chat(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    """Anthropic Messages API with prompt caching sandwich.
    
    Static layer (persona + rules) gets cache_control for prompt caching.
    Dynamic layer (memories + context) is injected after the cached prefix.
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    system_blocks = []
    chat_messages = []
    for i, msg in enumerate(messages):
        if msg["role"] == "system":
            block = {"type": "text", "text": msg["content"]}
            # First system message = static layer → cache it
            if len(system_blocks) == 0:
                block["cache_control"] = {"type": "ephemeral"}
            system_blocks.append(block)
        else:
            chat_messages.append(msg)

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": chat_messages,
    }
    if system_blocks:
        payload["system"] = system_blocks

    resp = await client.post(f"{base_url}/v1/messages", json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    # Log cache performance
    usage = data.get("usage", {})
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_create = usage.get("cache_creation_input_tokens", 0)
    if cache_read or cache_create:
        import logging
        logging.getLogger("gateway").info(f"Cache: read={cache_read} create={cache_create}")

    return "".join(block["text"] for block in data["content"] if block["type"] == "text")


async def close_client():
    global _client
    if _client:
        await _client.aclose()
        _client = None
