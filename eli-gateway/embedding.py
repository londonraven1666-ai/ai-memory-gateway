import httpx
from gateway.config import EMBEDDING_API_URL, EMBEDDING_MODEL

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=30.0)
    return _client


async def get_embedding(text: str) -> list[float]:
    """生成文本的 embedding 向量 (1024维 bge-m3)

    支持两种 API 格式：
    - Ollama: POST /api/embed  {"model": "bge-m3", "input": "text"}
    - OpenAI 兼容: POST /v1/embeddings  {"model": "bge-m3", "input": "text"}
    """
    client = _get_client()

    if "/v1/" in EMBEDDING_API_URL or "embeddings" in EMBEDDING_API_URL:
        resp = await client.post(
            EMBEDDING_API_URL,
            json={"model": EMBEDDING_MODEL, "input": text},
            headers={"Content-Type": "application/json"},
        )
        data = resp.json()
        if "data" in data and data["data"]:
            return data["data"][0].get("embedding", [])
        return data.get("embedding", [])
    else:
        resp = await client.post(
            EMBEDDING_API_URL,
            json={"model": EMBEDDING_MODEL, "input": text},
        )
        data = resp.json()
        if "embeddings" in data and data["embeddings"]:
            return data["embeddings"][0]
        return data.get("embedding", [])


async def close_client():
    global _client
    if _client:
        await _client.aclose()
        _client = None
