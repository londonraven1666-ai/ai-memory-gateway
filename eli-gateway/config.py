import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/eli")

EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:11434/api/embed")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_DIM = 1024

LLM_PROVIDERS = {
    "openai": {
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "default_model": os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o"),
    },
    "claude": {
        "base_url": os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com"),
        "api_key": os.getenv("CLAUDE_API_KEY", ""),
        "default_model": os.getenv("CLAUDE_DEFAULT_MODEL", "claude-sonnet-4-20250514"),
    },
}

DEFAULT_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")

MEMORY_SEARCH_TOP_K = int(os.getenv("MEMORY_SEARCH_TOP_K", "15"))
RECENT_MESSAGES_LIMIT = int(os.getenv("RECENT_MESSAGES_LIMIT", "12"))
MAX_TOKEN_BUDGET = int(os.getenv("MAX_TOKEN_BUDGET", "8000"))

QWEN_EXTRACTOR_URL = os.getenv("QWEN_EXTRACTOR_URL", "http://localhost:11434/api/chat")
QWEN_EXTRACTOR_MODEL = os.getenv("QWEN_EXTRACTOR_MODEL", "qwen3-vl:8b")
