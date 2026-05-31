import asyncpg
import uuid
from datetime import datetime, timezone
from gateway.config import DATABASE_URL

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    return _pool


async def init_schema():
    pool = await get_pool()
    with open("gateway/schema.sql") as f:
        sql = f.read()
    async with pool.acquire() as conn:
        await conn.execute(sql)


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# ── Sessions ──

async def create_session(user_id: str = "default", model: str = "") -> str:
    pool = await get_pool()
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO sessions (id, user_id, model, created_at, updated_at) VALUES ($1, $2, $3, $4, $4)",
            sid, user_id, model, now,
        )
    return sid


async def get_session(session_id: str) -> dict | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM sessions WHERE id = $1", session_id)
    return dict(row) if row else None


async def update_session_summary(session_id: str, summary: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET summary = $1, updated_at = $2 WHERE id = $3",
            summary, datetime.now(timezone.utc), session_id,
        )


# ── Messages ──

async def save_message(session_id: str, role: str, content: str, token_count: int = 0):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO messages (session_id, role, content, token_count, created_at) VALUES ($1, $2, $3, $4, $5)",
            session_id, role, content, token_count, datetime.now(timezone.utc),
        )


async def get_recent_messages(session_id: str, limit: int = 12) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content, created_at FROM messages WHERE session_id = $1 ORDER BY created_at DESC LIMIT $2",
            session_id, limit,
        )
    return [dict(r) for r in reversed(rows)]


# ── Memories (read) ──

async def vector_search(embedding: list[float], user_id: str = "default", top_k: int = 50) -> list[dict]:
    pool = await get_pool()
    vec_literal = "[" + ",".join(str(x) for x in embedding) + "]"
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, category AS type, topic, content, importance, heat, tags, created_at, updated_at AS last_used_at,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM core_memories
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            vec_literal, top_k,
        )
    return [dict(r) for r in rows]


async def keyword_search(query: str, user_id: str = "default", top_k: int = 50) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, category AS type, topic, content, importance, heat, tags, created_at, updated_at AS last_used_at,
                   ts_rank(to_tsvector('simple', content), plainto_tsquery('simple', $1)) AS rank
            FROM core_memories
            WHERE to_tsvector('simple', content) @@ plainto_tsquery('simple', $1)
            ORDER BY rank DESC
            LIMIT $2
            """,
            query, top_k,
        )
    return [dict(r) for r in rows]


async def touch_memory(memory_id: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE core_memories SET updated_at = $1, heat = LEAST(heat + 0.1, 3.0), activation_count = activation_count + 1 WHERE id = $2",
            datetime.now(timezone.utc), memory_id,
        )


# ── Memories (write, for extractor) ──

async def insert_memory(
    memory_id: str, user_id: str, mem_type: str, topic: str,
    content: str, importance: str, heat: float, tags: list[str],
    embedding: list[float] | None = None,
):
    pool = await get_pool()
    vec_literal = "[" + ",".join(str(x) for x in embedding) + "]" if embedding else None
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO core_memories (id, category, topic, content, importance, heat, tags, embedding, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector, $9, $9)
            ON CONFLICT (id) DO UPDATE SET content = $4, importance = $5, heat = $6, tags = $7, embedding = $8::vector, updated_at = $9
            """,
            memory_id, mem_type, topic, content, importance, heat, tags, vec_literal, now,
        )
