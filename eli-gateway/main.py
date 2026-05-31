"""
Eli API Gateway - MVP (Phase 1-3)

Phase 1: POST /v1/chat → LLM API → 返回回复
Phase 2: sessions + messages 落库 PostgreSQL
Phase 3: Memory Retriever hybrid search，记忆注入 prompt
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gateway import database as db
from gateway import model_client
from gateway import embedding as emb
from gateway.memory import retrieve_memories
from gateway.prompt import build_prompt
from gateway.token_cleaner import clean_context
from gateway.extractor import extract_memories
from gateway.config import RECENT_MESSAGES_LIMIT


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_schema()
    yield
    await db.close_pool()
    await model_client.close_client()
    await emb.close_client()


app = FastAPI(title="Eli Gateway", version="0.1.0", lifespan=lifespan)

from fastapi.responses import FileResponse as _FR
from fastapi.staticfiles import StaticFiles as _SF
import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "static")
if _os.path.isdir(_static_dir):
    app.mount("/static", _SF(directory=_static_dir), name="static")

@app.get("/config")
async def config_page():
    return _FR(_os.path.join(_static_dir, "config.html"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ──

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    user_id: str = "default"
    system_prompt: str = ""
    provider: str | None = None
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    enable_memory: bool = True

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    memories_used: int = 0


class MemorySearchRequest(BaseModel):
    query: str
    user_id: str = "default"
    top_k: int = 15

class MemorySearchResponse(BaseModel):
    results: list[dict]


class SessionResponse(BaseModel):
    session_id: str
    messages: list[dict]
    summary: str


# ── Endpoints ──

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Phase 2: 获取或创建 session
    if req.session_id:
        session = await db.get_session(req.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        session_id = req.session_id
    else:
        session_id = await db.create_session(req.user_id, req.model or "")
        session = await db.get_session(session_id)

    # Phase 2: 保存用户消息
    await db.save_message(session_id, "user", req.message)

    # Phase 3: 检索相关记忆
    memories = []
    if req.enable_memory:
        try:
            memories = await retrieve_memories(req.message, req.user_id)
        except Exception:
            pass

    # Phase 2: 获取最近对话
    recent_messages = await db.get_recent_messages(session_id, RECENT_MESSAGES_LIMIT)

    # 组装 prompt
    session_summary = session.get("summary", "") if session else ""
    messages = build_prompt(
        user_message=req.message,
        system_prompt=req.system_prompt,
        session_summary=session_summary,
        memories=memories,
        recent_messages=recent_messages[:-1] if len(recent_messages) > 1 else [],
    )

    # Phase 1: 调用 LLM
    try:
        reply = await model_client.chat_completion(
            messages=messages,
            provider=req.provider,
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {str(e)}")

    # Phase 2: 保存助手回复
    await db.save_message(session_id, "assistant", reply)

    # Phase 6 预留: 异步抽取记忆
    asyncio.create_task(_bg_extract(recent_messages + [{"role": "user", "content": req.message}, {"role": "assistant", "content": reply}], session_id, req.user_id))

    return ChatResponse(reply=reply, session_id=session_id, memories_used=len(memories))


@app.post("/v1/memories/search", response_model=MemorySearchResponse)
async def search_memories(req: MemorySearchRequest):
    results = await retrieve_memories(req.query, req.user_id, req.top_k)
    safe_results = []
    for r in results:
        safe_results.append({
            "id": r.get("id"),
            "type": r.get("type"),
            "topic": r.get("topic"),
            "content": r.get("content"),
            "importance": r.get("importance"),
            "heat": r.get("heat"),
            "tags": r.get("tags"),
            "score": round(r.get("final_score", 0), 4),
        })
    return MemorySearchResponse(results=safe_results)


@app.get("/v1/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = await db.get_recent_messages(session_id, limit=100)
    return SessionResponse(
        session_id=session_id,
        messages=messages,
        summary=session.get("summary", ""),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0", "phases": ["chat", "sessions", "memory_retriever"]}




# ── Runtime Config ──

class ConfigUpdate(BaseModel):
    provider: str | None = None
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None

@app.get("/v1/config")
async def get_config():
    """Get current LLM provider config (keys masked)."""
    from gateway.config import LLM_PROVIDERS, DEFAULT_PROVIDER
    result = {"default_provider": DEFAULT_PROVIDER, "providers": {}}
    for name, conf in LLM_PROVIDERS.items():
        key = conf.get("api_key", "")
        result["providers"][name] = {
            "base_url": conf.get("base_url", ""),
            "model": conf.get("default_model", ""),
            "has_key": bool(key),
            "key_preview": f"{key[:8]}...{key[-4:]}" if len(key) > 12 else ("set" if key else "empty"),
        }
    return result

@app.post("/v1/config")
async def update_config(req: ConfigUpdate):
    """Update LLM provider config at runtime. No restart needed."""
    from gateway import config
    provider = req.provider or config.DEFAULT_PROVIDER
    if provider not in config.LLM_PROVIDERS:
        config.LLM_PROVIDERS[provider] = {"base_url": "", "api_key": "", "default_model": ""}
    if req.api_key is not None:
        config.LLM_PROVIDERS[provider]["api_key"] = req.api_key
    if req.model is not None:
        config.LLM_PROVIDERS[provider]["default_model"] = req.model
    if req.base_url is not None:
        config.LLM_PROVIDERS[provider]["base_url"] = req.base_url
    if req.provider:
        config.DEFAULT_PROVIDER = req.provider
    return {"success": True, "provider": provider}



# ── OpenAI-compatible endpoint (for Hafen frontend) ──

class OpenAIChatRequest(BaseModel):
    model: str = "gateway"
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False

@app.post("/v1/chat/completions")
async def openai_compat_chat(req: OpenAIChatRequest, request: Request):
    """OpenAI-compatible chat endpoint. Drop-in replacement for Hafen."""
    # Read API key from Authorization header
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        incoming_key = auth_header[7:]
        if incoming_key:
            from gateway import config
            if incoming_key.startswith("sk-ant-"):
                config.LLM_PROVIDERS.setdefault("claude", {})["api_key"] = incoming_key
            else:
                config.LLM_PROVIDERS.setdefault("openai", {})["api_key"] = incoming_key

    # Read custom base_url from X-Base-URL header (for relay services)
    custom_base = request.headers.get("x-base-url", "")
    if custom_base:
        from gateway import config
        config.LLM_PROVIDERS.setdefault("openai", {})["base_url"] = custom_base.rstrip("/")
    # Extract user message and system prompt from OpenAI format
    user_message = ""
    system_prompt = ""
    for msg in req.messages:
        if msg["role"] == "user":
            user_message = msg["content"]
        elif msg["role"] == "system":
            system_prompt = msg["content"] if not system_prompt else system_prompt + "\n\n" + msg["content"]

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message")

    # Route through gateway logic
    # Use default provider. Model name is passed through as-is.
    provider = None
    model = req.model if req.model not in ("gateway", "gateway-no-memory") else None

    chat_req = ChatRequest(
        message=user_message,
        system_prompt=system_prompt,
        provider=provider,
        model=model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        enable_memory=req.model != "gateway-no-memory",
    )
    result = await chat(chat_req)

    # Return in OpenAI format
    return {
        "id": f"chatcmpl-{result.session_id[:8]}",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.reply},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

@app.get("/v1/models")
async def list_models():
    """Return currently configured model + common models."""
    from gateway import config as _cfg
    models = [{"id": "gateway", "object": "model"}]
    # Add the actual configured model from each provider
    for name, conf in _cfg.LLM_PROVIDERS.items():
        m = conf.get("default_model", "")
        if m and not any(x["id"] == m for x in models):
            models.append({"id": m, "object": "model"})
    return {"data": models}

async def _bg_extract(messages: list[dict], session_id: str, user_id: str):
    """后台异步抽取记忆（Phase 6 接入点）"""
    try:
        await extract_memories(messages, session_id, user_id)
    except Exception:
        pass
