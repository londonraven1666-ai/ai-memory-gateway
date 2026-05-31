from datetime import datetime, timezone
from gateway.database import vector_search, keyword_search, touch_memory
from gateway.embedding import get_embedding
from gateway.config import MEMORY_SEARCH_TOP_K

IMPORTANCE_SCORE = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}


def _recency_score(created_at) -> float:
    if not created_at:
        return 0.0
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at)
        except ValueError:
            return 0.0
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    days = (now - created_at).total_seconds() / 86400
    return 1.0 / (1.0 + days / 30.0)


def _compute_final_score(
    semantic_sim: float,
    kw_rank: float,
    importance: str,
    heat: float,
    created_at,
) -> float:
    """
    final_score =
      semantic_similarity * 0.45
    + keyword_match       * 0.20
    + importance_score    * 0.15
    + heat_score          * 0.10
    + recency_score       * 0.10
    """
    imp = IMPORTANCE_SCORE.get(importance, 0.5)
    heat_norm = min(heat / 3.0, 1.0)
    recency = _recency_score(created_at)
    return (
        semantic_sim * 0.45
        + kw_rank * 0.20
        + imp * 0.15
        + heat_norm * 0.10
        + recency * 0.10
    )


async def retrieve_memories(query: str, user_id: str = "default", top_k: int | None = None) -> list[dict]:
    """混合检索：向量 + 关键词，重排序后返回 top_k 条"""
    top_k = top_k or MEMORY_SEARCH_TOP_K

    embedding = await get_embedding(query)

    if embedding:
        vec_results = await vector_search(embedding, user_id, top_k=50)
    else:
        vec_results = []

    kw_results = await keyword_search(query, user_id, top_k=50)

    # 合并去重
    merged: dict[str, dict] = {}

    if vec_results:
        max_sim = max(r.get("similarity", 0) for r in vec_results) or 1.0
    else:
        max_sim = 1.0

    for r in vec_results:
        rid = r["id"]
        merged[rid] = {
            **r,
            "semantic_sim": r.get("similarity", 0) / max_sim,
            "kw_rank": 0.0,
        }

    if kw_results:
        max_rank = max(r.get("rank", 0) for r in kw_results) or 1.0
    else:
        max_rank = 1.0

    for r in kw_results:
        rid = r["id"]
        norm_rank = r.get("rank", 0) / max_rank
        if rid in merged:
            merged[rid]["kw_rank"] = norm_rank
        else:
            merged[rid] = {**r, "semantic_sim": 0.0, "kw_rank": norm_rank}

    # 计算 final_score 并排序
    for m in merged.values():
        m["final_score"] = _compute_final_score(
            m["semantic_sim"],
            m["kw_rank"],
            m.get("importance", "medium"),
            m.get("heat", 1.0),
            m.get("created_at"),
        )

    ranked = sorted(merged.values(), key=lambda x: x["final_score"], reverse=True)[:top_k]

    # 更新 last_used_at 和 heat
    for m in ranked:
        await touch_memory(m["id"])

    return ranked
