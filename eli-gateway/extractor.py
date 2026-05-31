"""
Memory Extractor - Phase 6 接口预留

本地 Qwen3-VL 8B 负责：
  1. 每轮对话结束后异步抽取记忆
  2. 区分：撒娇玩闹 vs 真实崩溃、一次性偏好 vs 长期偏好、普通聊天 vs 技术决策
  3. 15 类 topic 打标
  4. importance 判断
  5. heat 初始赋值
  6. 写入 PostgreSQL

输出 JSON:
{
  "should_save": true,
  "memories": [
    {
      "type": "technical",
      "topic": "gateway",
      "content": "...",
      "importance": "high",
      "heat": 1.2,
      "tags": ["api-gateway", "postgresql"]
    }
  ],
  "session_summary_update": "..."
}
"""


async def extract_memories(
    messages: list[dict],
    session_id: str,
    user_id: str = "default",
) -> dict | None:
    """Phase 6 实现：调用本地 Qwen 抽取记忆并写入 DB。MVP 阶段返回 None。"""
    return None
