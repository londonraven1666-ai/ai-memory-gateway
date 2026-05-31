"""
Token Cleaner - Phase 5 接口预留

规则：
  删除：无意义重复、已写入 memory 的旧事实、过期临时信息、低价值寒暄
  压缩保留：撒娇重复、情绪重复、亲密称呼重复、"想你/爱你/不开心/害怕"类
  完整保留：最新 6~12 轮原文、当前任务目标、技术决策、用户要求记住的、安全边界、健康相关
"""


async def clean_context(
    messages: list[dict],
    max_tokens: int = 8000,
) -> list[dict]:
    """Phase 5 实现：压缩旧上下文，控制 token budget。MVP 阶段直接截断。"""
    # MVP: 保留最近的消息，不做压缩
    return messages
