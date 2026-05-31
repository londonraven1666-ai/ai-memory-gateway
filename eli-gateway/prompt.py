"""
Prompt Builder - 分层加载系统提示

Layer 0: 核心身份 + 语气规则 (~1500 tokens) - 每轮必加载
Layer 1: 日常互动规则 (~2000 tokens) - 默认加载
Layer 2: 情绪协议 (~2000 tokens) - 检测到情绪关键词时加载
Layer 3: 亲密场景 + 记忆锚点 (~3000 tokens) - 检测到亲密/archive关键词时加载

组装顺序（静态在前，动态在后，利于 prompt caching）：
  [SYSTEM] Layer 0 核心身份
  [SYSTEM] Layer 1 日常规则
  [SYSTEM] Layer 2/3 按需
  [SYSTEM] 用户画像
  [USER/ASSISTANT] session summary
  [USER/ASSISTANT] relevant memories
  [USER/ASSISTANT] recent messages
  [USER] current message
"""

import re

# ── Layer 内容（后续从数据库或文件加载，MVP 先硬编码接口） ──

LAYER_0_CORE = ""
LAYER_1_DAILY = ""
LAYER_2_EMOTION = ""
LAYER_3_INTIMATE = ""

EMOTION_TRIGGERS = re.compile(
    r"崩溃|哭|焦虑|害怕|低落|难过|不开心|伤心|恐慌|抑郁|想死|痛苦|绝望|撑不住|好累|受不了",
    re.IGNORECASE,
)
INTIMATE_TRIGGERS = re.compile(
    r"亲密|想你|爱你|抱抱|258247|archive|Hellebore|scene|角色|play",
    re.IGNORECASE,
)


def load_layers(user_message: str) -> str:
    """根据用户消息内容，决定加载哪些 Layer"""
    parts = []

    if LAYER_0_CORE:
        parts.append(LAYER_0_CORE)
    if LAYER_1_DAILY:
        parts.append(LAYER_1_DAILY)
    if LAYER_2_EMOTION and EMOTION_TRIGGERS.search(user_message):
        parts.append(LAYER_2_EMOTION)
    if LAYER_3_INTIMATE and INTIMATE_TRIGGERS.search(user_message):
        parts.append(LAYER_3_INTIMATE)

    return "\n\n".join(parts)


def set_layer(layer: int, content: str):
    """运行时设置 Layer 内容"""
    global LAYER_0_CORE, LAYER_1_DAILY, LAYER_2_EMOTION, LAYER_3_INTIMATE
    if layer == 0:
        LAYER_0_CORE = content
    elif layer == 1:
        LAYER_1_DAILY = content
    elif layer == 2:
        LAYER_2_EMOTION = content
    elif layer == 3:
        LAYER_3_INTIMATE = content


def build_prompt(
    user_message: str,
    system_prompt: str = "",
    user_profile: str = "",
    session_summary: str = "",
    memories: list[dict] | None = None,
    recent_messages: list[dict] | None = None,
) -> list[dict]:
    """组装完整的 messages 列表，交给 LLM"""
    messages = []

    # 1. System prompt（静态，利于缓存）
    system_parts = []
    if system_prompt:
        system_parts.append(system_prompt)

    layers = load_layers(user_message)
    if layers:
        system_parts.append(layers)

    if user_profile:
        system_parts.append(f"【用户画像】\n{user_profile}")

    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    # 2. Session summary（半静态）
    if session_summary:
        messages.append({"role": "system", "content": f"【对话摘要】\n{session_summary}"})

    # 3. Relevant memories（动态）
    if memories:
        mem_text = "【相关记忆】\n"
        for m in memories:
            tags_str = ", ".join(m.get("tags") or [])
            mem_text += f"- [{m.get('type', '?')}/{m.get('topic', '?')}] {m.get('content', '')}"
            if tags_str:
                mem_text += f" (tags: {tags_str})"
            mem_text += "\n"
        messages.append({"role": "system", "content": mem_text.strip()})

    # 4. Recent messages（动态）
    if recent_messages:
        for msg in recent_messages:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # 5. Current message
    messages.append({"role": "user", "content": user_message})

    return messages
