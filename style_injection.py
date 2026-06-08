import json
from typing import Any, Dict, Optional


def parse_style_list(raw_value: str) -> list[str]:
    try:
        value = json.loads(raw_value or "[]")
    except (TypeError, json.JSONDecodeError):
        return []

    if not isinstance(value, list):
        return []

    result = []
    for item in value:
        if isinstance(item, str) and item and item not in result:
            result.append(item)
    return result


def parse_style(raw_value: str) -> Optional[Dict[str, str]]:
    try:
        value = json.loads(raw_value or "")
    except (TypeError, json.JSONDecodeError):
        return None

    if not isinstance(value, dict):
        return None

    title = value.get("title")
    content = value.get("content")
    if not isinstance(title, str) or not isinstance(content, str):
        return None

    return {"title": title, "content": content}


def inject_style_into_current_user_message(
    messages: list[dict[str, Any]],
    style_content: str,
) -> list[dict[str, Any]]:
    if not style_content.strip():
        return messages

    suffix = (
        "<style_instructions>\n"
        f"{style_content.strip()}\n"
        "</style_instructions>"
    )

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "user":
            continue

        next_message = dict(message)
        content = message.get("content", "")

        if isinstance(content, str):
            next_message["content"] = f"{content}\n\n{suffix}" if content else suffix
        elif isinstance(content, list):
            next_content = [
                dict(item) if isinstance(item, dict) else item
                for item in content
            ]
            text_index = next(
                (
                    item_index
                    for item_index in range(len(next_content) - 1, -1, -1)
                    if isinstance(next_content[item_index], dict)
                    and next_content[item_index].get("type") == "text"
                ),
                None,
            )

            if text_index is None:
                next_content.append({"type": "text", "text": suffix})
            else:
                text_item = dict(next_content[text_index])
                text = text_item.get("text", "")
                text_item["text"] = f"{text}\n\n{suffix}" if text else suffix
                next_content[text_index] = text_item

            next_message["content"] = next_content
        else:
            next_message["content"] = suffix

        next_messages = list(messages)
        next_messages[index] = next_message
        return next_messages

    return messages
