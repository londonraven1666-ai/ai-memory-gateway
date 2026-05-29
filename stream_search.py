"""流式 gateway XML 标签检测"""
import json
import re

STATE_NORMAL = 0
STATE_MAYBE_TAG = 1
STATE_INSIDE_TAG = 2

TAG_OPEN = "<SEARCH_MEMORY>"
TAG_CLOSE = "</SEARCH_MEMORY>"
SAVE_OPEN_RE = re.compile(r'<SAVE_MEMORY\s+title="([^"]*)">')
SAVE_CLOSE = "</SAVE_MEMORY>"
EXEC_OPEN = "<EXEC_VPS>"
EXEC_CLOSE = "</EXEC_VPS>"


def make_sse_content_chunk(text):
    """构造 SSE content delta"""
    data = {"choices": [{"delta": {"content": text}}]}
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def make_sse_done():
    """构造 SSE 结束标记"""
    return b"data: [DONE]\n\n"


class SearchTagDetector:
    def __init__(self, max_triggers=3):
        self.state = STATE_NORMAL
        self.buffer = ""
        self.tag_content = ""
        self.search_query = None
        self.trigger_count = 0
        self.max_triggers = max_triggers

    def feed(self, content_text):
        """
        返回 (safe_text, search_triggered)
        safe_text: 可以安全发给用户的文本
        search_triggered: 是否检测到完整标签
        
        注意：triggered 后同一次 feed 中剩余的文本会被丢弃
        （因为调用方会立即中断第一段流，剩余是模型无记忆的猜测）
        """
        if self.trigger_count >= self.max_triggers:
            # 达到触发上限，只做标签清理
            cleaned = re.sub(r'</?SEARCH_MEMORY[^>]*>', '', content_text)
            return cleaned, False

        safe = []
        for char in content_text:
            if self.state == STATE_NORMAL:
                if char == '<':
                    self.state = STATE_MAYBE_TAG
                    self.buffer = '<'
                else:
                    safe.append(char)

            elif self.state == STATE_MAYBE_TAG:
                self.buffer += char
                if TAG_OPEN.startswith(self.buffer):
                    if self.buffer == TAG_OPEN:
                        self.state = STATE_INSIDE_TAG
                        self.tag_content = ""
                        self.buffer = ""
                else:
                    # 不是标签，释放 buffer
                    safe.append(self.buffer)
                    self.buffer = ""
                    self.state = STATE_NORMAL

            elif self.state == STATE_INSIDE_TAG:
                self.tag_content += char
                if self.tag_content.endswith(TAG_CLOSE):
                    query = self.tag_content[:-len(TAG_CLOSE)].strip()
                    self.search_query = query
                    self.trigger_count += 1
                    self.search_query = query
                    self.state = STATE_NORMAL
                    self.tag_content = ""
                    # 标签后的文本丢弃（调用方会中断流）
                    return "".join(safe), True

        return "".join(safe), False

    @property
    def triggered(self):
        return self.trigger_count > 0

    def reset_for_next_stream(self):
        """Reset state for next stream segment, keep trigger count"""
        self.state = STATE_NORMAL
        self.buffer = ""
        self.tag_content = ""
        self.search_query = None

    def flush(self):
        """流结束时释放缓冲"""
        result = ""
        if self.buffer:
            result += self.buffer
            self.buffer = ""
        if self.state == STATE_INSIDE_TAG and self.tag_content:
            result += TAG_OPEN + self.tag_content
            self.tag_content = ""
        self.state = STATE_NORMAL
        return result


class SaveMemoryDetector:
    """检测并拦截 <SAVE_MEMORY title="...">content</SAVE_MEMORY>，不中断流"""

    def __init__(self):
        self.state = STATE_NORMAL
        self.buffer = ""
        self.tag_content = ""
        self.current_title = ""
        self.saved = []

    def feed(self, content_text):
        safe = []
        for char in content_text:
            if self.state == STATE_NORMAL:
                if char == '<':
                    self.state = STATE_MAYBE_TAG
                    self.buffer = '<'
                else:
                    safe.append(char)

            elif self.state == STATE_MAYBE_TAG:
                self.buffer += char
                if '<SAVE_MEMORY'.startswith(self.buffer):
                    continue
                if self.buffer.startswith('<SAVE_MEMORY'):
                    if '>' not in self.buffer:
                        continue
                    match = SAVE_OPEN_RE.fullmatch(self.buffer)
                    if match:
                        self.current_title = match.group(1)
                        self.tag_content = ""
                        self.buffer = ""
                        self.state = STATE_INSIDE_TAG
                    else:
                        safe.append(self.buffer)
                        self.buffer = ""
                        self.state = STATE_NORMAL
                else:
                    safe.append(self.buffer)
                    self.buffer = ""
                    self.state = STATE_NORMAL

            elif self.state == STATE_INSIDE_TAG:
                self.tag_content += char
                if self.tag_content.endswith(SAVE_CLOSE):
                    content = self.tag_content[:-len(SAVE_CLOSE)].strip()
                    title = self.current_title.strip()
                    if content:
                        self.saved.append({"title": title, "content": content})
                    self.current_title = ""
                    self.tag_content = ""
                    self.state = STATE_NORMAL

        return "".join(safe), list(self.saved)

    @property
    def triggered(self):
        return self.trigger_count > 0

    def reset_for_next_stream(self):
        """Reset state for next stream segment, keep trigger count"""
        self.state = STATE_NORMAL
        self.buffer = ""
        self.tag_content = ""
        self.search_query = None

    def flush(self):
        result = ""
        if self.buffer:
            result += self.buffer
            self.buffer = ""
        if self.state == STATE_INSIDE_TAG and self.tag_content:
            result += f'<SAVE_MEMORY title="{self.current_title}">' + self.tag_content
            self.tag_content = ""
            self.current_title = ""
        self.state = STATE_NORMAL
        return result


class ExecVpsDetector:
    """检测 <EXEC_VPS>cmd</EXEC_VPS>，触发后中断当前流"""

    def __init__(self):
        self.state = STATE_NORMAL
        self.buffer = ""
        self.tag_content = ""
        self.exec_command = None
        self.triggered = False

    def feed(self, content_text):
        if self.triggered:
            cleaned = re.sub(r'<EXEC_VPS>.*?</EXEC_VPS>', '', content_text, flags=re.DOTALL)
            cleaned = cleaned.replace(EXEC_OPEN, "").replace(EXEC_CLOSE, "")
            return cleaned, False

        safe = []
        for char in content_text:
            if self.state == STATE_NORMAL:
                if char == '<':
                    self.state = STATE_MAYBE_TAG
                    self.buffer = '<'
                else:
                    safe.append(char)

            elif self.state == STATE_MAYBE_TAG:
                self.buffer += char
                if EXEC_OPEN.startswith(self.buffer):
                    if self.buffer == EXEC_OPEN:
                        self.state = STATE_INSIDE_TAG
                        self.tag_content = ""
                        self.buffer = ""
                else:
                    safe.append(self.buffer)
                    self.buffer = ""
                    self.state = STATE_NORMAL

            elif self.state == STATE_INSIDE_TAG:
                self.tag_content += char
                if self.tag_content.endswith(EXEC_CLOSE):
                    self.exec_command = self.tag_content[:-len(EXEC_CLOSE)].strip()
                    self.triggered = True
                    self.state = STATE_NORMAL
                    self.tag_content = ""
                    return "".join(safe), True

        return "".join(safe), False

    @property
    def triggered(self):
        return self.trigger_count > 0

    def reset_for_next_stream(self):
        """Reset state for next stream segment, keep trigger count"""
        self.state = STATE_NORMAL
        self.buffer = ""
        self.tag_content = ""
        self.search_query = None

    def flush(self):
        result = ""
        if self.buffer:
            result += self.buffer
            self.buffer = ""
        if self.state == STATE_INSIDE_TAG and self.tag_content:
            result += EXEC_OPEN + self.tag_content
            self.tag_content = ""
        self.state = STATE_NORMAL
        return result
