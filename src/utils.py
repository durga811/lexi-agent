"""Small shared helpers."""
from __future__ import annotations


def message_text(content) -> str:
    """Flatten a LangChain message `.content` to plain text.

    Gemini 3.x returns content as a list of blocks (e.g.
    [{"type": "text", "text": "...", "extras": {...}}]) rather than a string.
    Both the UI and the eval need plain text, so normalise here in one place.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p for p in parts if p)
    return str(content)
