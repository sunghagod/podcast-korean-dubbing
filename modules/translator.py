"""
English → Korean translation using deep-translator (Google backend).
Splits long text into chunks to stay within Google Translate's character limit.
"""

import time
from typing import Callable, Optional

from deep_translator import GoogleTranslator


# Google Translate has a ~5000 char limit per request
CHUNK_SIZE = 4500


def _split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Split text into chunks at sentence boundaries (. ! ?) to preserve context.
    Falls back to hard split if no sentence boundary found.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    current = ""

    # Split by sentences first
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # If single sentence is too long, hard split it
            if len(sentence) > chunk_size:
                for i in range(0, len(sentence), chunk_size):
                    chunks.append(sentence[i : i + chunk_size])
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


def translate(
    text: str,
    source_lang: str = "en",
    target_lang: str = "ko",
    progress_callback: Optional[Callable[[str], None]] = None,
    retry_delay: float = 1.0,
) -> str:
    """
    Translate English text to Korean in chunks.

    Args:
        text: Source English text.
        source_lang: Source language code.
        target_lang: Target language code.
        progress_callback: Optional status callback.
        retry_delay: Seconds to wait between chunk requests (rate limit avoidance).

    Returns:
        Korean translated text.
    """
    if not text.strip():
        return ""

    chunks = _split_into_chunks(text)
    translator = GoogleTranslator(source=source_lang, target=target_lang)

    translated_chunks = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(f"번역 중... ({i + 1}/{total} 청크)")

        # Retry up to 3 times on failure
        for attempt in range(3):
            try:
                result = translator.translate(chunk)
                translated_chunks.append(result)
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"번역 실패 (청크 {i + 1}/{total}): {e}") from e
                time.sleep(retry_delay * (attempt + 1))

        # Small delay to avoid rate limiting
        if i < total - 1:
            time.sleep(retry_delay)

    return " ".join(translated_chunks)
