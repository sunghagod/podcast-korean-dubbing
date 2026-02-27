"""
English → Korean translation.
- Claude API (기본): 구어체 더빙 품질
- Google Translate (fallback): Claude API 키 없을 때
"""

import json
import re
import time
from typing import Callable, Dict, List, Optional


# ── Google Translate (fallback) ────────────────────────────────────────────────

CHUNK_SIZE = 4500


def _split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    current = ""
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            if len(sentence) > chunk_size:
                for i in range(0, len(sentence), chunk_size):
                    chunks.append(sentence[i : i + chunk_size])
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


def _google_translate(
    text: str,
    source_lang: str = "en",
    target_lang: str = "ko",
    progress_callback: Optional[Callable[[str], None]] = None,
    retry_delay: float = 1.0,
) -> str:
    from deep_translator import GoogleTranslator

    chunks = _split_into_chunks(text)
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    translated_chunks = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(f"번역 중... ({i + 1}/{total} 청크)")

        for attempt in range(3):
            try:
                result = translator.translate(chunk)
                translated_chunks.append(result)
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"번역 실패 (청크 {i + 1}/{total}): {e}") from e
                time.sleep(retry_delay * (attempt + 1))

        if i < total - 1:
            time.sleep(retry_delay)

    return " ".join(translated_chunks)


# ── Claude 번역 ────────────────────────────────────────────────────────────────

_DUBBING_SYSTEM_PROMPT = """당신은 영어 팟캐스트를 한국어로 더빙하는 전문 번역가입니다.

번역 원칙:
- 구어체 자연스러운 한국어로 번역 (문어체 금지)
- 원문의 리듬, 어조, 감정을 살려서 번역
- "Yeah", "Right", "Exactly", "I mean" 같은 구어 표현은 "응", "맞아", "그렇죠", "그러니까" 등 자연스러운 한국어 구어로
- 관용구와 비유는 직역 말고 의역
- 문장을 끊지 말고 자연스럽게 이어지게
- 존댓말/반말은 원문 화자의 관계와 톤에 맞게 일관되게 유지
- 번역 결과에 영어 문장이나 영어 구절을 절대 남기지 마세요. 반드시 전부 한국어로 번역해야 합니다.
- 인명·브랜드명 등 고유명사는 영문 그대로 두어도 됩니다. 단, 일반 단어·문장은 모두 한국어로 번역하세요."""

_DIARIZED_SYSTEM_PROMPT = """당신은 영어 팟캐스트 대화를 한국어로 더빙하는 전문 번역가입니다.

번역 원칙:
- 구어체 자연스러운 한국어 (문어체 금지)
- 대화 흐름과 앞뒤 맥락을 반영한 번역
- 각 화자의 개성과 어조 유지 (인터뷰어 vs 게스트 말투 차별화)
- "Yeah", "Right", "Uh-huh" 같은 짧은 반응은 "응", "맞아", "그렇죠" 등으로
- 관용구·비유는 의역, 직역 금지
- 번역 결과에 영어 문장이나 영어 구절을 절대 남기지 마세요. 반드시 전부 한국어로 번역해야 합니다.
- 인명·브랜드명 등 고유명사는 영문 그대로 두어도 됩니다. 단, 일반 단어·문장은 모두 한국어로 번역하세요.
- 반드시 아래 JSON 형식만 반환 (설명 텍스트 없이):
[{"speaker": "A", "text": "번역문"}, ...]"""


def _claude_translate_text(
    text: str,
    api_key: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """일반 텍스트를 Claude로 번역 (청크 분할)."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    # Claude는 200k 토큰이므로 청크를 크게 잡음
    chunks = _split_into_chunks(text, chunk_size=12000)
    total = len(chunks)
    results = []

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(f"Claude 번역 중... ({i + 1}/{total})")

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=8192,
            system=_DUBBING_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"다음 영어 팟캐스트 텍스트를 한국어로 번역해주세요. 번역문만 반환하세요.\n\n{chunk}",
                }
            ],
        )
        results.append(message.content[0].text.strip())

    return " ".join(results)


def translate_diarized_with_claude(
    utterances: List[Dict],
    api_key: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[Dict]:
    """
    화자 구분 발화 목록을 Claude로 번역.
    전체 대화 맥락을 한 번에 넘겨 자연스러운 구어체 번역.

    Args:
        utterances: [{"speaker": "A", "text": "...", "start_ms": ..., "end_ms": ...}, ...]
        api_key: Anthropic API 키.
        progress_callback: 진행 상황 콜백.

    Returns:
        번역된 utterances 리스트 (speaker/start_ms/end_ms 유지, text만 교체).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    # 한 번에 처리할 최대 발화 수 (너무 길면 분할)
    BATCH_SIZE = 40

    all_translated: List[Dict] = []
    total_batches = (len(utterances) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        batch = utterances[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        if progress_callback:
            start = batch_idx * BATCH_SIZE + 1
            end = min((batch_idx + 1) * BATCH_SIZE, len(utterances))
            progress_callback(f"Claude 번역 중... ({start}~{end}/{len(utterances)})")

        # 전체 대화를 JSON으로 구성
        conversation_json = json.dumps(
            [{"speaker": u["speaker"], "text": u["text"]} for u in batch],
            ensure_ascii=False,
            indent=2,
        )

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=8192,
            system=_DIARIZED_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"다음 팟캐스트 대화를 한국어로 번역해주세요. JSON 배열만 반환하세요.\n\n{conversation_json}",
                }
            ],
        )

        raw = message.content[0].text.strip()

        # JSON 파싱 (마크다운 코드블록 제거)
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            translated_batch = json.loads(raw)
        except json.JSONDecodeError:
            # 파싱 실패 시 Google Translate로 개별 재번역 (영어 원문 그대로 두지 않음)
            from deep_translator import GoogleTranslator
            gt = GoogleTranslator(source="en", target="ko")
            translated_batch = []
            for u in batch:
                try:
                    ko = gt.translate(u["text"]) or u["text"]
                except Exception:
                    ko = u["text"]
                translated_batch.append({"speaker": u["speaker"], "text": ko})

        # start_ms / end_ms 복원
        for orig, trans in zip(batch, translated_batch):
            all_translated.append({
                "speaker": orig["speaker"],
                "text": trans.get("text", orig["text"]),
                "start_ms": orig["start_ms"],
                "end_ms": orig["end_ms"],
            })

    return all_translated


# ── 공개 인터페이스 ────────────────────────────────────────────────────────────

def translate(
    text: str,
    source_lang: str = "en",
    target_lang: str = "ko",
    progress_callback: Optional[Callable[[str], None]] = None,
    retry_delay: float = 1.0,
    claude_api_key: str = "",
) -> str:
    """
    영어 → 한국어 번역.
    claude_api_key가 제공되면 Claude(구어체 더빙 품질),
    없으면 Google Translate(fallback) 사용.
    """
    if not text.strip():
        return ""

    if claude_api_key:
        try:
            return _claude_translate_text(text, claude_api_key, progress_callback)
        except Exception as e:
            if progress_callback:
                progress_callback(f"Claude 번역 실패, Google로 fallback: {e}")

    return _google_translate(text, source_lang, target_lang, progress_callback, retry_delay)
