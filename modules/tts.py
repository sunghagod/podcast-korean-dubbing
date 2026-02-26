"""
Korean TTS using edge-tts (ko-KR-SunHiNeural).
Generates MP3 audio from Korean text.
"""

import asyncio
import os
from typing import Callable, Optional

import edge_tts


VOICE = "ko-KR-SunHiNeural"

# edge-tts has a character limit per call; split long texts
TTS_CHUNK_SIZE = 3000


def _split_for_tts(text: str, chunk_size: int = TTS_CHUNK_SIZE) -> list[str]:
    """Split Korean text at sentence boundaries for TTS."""
    if len(text) <= chunk_size:
        return [text]

    import re
    # Korean sentence endings: . ? ! or Korean period/question mark
    sentences = re.split(r"(?<=[.?!。？！])\s*", text)

    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


async def _synthesize_chunk(text: str, output_path: str, rate: str = "+0%") -> None:
    """Synthesize a single chunk to an MP3 file."""
    communicate = edge_tts.Communicate(text, VOICE, rate=rate)
    await communicate.save(output_path)


async def _synthesize_all(chunks: list[str], output_dir: str, base_name: str, rate: str) -> list[str]:
    """Synthesize all chunks concurrently (up to 3 at a time)."""
    semaphore = asyncio.Semaphore(3)
    chunk_paths = []

    async def limited_synthesize(i: int, chunk: str) -> str:
        async with semaphore:
            path = os.path.join(output_dir, f"{base_name}_chunk_{i:04d}.mp3")
            await _synthesize_chunk(chunk, path, rate)
            return path

    tasks = [limited_synthesize(i, chunk) for i, chunk in enumerate(chunks)]
    chunk_paths = await asyncio.gather(*tasks)
    return list(chunk_paths)


def _merge_mp3_files(chunk_paths: list[str], output_path: str) -> None:
    """Concatenate MP3 chunk files into a single output MP3."""
    from pydub import AudioSegment

    combined = AudioSegment.empty()
    for path in chunk_paths:
        seg = AudioSegment.from_mp3(path)
        combined += seg

    combined.export(output_path, format="mp3", bitrate="128k")

    # Clean up chunk files
    for path in chunk_paths:
        try:
            os.remove(path)
        except OSError:
            pass


def generate_tts(
    korean_text: str,
    output_path: str,
    rate: str = "+0%",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Generate Korean TTS MP3 from text.

    Args:
        korean_text: Korean text to synthesize.
        output_path: Full path for the output MP3 file.
        rate: Speech rate adjustment (e.g. '+10%', '-5%').
        progress_callback: Optional status callback.

    Returns:
        Path to the generated MP3 file.
    """
    if not korean_text.strip():
        raise ValueError("TTS 입력 텍스트가 비어 있습니다.")

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(output_path))[0]

    chunks = _split_for_tts(korean_text)

    if progress_callback:
        progress_callback(f"TTS 생성 중... ({len(chunks)} 청크)")

    # Run async synthesis in a new event loop
    loop = asyncio.new_event_loop()
    try:
        chunk_paths = loop.run_until_complete(
            _synthesize_all(chunks, output_dir, base_name, rate)
        )
    finally:
        loop.close()

    if progress_callback:
        progress_callback("TTS 오디오 합치는 중...")

    if len(chunk_paths) == 1:
        # Single chunk: just rename
        os.replace(chunk_paths[0], output_path)
    else:
        _merge_mp3_files(chunk_paths, output_path)

    if progress_callback:
        progress_callback(f"TTS 완료: {output_path}")

    return output_path
