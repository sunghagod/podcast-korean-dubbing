"""
Korean TTS using edge-tts (ko-KR-SunHiNeural) or Fish Speech (voice cloning).
Generates MP3 audio from Korean text.
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from typing import Callable, Optional

import edge_tts

logger = logging.getLogger(__name__)

# ── Fish Speech server management ─────────────────────────────────────────────

_fish_server: Optional[subprocess.Popen] = None


def _is_fish_server_up() -> bool:
    """Return True if Fish Speech server is already responding on port 8080."""
    try:
        import httpx as _httpx
        resp = _httpx.get("http://localhost:8080/", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _ensure_fish_server(checkpoint_dir: str) -> None:
    """Start Fish Speech API server if not already running. Polls / for up to 120s."""
    global _fish_server

    # Already up (process we started, or external process)
    if _is_fish_server_up():
        return

    if _fish_server is not None and _fish_server.poll() is None:
        return  # Process running, give it more time below

    decoder_path = os.path.join(checkpoint_dir, "codec.pth")
    log_path = os.path.join(os.path.dirname(checkpoint_dir), "fish_speech_server.log")
    cmd = [
        sys.executable, "-m", "tools.api_server",
        "--listen", "0.0.0.0:8080",
        "--llama-checkpoint-path", checkpoint_dir,
        "--decoder-checkpoint-path", decoder_path,
        "--decoder-config-name", "modded_dac_vq",
    ]
    logger.info("Starting Fish Speech server: %s", " ".join(cmd))
    log_file = open(log_path, "w", encoding="utf-8")
    _fish_server = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
    )

    # Poll GET / for up to 120 seconds (model loading takes ~15s)
    for _ in range(120):
        if _is_fish_server_up():
            logger.info("Fish Speech server ready.")
            return
        time.sleep(1)

    raise RuntimeError("Fish Speech 서버가 120초 내에 시작되지 않았습니다.")


def generate_tts_fish_speech(
    korean_text: str,
    output_path: str,
    reference_audio_path: str,
    reference_text: str,
    checkpoint_dir: str,
    rate: str = "+0%",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Generate Korean TTS MP3 via Fish Speech (voice cloning).

    Args:
        korean_text: Korean text to synthesize.
        output_path: Full path for the output MP3 file.
        reference_audio_path: Path to the reference WAV (30s clip).
        reference_text: Transcription of the reference audio.
        checkpoint_dir: Path to openaudio-s1-mini checkpoint directory.
        rate: Ignored for Fish Speech (kept for API compatibility).
        progress_callback: Optional status callback.

    Returns:
        Path to the generated MP3 file.
    """
    try:
        import httpx
        import ormsgpack
    except ImportError as e:
        raise RuntimeError(f"Fish Speech 의존성 미설치: {e}. pip install ormsgpack httpx")

    if progress_callback:
        progress_callback("Fish Speech 서버 준비 중...")

    _ensure_fish_server(checkpoint_dir)

    if progress_callback:
        progress_callback("Fish Speech TTS 생성 중...")

    with open(reference_audio_path, "rb") as f:
        ref_audio_bytes = f.read()

    payload = {
        "text": korean_text,
        "references": [{"audio": ref_audio_bytes, "text": reference_text}],
        "format": "mp3",
        "normalize": True,
    }

    resp = httpx.post(
        "http://localhost:8080/v1/tts",
        content=ormsgpack.packb(payload, option=ormsgpack.OPT_SERIALIZE_NUMPY),
        headers={"Content-Type": "application/msgpack"},
        timeout=600,
    )
    resp.raise_for_status()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(resp.content)

    if progress_callback:
        progress_callback(f"Fish Speech TTS 완료: {output_path}")

    return output_path


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


async def _synthesize_chunk(text: str, output_path: str, rate: str = "+0%", voice: str = VOICE, pitch: str = "+0Hz") -> None:
    """Synthesize a single chunk to an MP3 file."""
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_path)


async def _synthesize_all(chunks: list[str], output_dir: str, base_name: str, rate: str, voice: str = VOICE, pitch: str = "+0Hz") -> list[str]:
    """Synthesize all chunks concurrently (up to 3 at a time)."""
    semaphore = asyncio.Semaphore(3)
    chunk_paths = []

    async def limited_synthesize(i: int, chunk: str) -> str:
        async with semaphore:
            path = os.path.join(output_dir, f"{base_name}_chunk_{i:04d}.mp3")
            await _synthesize_chunk(chunk, path, rate, voice, pitch)
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
    voice: str = VOICE,
    pitch: str = "+0Hz",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Generate Korean TTS MP3 from text.

    Args:
        korean_text: Korean text to synthesize.
        output_path: Full path for the output MP3 file.
        rate: Speech rate adjustment (e.g. '+10%', '-5%').
        voice: edge-tts voice name (e.g. 'ko-KR-SunHiNeural', 'ko-KR-InJoonNeural').
        pitch: Pitch adjustment (e.g. '+0Hz', '-15Hz').
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
            _synthesize_all(chunks, output_dir, base_name, rate, voice, pitch)
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
