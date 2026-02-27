"""
faster-whisper GPU Speech-to-Text module.
Transcribes audio files to English text using the large-v3 model.
"""

import os
from typing import Callable, Optional

from faster_whisper import WhisperModel


# Singleton model instance (loaded once, reused across calls)
_model: Optional[WhisperModel] = None


def _get_model(model_size: str = "large-v3", device: str = "cuda") -> WhisperModel:
    global _model
    if _model is None:
        print(f"[Transcriber] Loading Whisper model '{model_size}' on {device}...")
        _model = WhisperModel(
            model_size,
            device=device,
            compute_type="float16",  # FP16 for GPU speed
        )
        print("[Transcriber] Model loaded.")
    return _model


def transcribe(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "cuda",
    language: str = "en",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Transcribe an audio file to text.

    Args:
        audio_path: Path to WAV/MP3 audio file.
        model_size: Whisper model size (default: large-v3).
        device: 'cuda' or 'cpu'.
        language: Language code (e.g. 'en', 'ko').
        progress_callback: Optional function called with status strings.

    Returns:
        Transcribed text as a single string.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _get_model(model_size, device)

    if progress_callback:
        progress_callback("Whisper STT 처리 중...")

    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,  # Skip silent parts
        vad_parameters={"min_silence_duration_ms": 500},
    )

    texts = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            texts.append(text)

    full_text = " ".join(texts)
    return full_text
