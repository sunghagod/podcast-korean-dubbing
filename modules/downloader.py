"""
YouTube subtitle/audio downloader using yt-dlp.
Tries to extract existing subtitles first; falls back to downloading audio for Whisper STT.
"""

import os
import re
import json
import tempfile
from pathlib import Path
from typing import Optional

import yt_dlp


def _sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def get_video_info(url: str) -> dict:
    """Return basic video metadata (title, duration, id)."""
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "id": info.get("id", ""),
        "title": info.get("title", "Unknown"),
        "duration": info.get("duration", 0),
        "url": url,
    }


def extract_subtitles(url: str, output_dir: str) -> Optional[str]:
    """
    Try to download English subtitles (auto-generated or manual).
    Returns path to the subtitle text file, or None if unavailable.
    """
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "json3",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id", "")

    # Look for downloaded subtitle files
    for lang in ["en", "en-US", "en-GB"]:
        for suffix in [f".{lang}.json3", f".{lang}-orig.json3"]:
            sub_path = os.path.join(output_dir, f"{video_id}{suffix}")
            if os.path.exists(sub_path):
                text = _parse_json3_subtitle(sub_path)
                if text.strip():
                    txt_path = os.path.join(output_dir, f"{video_id}_en.txt")
                    Path(txt_path).write_text(text, encoding="utf-8")
                    return txt_path

    return None


def _parse_json3_subtitle(json3_path: str) -> str:
    """Convert yt-dlp json3 subtitle format to plain text."""
    with open(json3_path, encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    for event in data.get("events", []):
        segs = event.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if text and text != "\n":
            lines.append(text)

    # Join and clean up
    full_text = " ".join(lines)
    # Collapse multiple spaces/newlines
    full_text = re.sub(r"\s+", " ", full_text).strip()
    return full_text


def download_reference_clip(url: str, output_dir: str, duration: int = 30) -> str:
    """
    Download the first `duration` seconds of a YouTube video as WAV for voice cloning.
    Returns path to the WAV file named {video_id}_ref.wav.
    """
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(id)s_ref.%(ext)s"),
        "download_ranges": yt_dlp.utils.download_range_func(None, [(0, duration)]),
        "force_keyframes_at_cuts": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id", "")

    wav_path = os.path.join(output_dir, f"{video_id}_ref.wav")
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Reference audio not found after download: {wav_path}")

    return wav_path


def download_voice_clip(url: str, start_sec: float, end_sec: float, output_path: str) -> str:
    """
    Download a specific time range from a YouTube video as WAV for voice cloning.

    Args:
        url: YouTube URL.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        output_path: Full path for the output WAV file.

    Returns:
        Path to the saved WAV file.
    """
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    tmp_outtmpl = os.path.join(output_dir, "_voice_clip_tmp.%(ext)s")
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": tmp_outtmpl,
        "download_ranges": yt_dlp.utils.download_range_func(None, [(start_sec, end_sec)]),
        "force_keyframes_at_cuts": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    tmp_wav = os.path.join(output_dir, "_voice_clip_tmp.wav")
    if not os.path.exists(tmp_wav):
        raise FileNotFoundError(f"클립 다운로드 실패: {tmp_wav}")

    os.replace(tmp_wav, output_path)
    return output_path


def download_audio(url: str, output_dir: str) -> str:
    """
    Download best audio from YouTube and convert to WAV for Whisper.
    Returns path to the WAV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id", "")

    wav_path = os.path.join(output_dir, f"{video_id}.wav")
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found after download: {wav_path}")

    return wav_path
