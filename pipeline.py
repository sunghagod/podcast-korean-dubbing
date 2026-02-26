"""
Full processing pipeline: YouTube URL → Korean dubbed MP3.
Supports parallel processing of multiple URLs (max 3 concurrent).
"""

import os
import re
import time
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

from modules.downloader import get_video_info, extract_subtitles, download_audio
from modules.transcriber import transcribe
from modules.translator import translate
from modules.tts import generate_tts


class Stage(str, Enum):
    PENDING = "대기 중"
    FETCHING_INFO = "영상 정보 가져오는 중"
    EXTRACTING_SUBTITLES = "자막 추출 중"
    DOWNLOADING_AUDIO = "오디오 다운로드 중"
    TRANSCRIBING = "STT 변환 중"
    TRANSLATING = "한국어 번역 중"
    GENERATING_TTS = "TTS 생성 중"
    DONE = "완료"
    ERROR = "오류"


@dataclass
class JobResult:
    url: str
    title: str = ""
    stage: Stage = Stage.PENDING
    progress_message: str = ""
    english_text: str = ""
    korean_text: str = ""
    audio_path: Optional[str] = None
    error: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    @property
    def elapsed(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def is_done(self) -> bool:
        return self.stage in (Stage.DONE, Stage.ERROR)


def _sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)[:80]


def process_single_url(
    url: str,
    output_dir: str,
    whisper_device: str = "cuda",
    whisper_model: str = "large-v3",
    tts_rate: str = "+0%",
    on_progress: Optional[Callable[[JobResult], None]] = None,
) -> JobResult:
    """
    Process a single YouTube URL through the full pipeline.

    Args:
        url: YouTube video URL.
        output_dir: Directory for output MP3 and temp files.
        whisper_device: 'cuda' or 'cpu'.
        whisper_model: Whisper model size.
        tts_rate: TTS speech rate (e.g. '+0%').
        on_progress: Callback invoked whenever job state changes.

    Returns:
        Completed JobResult.
    """
    job = JobResult(url=url)
    temp_dir = os.path.join(output_dir, "_tmp")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    def update(stage: Stage, msg: str = "") -> None:
        job.stage = stage
        job.progress_message = msg
        if on_progress:
            on_progress(job)

    try:
        # 1. Fetch video metadata
        update(Stage.FETCHING_INFO)
        info = get_video_info(url)
        job.title = info["title"]
        video_id = info["id"]
        safe_title = _sanitize_filename(job.title)

        # 2. Try to get subtitles
        update(Stage.EXTRACTING_SUBTITLES)
        subtitle_path = extract_subtitles(url, temp_dir)

        if subtitle_path:
            update(Stage.EXTRACTING_SUBTITLES, "자막 파일 발견됨")
            job.english_text = Path(subtitle_path).read_text(encoding="utf-8")
        else:
            # 3. No subtitles — download audio and transcribe
            update(Stage.DOWNLOADING_AUDIO, "자막 없음 → 오디오 다운로드 시작")
            audio_path = download_audio(url, temp_dir)

            def whisper_progress(msg: str) -> None:
                update(Stage.TRANSCRIBING, msg)

            update(Stage.TRANSCRIBING, "faster-whisper 실행 중...")
            job.english_text = transcribe(
                audio_path,
                model_size=whisper_model,
                device=whisper_device,
                progress_callback=whisper_progress,
            )

            # Clean up downloaded audio
            try:
                os.remove(audio_path)
            except OSError:
                pass

        # 4. Translate to Korean
        def trans_progress(msg: str) -> None:
            update(Stage.TRANSLATING, msg)

        update(Stage.TRANSLATING, "번역 시작...")
        job.korean_text = translate(
            job.english_text,
            progress_callback=trans_progress,
        )

        # 5. Generate TTS
        output_filename = f"{safe_title}_{video_id}.mp3"
        output_path = os.path.join(output_dir, output_filename)

        def tts_progress(msg: str) -> None:
            update(Stage.GENERATING_TTS, msg)

        update(Stage.GENERATING_TTS, "TTS 시작...")
        generate_tts(
            job.korean_text,
            output_path,
            rate=tts_rate,
            progress_callback=tts_progress,
        )
        job.audio_path = output_path

        job.end_time = time.time()
        update(Stage.DONE, f"완료! ({job.elapsed:.0f}초)")

    except Exception as e:
        job.stage = Stage.ERROR
        job.error = str(e)
        job.end_time = time.time()
        if on_progress:
            on_progress(job)

    return job


def process_urls_parallel(
    urls: List[str],
    output_dir: str,
    whisper_device: str = "cuda",
    whisper_model: str = "large-v3",
    tts_rate: str = "+0%",
    max_workers: int = 3,
    on_progress: Optional[Callable[[JobResult], None]] = None,
) -> Dict[str, JobResult]:
    """
    Process multiple YouTube URLs in parallel (max 3 concurrent).

    Returns:
        Dict mapping URL → JobResult.
    """
    results: Dict[str, JobResult] = {}
    lock = threading.Lock()

    def progress_handler(job: JobResult) -> None:
        with lock:
            results[job.url] = job
        if on_progress:
            on_progress(job)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(urls))) as executor:
        futures = {
            executor.submit(
                process_single_url,
                url,
                output_dir,
                whisper_device,
                whisper_model,
                tts_rate,
                progress_handler,
            ): url
            for url in urls
        }

        for future in as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                with lock:
                    results[url] = result
            except Exception as e:
                with lock:
                    results[url] = JobResult(url=url, stage=Stage.ERROR, error=str(e))

    return results
