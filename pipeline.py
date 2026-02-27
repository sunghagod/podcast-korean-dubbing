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

from deep_translator import GoogleTranslator

from modules.downloader import get_video_info, extract_subtitles, download_audio, download_reference_clip
from modules.transcriber import transcribe
from modules.translator import translate, translate_diarized_with_claude
from modules.tts import generate_tts, generate_tts_fish_speech, generate_tts_diarized


class Stage(str, Enum):
    PENDING = "대기 중"
    FETCHING_INFO = "영상 정보 가져오는 중"
    EXTRACTING_SUBTITLES = "자막 추출 중"
    DOWNLOADING_AUDIO = "오디오 다운로드 중"
    DOWNLOADING_REFERENCE = "레퍼런스 오디오 추출 중"
    REFERENCE_STT = "레퍼런스 STT 중"
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
    diarized_segments: Optional[List[Dict]] = None
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


def _translate_title(title: str) -> str:
    """Translate an English video title to a concise Korean title."""
    try:
        translator = GoogleTranslator(source="en", target="ko")
        return translator.translate(title) or title
    except Exception:
        return title


def process_single_url(
    url: str,
    output_dir: str,
    whisper_device: str = "cuda",
    whisper_model: str = "large-v3",
    tts_rate: str = "+0%",
    tts_voice: str = "ko-KR-SunHiNeural",
    tts_pitch: str = "+0Hz",
    use_voice_cloning: bool = False,
    checkpoint_dir: str = "",
    lib_ref_audio: str = "",
    lib_ref_text: str = "",
    use_diarization: bool = False,
    assemblyai_api_key: str = "",
    claude_api_key: str = "",
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
        tts_voice: edge-tts voice name (e.g. 'ko-KR-SunHiNeural', 'ko-KR-InJoonNeural').
        use_voice_cloning: If True, use Fish Speech for voice-cloned TTS.
        checkpoint_dir: Path to openaudio-s1-mini checkpoint directory.
        use_diarization: If True, use AssemblyAI speaker diarization.
        assemblyai_api_key: AssemblyAI API key (required when use_diarization=True).
        claude_api_key: Anthropic API key for high-quality conversational translation.
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

    ref_wav: Optional[str] = None
    _cleanup_ref_wav = False  # only cleanup ref we downloaded ourselves

    try:
        # 1. Fetch video metadata
        update(Stage.FETCHING_INFO)
        info = get_video_info(url)
        job.title = info["title"]
        video_id = info["id"]
        safe_title = _sanitize_filename(job.title)

        # 1b. Reference audio for voice cloning
        ref_text = ""
        if use_voice_cloning and checkpoint_dir:
            if lib_ref_audio and lib_ref_text and os.path.exists(lib_ref_audio):
                # Use pre-existing library voice — no download needed
                ref_wav = lib_ref_audio
                ref_text = lib_ref_text
            else:
                update(Stage.DOWNLOADING_REFERENCE, "레퍼런스 오디오 30초 다운로드 중...")
                try:
                    ref_wav = download_reference_clip(url, temp_dir, duration=30)
                    _cleanup_ref_wav = True
                    update(Stage.REFERENCE_STT, "레퍼런스 STT 실행 중...")
                    ref_text = transcribe(
                        ref_wav,
                        model_size=whisper_model,
                        device=whisper_device,
                    )
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        "레퍼런스 오디오/STT 실패, edge-tts로 fallback: %s", e
                    )
                    use_voice_cloning = False
                    ref_wav = None

        # 2. Diarization 분기
        if use_diarization and assemblyai_api_key:
            from modules.diarizer import transcribe_with_diarization

            # 항상 오디오 다운로드 (자막 추출 건너뜀)
            update(Stage.DOWNLOADING_AUDIO, "화자 구분 모드 → 오디오 다운로드 시작")
            audio_path_for_diarize = download_audio(url, temp_dir)

            def diarize_progress(msg: str) -> None:
                update(Stage.TRANSCRIBING, msg)

            update(Stage.TRANSCRIBING, "AssemblyAI 화자 구분 STT 중...")
            utterances = transcribe_with_diarization(
                audio_path_for_diarize,
                api_key=assemblyai_api_key,
                progress_callback=diarize_progress,
            )

            try:
                os.remove(audio_path_for_diarize)
            except OSError:
                pass

            # 4d. 각 utterance 번역
            def trans_progress_d(msg: str) -> None:
                update(Stage.TRANSLATING, msg)

            update(Stage.TRANSLATING, "화자별 번역 시작...")
            if claude_api_key:
                translated_segments = translate_diarized_with_claude(
                    utterances,
                    api_key=claude_api_key,
                    progress_callback=trans_progress_d,
                )
            else:
                translated_segments: List[Dict] = []
                for idx, utt in enumerate(utterances):
                    update(Stage.TRANSLATING, f"번역 중... ({idx + 1}/{len(utterances)})")
                    ko_text = translate(utt["text"], progress_callback=None)
                    translated_segments.append({
                        "speaker": utt["speaker"],
                        "text": ko_text,
                        "start_ms": utt["start_ms"],
                        "end_ms": utt["end_ms"],
                    })

            job.diarized_segments = translated_segments

            # 원문 / 번역문 텍스트 조합
            job.english_text = "\n".join(
                f"[화자 {u['speaker']}] {u['text']}" for u in utterances
            )
            job.korean_text = "\n".join(
                f"[화자 {s['speaker']}] {s['text']}" for s in translated_segments
            )

            # 5d. Generate diarized TTS
            korean_title = _translate_title(job.title)
            safe_korean_title = _sanitize_filename(korean_title)
            output_filename = f"{safe_korean_title}_{video_id}.mp3"
            output_path = os.path.join(output_dir, output_filename)

            def tts_progress_d(msg: str) -> None:
                update(Stage.GENERATING_TTS, msg)

            update(Stage.GENERATING_TTS, "화자 구분 TTS 시작...")
            generate_tts_diarized(
                translated_segments,
                output_path,
                rate=tts_rate,
                progress_callback=tts_progress_d,
            )
            job.audio_path = output_path

        else:
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
                claude_api_key=claude_api_key,
            )

            # 5. Generate TTS
            korean_title = _translate_title(job.title)
            safe_korean_title = _sanitize_filename(korean_title)
            output_filename = f"{safe_korean_title}_{video_id}.mp3"
            output_path = os.path.join(output_dir, output_filename)

            def tts_progress(msg: str) -> None:
                update(Stage.GENERATING_TTS, msg)

            update(Stage.GENERATING_TTS, "TTS 시작...")
            if use_voice_cloning and ref_wav and ref_text and checkpoint_dir:
                try:
                    generate_tts_fish_speech(
                        job.korean_text,
                        output_path,
                        reference_audio_path=ref_wav,
                        reference_text=ref_text,
                        checkpoint_dir=checkpoint_dir,
                        rate=tts_rate,
                        progress_callback=tts_progress,
                    )
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Fish Speech TTS 실패, edge-tts로 fallback: %s", e
                    )
                    generate_tts(
                        job.korean_text,
                        output_path,
                        rate=tts_rate,
                        voice=tts_voice,
                        pitch=tts_pitch,
                        progress_callback=tts_progress,
                    )
            else:
                generate_tts(
                    job.korean_text,
                    output_path,
                    rate=tts_rate,
                    voice=tts_voice,
                    pitch=tts_pitch,
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
    finally:
        # Only cleanup reference audio we downloaded ourselves
        if ref_wav and _cleanup_ref_wav:
            try:
                os.remove(ref_wav)
            except OSError:
                pass

    return job


def process_urls_parallel(
    urls: List[str],
    output_dir: str,
    whisper_device: str = "cuda",
    whisper_model: str = "large-v3",
    tts_rate: str = "+0%",
    tts_voice: str = "ko-KR-SunHiNeural",
    tts_pitch: str = "+0Hz",
    use_voice_cloning: bool = False,
    checkpoint_dir: str = "",
    lib_ref_audio: str = "",
    lib_ref_text: str = "",
    use_diarization: bool = False,
    assemblyai_api_key: str = "",
    claude_api_key: str = "",
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
                tts_voice,
                tts_pitch,
                use_voice_cloning,
                checkpoint_dir,
                lib_ref_audio,
                lib_ref_text,
                use_diarization,
                assemblyai_api_key,
                claude_api_key,
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
