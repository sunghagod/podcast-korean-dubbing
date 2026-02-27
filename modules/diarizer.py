"""
AssemblyAI 기반 화자 구분(Speaker Diarization) 모듈.
오디오 파일을 업로드하고, 화자별 발화 목록을 반환한다.
"""

from typing import Callable, Dict, List, Optional


def transcribe_with_diarization(
    audio_path: str,
    api_key: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[Dict]:
    """
    AssemblyAI를 사용해 화자 구분 STT를 수행한다.

    Args:
        audio_path: 로컬 오디오 파일 경로.
        api_key: AssemblyAI API 키.
        progress_callback: 진행 상황 콜백 (메시지 문자열 수신).

    Returns:
        발화 목록:
        [{"speaker": "A", "text": "...", "start_ms": 1200, "end_ms": 4500}, ...]
    """
    try:
        import assemblyai as aai
    except ImportError:
        raise RuntimeError(
            "assemblyai 패키지가 설치되지 않았습니다. pip install assemblyai"
        )

    if not api_key or not api_key.strip():
        raise ValueError("AssemblyAI API 키가 비어 있습니다.")

    aai.settings.api_key = api_key.strip()

    if progress_callback:
        progress_callback("AssemblyAI 업로드 중...")

    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(speaker_labels=True, speech_models=["universal-2"])
    transcript = transcriber.transcribe(audio_path, config)

    if transcript.error:
        raise RuntimeError(f"AssemblyAI 오류: {transcript.error}")

    utterances = transcript.utterances or []

    if progress_callback:
        progress_callback(f"화자 구분 완료: {len(utterances)}개 발화")

    return [
        {
            "speaker": u.speaker,
            "text": u.text,
            "start_ms": u.start,
            "end_ms": u.end,
        }
        for u in utterances
    ]
