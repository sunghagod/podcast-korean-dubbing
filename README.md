# 🎙️ 영어 팟캐스트 한국어 더빙 서비스

YouTube 영어 팟캐스트 URL을 입력하면 자동으로 한국어 더빙 MP3를 생성합니다.

## 파이프라인

```
YouTube URL
  → yt-dlp 자막 추출 (없으면 오디오 다운로드 → faster-whisper STT)
  → deep-translator (Google) 한국어 번역
  → edge-tts (ko-KR-SunHiNeural) TTS 생성
  → Gradio 웹 플레이어
```

## 필수 환경

- Windows 11
- Python 3.10+
- NVIDIA GPU (CUDA 지원) — Whisper 가속용
- FFmpeg (PATH에 추가 필요)

## 설치

### 1. FFmpeg 설치
[FFmpeg 다운로드](https://ffmpeg.org/download.html) 후 PATH에 추가하거나
[winget](https://learn.microsoft.com/en-us/windows/package-manager/) 사용:
```
winget install ffmpeg
```

### 2. Python 패키지 설치
```bash
pip install -r requirements.txt
```

CUDA GPU 가속을 위해 PyTorch CUDA 버전도 설치:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 서버 실행
```bash
python app.py
```

브라우저에서 http://localhost:7860 접속

## 사용법

1. 웹 UI에서 YouTube URL 입력 (여러 개는 줄바꿈/쉼표 구분)
2. Whisper 모델 및 TTS 속도 선택
3. "처리 시작" 클릭
4. 진행 상황 실시간 확인
5. 완료 후 오디오 플레이어에서 재생 / 번역 텍스트 확인

## 옵션

| 설정 | 설명 |
|------|------|
| Whisper 모델 | `large-v3` (최고 정확도) ~ `base` (빠름) |
| 처리 장치 | `cuda` (GPU, 권장) / `cpu` |
| TTS 속도 | -10% ~ +30% |

## 프로젝트 구조

```
팟케스트 번역/
├── app.py              # Gradio 웹 UI
├── pipeline.py         # 파이프라인 오케스트레이션
├── modules/
│   ├── downloader.py   # YouTube 자막/오디오 다운로드
│   ├── transcriber.py  # Whisper STT
│   ├── translator.py   # 한국어 번역
│   └── tts.py          # edge-tts TTS 생성
├── output/             # 생성된 MP3 저장
└── requirements.txt
```

## 처리 시간 참고

| 영상 길이 | 자막 있음 | 자막 없음 (GPU) |
|-----------|-----------|----------------|
| 10분 | ~2분 | ~5분 |
| 30분 | ~5분 | ~10분 |
| 1시간 | ~10분 | ~20분 |

*RTX 3080 기준 추정치*
