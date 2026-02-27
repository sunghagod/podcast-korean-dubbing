"""
Gradio web UI – YouTube-style audio player
"""

import hashlib
import os
import re
import threading
import time
import urllib.parse
from typing import Dict, List, Optional

import gradio as gr

from modules.downloader import download_voice_clip
from modules.transcriber import transcribe
from pipeline import JobResult, Stage, process_urls_parallel

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "openaudio-s1-mini")
VOICES_DIR = os.path.join(PROJECT_DIR, "references")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)


def _load_config_env() -> dict:
    """config.env 파일에서 KEY=VALUE 형식의 설정을 읽어 반환."""
    config = {}
    config_path = os.path.join(PROJECT_DIR, "config.env")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    config[key.strip()] = value.strip()
    return config


_CONFIG = _load_config_env()
DEFAULT_CLAUDE_KEY = _CONFIG.get("CLAUDE_API_KEY", "")
DEFAULT_ASSEMBLYAI_KEY = _CONFIG.get("ASSEMBLYAI_API_KEY", "")

MAX_SLOTS = 3

_jobs: Dict[str, JobResult] = {}
_lock = threading.Lock()

# ── File URL helper ───────────────────────────────────────────────────────────

def make_audio_url(filepath: str) -> str:
    fp = os.path.abspath(filepath).replace("\\", "/")
    return f"/gradio_api/file={urllib.parse.quote(fp, safe=':/')}"


def get_uid(filepath: str) -> str:
    basename = os.path.basename(filepath)
    m = re.search(r"_([A-Za-z0-9_-]{8,12})\.mp3$", basename)
    if m:
        return m.group(1)
    return hashlib.md5(filepath.encode()).hexdigest()[:10]


# ── YouTube-style player HTML generator ──────────────────────────────────────

def make_yt_player(uid: str, title: str, filepath: str) -> str:
    audio_url = make_audio_url(filepath)
    # escape single quotes in title for inline JS / HTML attribute safety
    safe_title = title.replace("'", "&#39;").replace('"', "&quot;")

    return f"""
<div id="ytp-{uid}" style="
    background:#0f0f0f;
    border-radius:12px;
    overflow:hidden;
    color:#fff;
    font-family:'YouTube Sans','Roboto','Noto Sans KR',Arial,sans-serif;
    box-shadow:0 4px 24px rgba(0,0,0,0.6);
    max-width:100%;
    margin-bottom:8px;
">
  <!-- Title -->
  <div style="
      padding:12px 16px 4px;
      font-size:15px;
      font-weight:500;
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
      letter-spacing:.01em;
  ">{safe_title}</div>

  <!-- Progress bar -->
  <div style="padding:10px 0 2px;">
    <div id="ybar-{uid}"
         onclick="ytSeek('{uid}',event)"
         style="
             height:4px;
             background:rgba(255,255,255,0.2);
             cursor:pointer;
             position:relative;
             transition:height .1s;
         "
         onmouseenter="
             this.style.height='7px';
             var h=document.getElementById('yhandle-{uid}');
             if(h) h.style.opacity='1';
         "
         onmouseleave="
             this.style.height='4px';
             var h=document.getElementById('yhandle-{uid}');
             if(h) h.style.opacity='0';
         ">
      <div id="yfill-{uid}" style="
          height:100%;
          width:0%;
          background:#f00;
          position:relative;
          pointer-events:none;
      ">
        <div id="yhandle-{uid}" style="
            position:absolute;
            right:-7px;
            top:50%;
            transform:translateY(-50%);
            width:14px;
            height:14px;
            background:#f00;
            border-radius:50%;
            opacity:0;
            transition:opacity .15s;
        "></div>
      </div>
    </div>
  </div>

  <!-- Controls bar -->
  <div style="display:flex;align-items:center;padding:4px 8px 16px;gap:2px;">

    <!-- Play / Pause -->
    <button onclick="ytPlay('{uid}')" title="재생 (k)"
            style="background:none;border:none;cursor:pointer;padding:6px;color:#fff;display:flex;align-items:center;border-radius:50%;"
            onmouseenter="this.style.background='rgba(255,255,255,0.1)'"
            onmouseleave="this.style.background='none'">
      <svg id="yicon-{uid}" width="40" height="40" viewBox="0 0 24 24" fill="white">
        <path d="M8 5v14l11-7z"/>
      </svg>
    </button>

    <!-- Replay 10 s -->
    <button onclick="ytSkip('{uid}',-10)" title="10초 뒤로 (j)"
            style="background:none;border:none;cursor:pointer;padding:6px;color:#fff;display:flex;align-items:center;border-radius:50%;"
            onmouseenter="this.style.background='rgba(255,255,255,0.1)'"
            onmouseleave="this.style.background='none'">
      <svg width="30" height="30" viewBox="0 0 24 24">
        <path d="M11.99 5V1l-5 5 5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6h-2c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z" fill="white"/>
        <text x="12" y="17.5" font-size="5.5" text-anchor="middle" fill="white" font-weight="bold" font-family="Arial,sans-serif">10</text>
      </svg>
    </button>

    <!-- Forward 10 s -->
    <button onclick="ytSkip('{uid}',10)" title="10초 앞으로 (l)"
            style="background:none;border:none;cursor:pointer;padding:6px;color:#fff;display:flex;align-items:center;border-radius:50%;"
            onmouseenter="this.style.background='rgba(255,255,255,0.1)'"
            onmouseleave="this.style.background='none'">
      <svg width="30" height="30" viewBox="0 0 24 24">
        <path d="M12.01 5V1l5 5-5 5V7c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6h2c0 4.42-3.58 8-8 8s-8-3.58-8-8 3.58-8 8-8z" fill="white"/>
        <text x="12" y="17.5" font-size="5.5" text-anchor="middle" fill="white" font-weight="bold" font-family="Arial,sans-serif">10</text>
      </svg>
    </button>

    <!-- Mute toggle -->
    <button onclick="ytMute('{uid}')"
            style="background:none;border:none;cursor:pointer;padding:6px;color:#fff;display:flex;align-items:center;border-radius:50%;"
            onmouseenter="this.style.background='rgba(255,255,255,0.1)'"
            onmouseleave="this.style.background='none'">
      <svg id="yvolicon-{uid}" width="26" height="26" viewBox="0 0 24 24" fill="white">
        <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>
      </svg>
    </button>

    <!-- Volume slider -->
    <input type="range" id="yvs-{uid}" min="0" max="100" value="100"
           oninput="ytVol('{uid}',this.value)"
           style="width:72px;accent-color:#fff;cursor:pointer;vertical-align:middle;">

    <!-- Time display -->
    <span id="ytime-{uid}"
          style="font-size:13px;color:#bbb;white-space:nowrap;margin-left:8px;">
      0:00 / 0:00
    </span>

    <div style="flex:1;"></div>

    <!-- Speed -->
    <select onchange="ytSpeed('{uid}',this.value)"
            style="background:#272727;color:#fff;border:1px solid #555;padding:5px 8px;border-radius:4px;cursor:pointer;font-size:13px;outline:none;font-family:inherit;">
      <option value="0.5">0.5x</option>
      <option value="0.75">0.75x</option>
      <option value="1" selected>1x</option>
      <option value="1.25">1.25x</option>
      <option value="1.5">1.5x</option>
      <option value="2">2x</option>
    </select>
  </div>

  <!-- Hidden audio element – picked up by MutationObserver -->
  <audio id="ya-{uid}" preload="metadata" src="{audio_url}"></audio>
</div>
"""


# ── Global JS injected once at page load ──────────────────────────────────────

GLOBAL_JS = """
// ── Control functions ──
window.ytPlay = function(id) {
  var a = document.getElementById('ya-'+id);
  if (a) { a.paused ? a.play() : a.pause(); }
};

window.ytSkip = function(id, s) {
  var a = document.getElementById('ya-'+id);
  if (a) { a.currentTime = Math.max(0, Math.min(a.duration||0, a.currentTime+s)); }
};

window.ytSeek = function(id, e) {
  var bar = document.getElementById('ybar-'+id);
  var a   = document.getElementById('ya-'+id);
  if (!bar || !a || !a.duration) return;
  var r = bar.getBoundingClientRect();
  a.currentTime = Math.max(0, Math.min(1, (e.clientX-r.left)/r.width)) * a.duration;
};

window.ytVol = function(id, v) {
  var a = document.getElementById('ya-'+id);
  if (a) { a.volume = v/100; a.muted = (v==0); }
};

window.ytMute = function(id) {
  var a = document.getElementById('ya-'+id);
  if (!a) return;
  a.muted = !a.muted;
  var s = document.getElementById('yvs-'+id);
  if (s) s.value = a.muted ? 0 : Math.round(a.volume*100);
};

window.ytSpeed = function(id, v) {
  var a = document.getElementById('ya-'+id);
  if (a) a.playbackRate = parseFloat(v);
};

window._ytFmt = function(s) {
  s = Math.floor(s||0);
  var h=Math.floor(s/3600), m=Math.floor((s%3600)/60), sec=s%60;
  if (h>0) return h+':'+String(m).padStart(2,'0')+':'+String(sec).padStart(2,'0');
  return m+':'+String(sec).padStart(2,'0');
};

// ── Per-player event-listener setup (called by observer) ──
window._setupYTPlayer = function(uid, audio) {
  if (audio._ytReady) return;
  audio._ytReady = true;

  var PLAY_PATH  = '<path d="M8 5v14l11-7z"/>';
  var PAUSE_PATH = '<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>';
  var VOL_HIGH   = '<path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>';
  var VOL_LOW    = '<path d="M18.5 12c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM5 9v6h4l5 5V4L9 9H5z"/>';
  var VOL_MUTE   = '<path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/>';

  function setTime() {
    var el = document.getElementById('ytime-'+uid);
    if (el) el.textContent = window._ytFmt(audio.currentTime) + ' / ' + window._ytFmt(audio.duration||0);
  }
  function setFill() {
    var f = document.getElementById('yfill-'+uid);
    if (f && audio.duration) f.style.width = (audio.currentTime/audio.duration*100)+'%';
  }
  function setIcon(path) {
    var el = document.getElementById('yicon-'+uid);
    if (el) el.innerHTML = path;
  }
  function setVolIcon() {
    var el = document.getElementById('yvolicon-'+uid);
    if (!el) return;
    el.innerHTML = (audio.muted||audio.volume===0) ? VOL_MUTE : (audio.volume<0.5 ? VOL_LOW : VOL_HIGH);
  }

  audio.addEventListener('loadedmetadata', setTime);
  audio.addEventListener('timeupdate', function() { setFill(); setTime(); });
  audio.addEventListener('play',    function() { setIcon(PAUSE_PATH); });
  audio.addEventListener('pause',   function() { setIcon(PLAY_PATH); });
  audio.addEventListener('ended',   function() { setIcon(PLAY_PATH); });
  audio.addEventListener('volumechange', setVolIcon);
};

// ── MutationObserver: auto-setup players when inserted into DOM ──
var _ytObs = new MutationObserver(function(muts) {
  for (var i=0; i<muts.length; i++) {
    for (var j=0; j<muts[i].addedNodes.length; j++) {
      var node = muts[i].addedNodes[j];
      if (!node || node.nodeType !== 1) continue;
      // node itself
      if (node.tagName==='AUDIO' && node.id && node.id.indexOf('ya-')===0) {
        window._setupYTPlayer(node.id.slice(3), node);
      }
      // descendants
      if (node.querySelectorAll) {
        var els = node.querySelectorAll('audio[id^="ya-"]');
        for (var k=0; k<els.length; k++) {
          window._setupYTPlayer(els[k].id.slice(3), els[k]);
        }
      }
    }
  }
});

setTimeout(function() {
  if (document.body) _ytObs.observe(document.body, { childList:true, subtree:true });
}, 200);
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_urls(raw: str) -> List[str]:
    raw = raw.replace(",", "\n")
    urls = [u.strip() for u in raw.splitlines() if u.strip()]
    valid = [u for u in urls if re.match(r"https?://(www\.)?(youtube\.com|youtu\.be)/", u)]
    return list(dict.fromkeys(valid))


def _stage_emoji(stage: Stage) -> str:
    return {
        Stage.PENDING: "⏳", Stage.FETCHING_INFO: "🔍",
        Stage.EXTRACTING_SUBTITLES: "📝", Stage.DOWNLOADING_AUDIO: "⬇️",
        Stage.DOWNLOADING_REFERENCE: "🎵", Stage.REFERENCE_STT: "🔎",
        Stage.TRANSCRIBING: "🎙️", Stage.TRANSLATING: "🌐",
        Stage.GENERATING_TTS: "🔊", Stage.DONE: "✅", Stage.ERROR: "❌",
    }.get(stage, "•")


def _build_status_md(jobs: Dict[str, JobResult]) -> str:
    if not jobs:
        return "_URL을 입력하고 처리 시작을 클릭하세요._"
    lines = ["| # | 제목 | 상태 | 진행 | 경과 |", "|---|------|------|------|------|"]
    for i, (url, job) in enumerate(jobs.items(), 1):
        title = (job.title or url)[:35]
        progress = (job.progress_message or "")[:40]
        lines.append(
            f"| {i} | {title} | {_stage_emoji(job.stage)} {job.stage.value} "
            f"| {progress} | {job.elapsed:.0f}초 |"
        )
    return "\n".join(lines)


def _slot_updates(jobs: Dict[str, JobResult]):
    done = [j for j in jobs.values() if j.stage == Stage.DONE]
    updates = []
    for i in range(MAX_SLOTS):
        if i < len(done):
            job = done[i]
            uid = get_uid(job.audio_path) if job.audio_path else f"slot{i}"
            player_html = make_yt_player(uid, job.title, job.audio_path) if job.audio_path else ""
            updates.append((player_html, job.korean_text, job.english_text, True))
        else:
            updates.append(("", "", "", False))
    return updates


def _make_outputs(status="", snap=None, btn_interactive=True):
    snap = snap or {}
    slot_data = _slot_updates(snap)
    out = [gr.update(value=status), gr.update(interactive=btn_interactive)]
    for player_html, ko_text, en_text, visible in slot_data:
        out += [
            gr.update(value=player_html),
            gr.update(value=ko_text),
            gr.update(value=en_text),
            gr.update(visible=visible),
        ]
    return out


# ── Processing ────────────────────────────────────────────────────────────────

# (voice_id, pitch)
VOICE_MAP = {
    "여성 — SunHi": ("ko-KR-SunHiNeural", "+0Hz"),
    "남성 — InJoon": ("ko-KR-InJoonNeural", "+0Hz"),
    "남성 — InJoon (저음 -10Hz)": ("ko-KR-InJoonNeural", "-10Hz"),
    "남성 — InJoon (극저음 -20Hz)": ("ko-KR-InJoonNeural", "-20Hz"),
    "남성 — Hyunsu (다국어)": ("ko-KR-HyunsuMultilingualNeural", "+0Hz"),
    "남성 — Hyunsu (저음 -10Hz)": ("ko-KR-HyunsuMultilingualNeural", "-10Hz"),
}


def start_processing(url_input, whisper_model, whisper_device, tts_rate, tts_voice_label, ref_voice_id, use_voice_cloning,
                     claude_api_key, use_diarization, assemblyai_api_key):
    global _jobs
    urls = _parse_urls(url_input)
    if not urls:
        yield _make_outputs(status="⚠️ 유효한 YouTube URL을 입력해 주세요.", btn_interactive=True)
        return

    if use_diarization and not (assemblyai_api_key or "").strip():
        yield _make_outputs(status="⚠️ 화자 구분 더빙을 사용하려면 AssemblyAI API 키를 입력해 주세요.", btn_interactive=True)
        return

    with _lock:
        _jobs = {}

    rate_map = {"느림 (-10%)": "-10%", "보통 (+0%)": "+0%", "빠름 (+15%)": "+15%", "매우 빠름 (+30%)": "+30%"}
    rate = rate_map.get(tts_rate, "+0%")
    voice, pitch = VOICE_MAP.get(tts_voice_label, ("ko-KR-InJoonNeural", "+0Hz"))

    # Resolve library voice reference files
    lib_ref_audio, lib_ref_text = "", ""
    if use_voice_cloning and ref_voice_id:
        voice_dir = os.path.join(VOICES_DIR, ref_voice_id)
        wav = os.path.join(voice_dir, "sample.wav")
        lab = os.path.join(voice_dir, "sample.lab")
        if os.path.exists(wav) and os.path.exists(lab):
            lib_ref_audio = wav
            lib_ref_text = open(lab, encoding="utf-8").read().strip()

    def progress_cb(job: JobResult):
        with _lock:
            _jobs[job.url] = job

    thread = threading.Thread(
        target=process_urls_parallel,
        kwargs=dict(
            urls=urls, output_dir=OUTPUT_DIR,
            whisper_device=whisper_device, whisper_model=whisper_model,
            tts_rate=rate,
            tts_voice=voice,
            tts_pitch=pitch,
            use_voice_cloning=use_voice_cloning,
            checkpoint_dir=CHECKPOINT_DIR,
            lib_ref_audio=lib_ref_audio,
            lib_ref_text=lib_ref_text,
            use_diarization=use_diarization,
            assemblyai_api_key=(assemblyai_api_key or "").strip(),
            claude_api_key=(claude_api_key or "").strip(),
            on_progress=progress_cb,
        ),
        daemon=True,
    )
    thread.start()

    while thread.is_alive():
        time.sleep(2)
        with _lock:
            snap = dict(_jobs)
        yield _make_outputs(status=_build_status_md(snap), snap=snap, btn_interactive=False)

    with _lock:
        snap = dict(_jobs)
    yield _make_outputs(status=_build_status_md(snap), snap=snap, btn_interactive=True)


# ── Voice Library ─────────────────────────────────────────────────────────────

def _parse_time(s: str) -> float:
    """Convert 'MM:SS', 'H:MM:SS', or plain seconds string to float seconds."""
    s = s.strip()
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        else:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        raise ValueError(f"시간 형식 오류: '{s}' — MM:SS 또는 초 단위로 입력하세요.")


def _list_ref_voices() -> List[str]:
    """Return names of all valid voice library entries."""
    if not os.path.exists(VOICES_DIR):
        return []
    voices = []
    for name in sorted(os.listdir(VOICES_DIR)):
        d = os.path.join(VOICES_DIR, name)
        if os.path.isdir(d):
            wav = os.path.join(d, "sample.wav")
            lab = os.path.join(d, "sample.lab")
            if os.path.exists(wav) and os.path.exists(lab):
                voices.append(name)
    return voices


def list_voice_library():
    voices = _list_ref_voices()
    if not voices:
        md = "_등록된 목소리가 없습니다. 아래 폼으로 추가하세요._"
    else:
        rows = ["| # | 이름 | 크기 |", "|---|------|------|"]
        for i, name in enumerate(voices, 1):
            wav = os.path.join(VOICES_DIR, name, "sample.wav")
            size_mb = os.path.getsize(wav) / 1024 / 1024 if os.path.exists(wav) else 0
            rows.append(f"| {i} | **{name}** | {size_mb:.1f} MB |")
        md = "\n".join(rows)
    choices_update = gr.update(choices=voices, value=voices[0] if voices else None)
    return md, choices_update, choices_update


def add_voice_to_library(url, start_time, end_time, voice_name, whisper_device):
    """Generator: downloads clip, transcribes it, saves to references/voice_name/."""
    voice_name = (voice_name or "").strip()
    if not voice_name:
        yield "⚠️ 목소리 이름을 입력하세요.", *list_voice_library()
        return
    if not url or not url.strip():
        yield "⚠️ YouTube URL을 입력하세요.", *list_voice_library()
        return

    voice_dir = os.path.join(VOICES_DIR, voice_name)
    if os.path.exists(voice_dir):
        yield f"⚠️ '{voice_name}' 이름이 이미 존재합니다. 다른 이름을 사용하세요.", *list_voice_library()
        return

    try:
        start_sec = _parse_time(start_time)
        end_sec = _parse_time(end_time)
    except ValueError as e:
        yield f"⚠️ {e}", *list_voice_library()
        return

    if end_sec <= start_sec:
        yield "⚠️ 끝 시간이 시작 시간보다 커야 합니다.", *list_voice_library()
        return
    if end_sec - start_sec > 60:
        yield "⚠️ 최대 60초 구간까지 지원합니다.", *list_voice_library()
        return

    os.makedirs(voice_dir, exist_ok=True)
    wav_path = os.path.join(voice_dir, "sample.wav")
    lab_path = os.path.join(voice_dir, "sample.lab")

    try:
        yield f"⬇️ 오디오 클립 다운로드 중... ({start_time} ~ {end_time})", *list_voice_library()
        download_voice_clip(url.strip(), start_sec, end_sec, wav_path)

        yield f"🎙️ STT 변환 중... (한국어)", *list_voice_library()
        ref_text = transcribe(wav_path, device=whisper_device, language="ko")

        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(ref_text)

        yield f"✅ '{voice_name}' 등록 완료!\n\n**STT 결과:** {ref_text[:200]}", *list_voice_library()

    except Exception as e:
        import shutil
        shutil.rmtree(voice_dir, ignore_errors=True)
        yield f"❌ 오류: {e}", *list_voice_library()


def delete_voice_from_library(voice_name):
    if voice_name:
        import shutil
        shutil.rmtree(os.path.join(VOICES_DIR, voice_name), ignore_errors=True)
    return list_voice_library()


# ── History ───────────────────────────────────────────────────────────────────

def _mp3_files() -> List[str]:
    return sorted(
        [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".mp3")],
        key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)),
        reverse=True,
    )


def list_history():
    mp3_files = _mp3_files()
    if not mp3_files:
        return "_아직 생성된 파일이 없습니다._", "", gr.update(choices=[], value=None), gr.update(value=None, visible=False)

    lines = ["| # | 파일명 | 크기 |", "|---|--------|------|"]
    for i, fname in enumerate(mp3_files[:10], 1):
        size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, fname)) / 1024 / 1024
        lines.append(f"| {i} | {fname[:60]} | {size_mb:.1f} MB |")

    latest = os.path.join(OUTPUT_DIR, mp3_files[0])
    uid = get_uid(latest)
    title = re.sub(r"_[A-Za-z0-9_-]{8,12}\.mp3$", "", mp3_files[0]).replace("_", " ")
    player_html = make_yt_player(uid, title, latest)
    return "\n".join(lines), player_html, gr.update(choices=mp3_files, value=mp3_files[0]), gr.update(value=latest, visible=True)


def _get_download_path(fname: str):
    if fname:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            return gr.update(value=path, visible=True)
    return gr.update(value=None, visible=False)


def open_output_folder():
    import subprocess
    folder = OUTPUT_DIR.replace("/", "\\")
    subprocess.Popen(["explorer", folder])


def delete_file(fname: str):
    if fname:
        try:
            os.remove(os.path.join(OUTPUT_DIR, fname))
        except OSError:
            pass
    return list_history()


def delete_all_files():
    for f in _mp3_files():
        try:
            os.remove(os.path.join(OUTPUT_DIR, f))
        except OSError:
            pass
    return list_history()


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="🎙️ 팟캐스트 한국어 더빙") as demo:
    gr.HTML('<h2 style="margin:0 0 4px">🎙️ 영어 팟캐스트 한국어 더빙 서비스</h2>')
    gr.HTML("<p>YouTube URL을 입력하면 영어 팟캐스트를 자동으로 한국어로 더빙합니다.</p>")

    with gr.Row():
        with gr.Column(scale=2):
            url_input = gr.Textbox(
                label="YouTube URL (여러 개는 줄바꿈 또는 쉼표로 구분)",
                placeholder="https://www.youtube.com/watch?v=...\nhttps://youtu.be/...",
                lines=4,
            )
            voice_clone_cb = gr.Checkbox(
                label="🎤 원본 화자 목소리로 더빙 (Fish Speech — checkpoints/openaudio-s1-mini 필요)",
                value=False,
            )
            ref_voice_selector = gr.Dropdown(
                label="📚 라이브러리 목소리 선택 (voice cloning 활성 시 사용 — 비어있으면 영상에서 자동 추출)",
                choices=_list_ref_voices(),
                value=None,
            )
            claude_key_input = gr.Textbox(
                label="🤖 Claude API 키 (번역 품질 향상 — 없으면 Google 번역 사용)",
                placeholder="sk-ant-...",
                type="password",
                value=DEFAULT_CLAUDE_KEY,
            )
            diarization_cb = gr.Checkbox(
                label="🎙️ 화자 구분 더빙 (AssemblyAI — 인터뷰어/게스트 목소리 분리)",
                value=False,
            )
            assemblyai_key_input = gr.Textbox(
                label="AssemblyAI API 키",
                placeholder="your_assemblyai_api_key_here",
                type="password",
                visible=False,
                value=DEFAULT_ASSEMBLYAI_KEY,
            )
            diarization_cb.change(
                fn=lambda checked: gr.update(visible=checked),
                inputs=[diarization_cb],
                outputs=[assemblyai_key_input],
            )
            with gr.Row():
                whisper_model = gr.Dropdown(
                    label="Whisper 모델", choices=["large-v3", "medium", "small", "base"], value="large-v3"
                )
                whisper_device = gr.Dropdown(
                    label="처리 장치", choices=["cuda", "cpu"], value="cuda"
                )
                tts_rate = gr.Dropdown(
                    label="TTS 속도",
                    choices=["느림 (-10%)", "보통 (+0%)", "빠름 (+15%)", "매우 빠름 (+30%)"],
                    value="보통 (+0%)",
                )
                tts_voice = gr.Dropdown(
                    label="TTS 목소리",
                    choices=list(VOICE_MAP.keys()),
                    value="남성 — InJoon (극저음 -20Hz)",
                )
            start_btn = gr.Button("🚀 처리 시작", variant="primary", size="lg")

        with gr.Column(scale=3):
            status_md = gr.Markdown("_URL을 입력하고 처리 시작을 클릭하세요._")

    gr.HTML("<hr>")

    with gr.Tab("처리 결과"):
        slot_players   = []
        slot_ko_texts  = []
        slot_en_texts  = []
        slot_groups    = []

        for i in range(MAX_SLOTS):
            with gr.Group(visible=False) as grp:
                player = gr.HTML("")
                with gr.Accordion("한국어 번역 보기", open=False):
                    ko_text = gr.Textbox(label="한국어", lines=8, max_lines=20, interactive=False)
                with gr.Accordion("영어 원문 보기", open=False):
                    en_text = gr.Textbox(label="English", lines=8, max_lines=20, interactive=False)

            slot_players.append(player)
            slot_ko_texts.append(ko_text)
            slot_en_texts.append(en_text)
            slot_groups.append(grp)

    with gr.Tab("🎤 목소리 라이브러리"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 새 목소리 추가")
                lib_url = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
                with gr.Row():
                    lib_start = gr.Textbox(label="시작 시간 (MM:SS)", placeholder="0:30", scale=1)
                    lib_end   = gr.Textbox(label="끝 시간 (MM:SS)", placeholder="1:00", scale=1)
                    lib_name  = gr.Textbox(label="목소리 이름", placeholder="나레이터", scale=2)
                lib_device = gr.Dropdown(label="STT 장치", choices=["cuda", "cpu"], value="cuda")
                with gr.Row():
                    lib_add_btn    = gr.Button("🎙️ 목소리 추출 & 등록", variant="primary")
                    lib_delete_btn = gr.Button("🗑️ 선택 삭제", variant="stop")
                lib_status = gr.Markdown("_URL과 구간을 입력하고 '목소리 추출 & 등록'을 클릭하세요._")

            with gr.Column(scale=3):
                gr.Markdown("### 등록된 목소리")
                lib_table      = gr.Markdown()
                lib_selector   = gr.Dropdown(label="삭제할 목소리 선택", choices=[])
                lib_refresh_btn = gr.Button("🔄 새로고침")

        lib_add_outputs    = [lib_status, lib_table, lib_selector, ref_voice_selector]
        lib_delete_outputs = [lib_table, lib_selector, ref_voice_selector]

        def _list_voice_library_3():
            md, sel, main = list_voice_library()
            return md, sel, main

        def _delete_voice(name):
            md, sel, main = delete_voice_from_library(name)
            return md, sel, main

        lib_add_btn.click(
            fn=add_voice_to_library,
            inputs=[lib_url, lib_start, lib_end, lib_name, lib_device],
            outputs=lib_add_outputs,
        )
        lib_delete_btn.click(fn=_delete_voice, inputs=[lib_selector], outputs=lib_delete_outputs)
        lib_refresh_btn.click(fn=_list_voice_library_3, outputs=lib_delete_outputs)
        demo.load(fn=_list_voice_library_3, outputs=lib_delete_outputs)

    with gr.Tab("처리 히스토리"):
        history_md     = gr.Markdown()
        history_player = gr.HTML()
        with gr.Row():
            file_selector  = gr.Dropdown(label="파일 선택", choices=[], scale=4)
            open_folder_btn = gr.Button("📂 폴더 열기", scale=1)
            delete_btn     = gr.Button("🗑️ 선택 삭제", variant="stop", scale=1)
            delete_all_btn = gr.Button("🗑️ 전체 삭제", variant="stop", scale=1)
            refresh_btn    = gr.Button("🔄 새로고침", scale=1)
        download_file = gr.File(label="⬇️ 다운로드", interactive=False, visible=False)

        history_outputs = [history_md, history_player, file_selector, download_file]
        refresh_btn.click(fn=list_history, outputs=history_outputs)
        delete_btn.click(fn=delete_file, inputs=[file_selector], outputs=history_outputs)
        delete_all_btn.click(fn=delete_all_files, outputs=history_outputs)
        file_selector.change(fn=_get_download_path, inputs=[file_selector], outputs=[download_file])
        open_folder_btn.click(fn=open_output_folder)
        demo.load(fn=list_history, outputs=history_outputs)

    # Wire outputs
    all_outputs = [status_md, start_btn]
    for i in range(MAX_SLOTS):
        all_outputs += [slot_players[i], slot_ko_texts[i], slot_en_texts[i], slot_groups[i]]

    start_btn.click(
        fn=start_processing,
        inputs=[url_input, whisper_model, whisper_device, tts_rate, tts_voice, ref_voice_selector, voice_clone_cb,
                claude_key_input, diarization_cb, assemblyai_key_input],
        outputs=all_outputs,
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        allowed_paths=[OUTPUT_DIR],
        js=GLOBAL_JS,
    )
