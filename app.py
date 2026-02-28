"""
Gradio web UI – Modern Dark Redesign (v2)
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

# ── File URL helper ────────────────────────────────────────────────────────────

def make_audio_url(filepath: str) -> str:
    fp = os.path.abspath(filepath).replace("\\", "/")
    return f"/gradio_api/file={urllib.parse.quote(fp, safe=':/')}"


def get_uid(filepath: str) -> str:
    basename = os.path.basename(filepath)
    m = re.search(r"_([A-Za-z0-9_-]{8,12})\.mp3$", basename)
    if m:
        return m.group(1)
    return hashlib.md5(filepath.encode()).hexdigest()[:10]


# ── Modern Dark Audio Player ───────────────────────────────────────────────────

def make_yt_player(uid: str, title: str, filepath: str) -> str:
    audio_url = make_audio_url(filepath)
    safe_title = title.replace("'", "&#39;").replace('"', "&quot;")

    return f"""
<div id="ytp-{uid}" style="
    background: linear-gradient(145deg, #0c0c1a 0%, #110d20 60%, #0c1220 100%);
    border: 1px solid rgba(124,58,237,0.22);
    border-radius: 16px;
    overflow: hidden;
    color: #fff;
    font-family: 'Inter','Noto Sans KR',system-ui,sans-serif;
    box-shadow: 0 8px 32px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.03), inset 0 1px 0 rgba(255,255,255,0.05);
    max-width: 100%;
    margin-bottom: 12px;
    position: relative;
">
  <!-- Decorative glow orbs -->
  <div style="position:absolute;top:-40px;right:-40px;width:160px;height:160px;
      background:radial-gradient(circle,rgba(124,58,237,0.12) 0%,transparent 70%);
      pointer-events:none;border-radius:50%;"></div>
  <div style="position:absolute;bottom:-30px;left:20%;width:100px;height:100px;
      background:radial-gradient(circle,rgba(6,182,212,0.08) 0%,transparent 70%);
      pointer-events:none;border-radius:50%;"></div>

  <!-- Title bar -->
  <div style="
      padding: 14px 18px 12px;
      display: flex;
      align-items: center;
      gap: 10px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      position: relative;
  ">
    <span style="
        background: linear-gradient(135deg,rgba(124,58,237,0.3),rgba(6,182,212,0.2));
        border: 1px solid rgba(124,58,237,0.35);
        color: #c4b5fd;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 20px;
        letter-spacing: 0.06em;
        flex-shrink: 0;
        text-transform: uppercase;
    ">KR</span>
    <span style="
        font-size: 13px;
        font-weight: 600;
        color: #e2e8f0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        letter-spacing: .01em;
    ">{safe_title}</span>
  </div>

  <!-- Progress bar -->
  <div style="padding: 14px 18px 6px;">
    <div id="ybar-{uid}"
         onclick="ytSeek('{uid}',event)"
         style="
             height: 3px;
             background: rgba(255,255,255,0.1);
             cursor: pointer;
             position: relative;
             border-radius: 2px;
             transition: height .15s ease;
         "
         onmouseenter="
             this.style.height='6px';
             var h=document.getElementById('yhandle-{uid}');
             if(h) h.style.opacity='1';
         "
         onmouseleave="
             this.style.height='3px';
             var h=document.getElementById('yhandle-{uid}');
             if(h) h.style.opacity='0';
         ">
      <div id="yfill-{uid}" style="
          height: 100%;
          width: 0%;
          background: linear-gradient(90deg, #7c3aed 0%, #06b6d4 100%);
          position: relative;
          border-radius: 2px;
          pointer-events: none;
          transition: width .1s linear;
      ">
        <div id="yhandle-{uid}" style="
            position: absolute;
            right: -7px;
            top: 50%;
            transform: translateY(-50%);
            width: 14px;
            height: 14px;
            background: #a78bfa;
            border-radius: 50%;
            opacity: 0;
            transition: opacity .15s;
            box-shadow: 0 0 10px rgba(167,139,250,0.9), 0 0 20px rgba(124,58,237,0.4);
        "></div>
      </div>
    </div>
  </div>

  <!-- Controls -->
  <div style="display:flex;align-items:center;padding:6px 12px 16px;gap:2px;">

    <!-- Play/Pause -->
    <button onclick="ytPlay('{uid}')" title="재생 (k)"
            style="background:none;border:none;cursor:pointer;padding:8px;color:#fff;display:flex;align-items:center;border-radius:50%;transition:background .15s;"
            onmouseenter="this.style.background='rgba(124,58,237,0.25)'"
            onmouseleave="this.style.background='none'">
      <svg id="yicon-{uid}" width="34" height="34" viewBox="0 0 24 24" fill="#c4b5fd">
        <path d="M8 5v14l11-7z"/>
      </svg>
    </button>

    <!-- Replay 10s -->
    <button onclick="ytSkip('{uid}',-10)" title="10초 뒤로"
            style="background:none;border:none;cursor:pointer;padding:6px;color:#fff;display:flex;align-items:center;border-radius:50%;transition:background .15s;"
            onmouseenter="this.style.background='rgba(255,255,255,0.07)'"
            onmouseleave="this.style.background='none'">
      <svg width="24" height="24" viewBox="0 0 24 24">
        <path d="M11.99 5V1l-5 5 5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6h-2c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z" fill="#64748b"/>
        <text x="12" y="17.5" font-size="5.5" text-anchor="middle" fill="#94a3b8" font-weight="bold" font-family="Inter,Arial,sans-serif">10</text>
      </svg>
    </button>

    <!-- Forward 10s -->
    <button onclick="ytSkip('{uid}',10)" title="10초 앞으로"
            style="background:none;border:none;cursor:pointer;padding:6px;color:#fff;display:flex;align-items:center;border-radius:50%;transition:background .15s;"
            onmouseenter="this.style.background='rgba(255,255,255,0.07)'"
            onmouseleave="this.style.background='none'">
      <svg width="24" height="24" viewBox="0 0 24 24">
        <path d="M12.01 5V1l5 5-5 5V7c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6h2c0 4.42-3.58 8-8 8s-8-3.58-8-8 3.58-8 8-8z" fill="#64748b"/>
        <text x="12" y="17.5" font-size="5.5" text-anchor="middle" fill="#94a3b8" font-weight="bold" font-family="Inter,Arial,sans-serif">10</text>
      </svg>
    </button>

    <!-- Mute -->
    <button onclick="ytMute('{uid}')"
            style="background:none;border:none;cursor:pointer;padding:6px;color:#fff;display:flex;align-items:center;border-radius:50%;transition:background .15s;"
            onmouseenter="this.style.background='rgba(255,255,255,0.07)'"
            onmouseleave="this.style.background='none'">
      <svg id="yvolicon-{uid}" width="20" height="20" viewBox="0 0 24 24" fill="#64748b">
        <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>
      </svg>
    </button>

    <!-- Volume slider -->
    <input type="range" id="yvs-{uid}" min="0" max="100" value="100"
           oninput="ytVol('{uid}',this.value)"
           style="width:60px;accent-color:#7c3aed;cursor:pointer;vertical-align:middle;opacity:0.75;">

    <!-- Time -->
    <span id="ytime-{uid}" style="font-size:11px;color:#475569;white-space:nowrap;margin-left:10px;font-variant-numeric:tabular-nums;font-family:'Inter',monospace;">
      0:00 / 0:00
    </span>

    <div style="flex:1;"></div>

    <!-- Speed -->
    <select onchange="ytSpeed('{uid}',this.value)"
            style="background:rgba(255,255,255,0.05);color:#64748b;border:1px solid rgba(255,255,255,0.08);padding:4px 8px;border-radius:8px;cursor:pointer;font-size:11px;outline:none;font-family:inherit;letter-spacing:0.02em;">
      <option value="0.5">0.5×</option>
      <option value="0.75">0.75×</option>
      <option value="1" selected>1×</option>
      <option value="1.25">1.25×</option>
      <option value="1.5">1.5×</option>
      <option value="2">2×</option>
    </select>
  </div>

  <audio id="ya-{uid}" preload="metadata" src="{audio_url}"></audio>
</div>
"""


# ── Global JS ──────────────────────────────────────────────────────────────────

GLOBAL_JS = """
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

var _ytObs = new MutationObserver(function(muts) {
  for (var i=0; i<muts.length; i++) {
    for (var j=0; j<muts[i].addedNodes.length; j++) {
      var node = muts[i].addedNodes[j];
      if (!node || node.nodeType !== 1) continue;
      if (node.tagName==='AUDIO' && node.id && node.id.indexOf('ya-')===0) {
        window._setupYTPlayer(node.id.slice(3), node);
      }
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


# ── Custom CSS ─────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ═══════════════════════════════════════
   GLOBAL — Dark Background
═══════════════════════════════════════ */
:root {
    --c-bg:      #080812;
    --c-surface: #0e0e1c;
    --c-card:    rgba(15,14,30,0.9);
    --c-border:  rgba(255,255,255,0.07);
    --c-primary: #7c3aed;
    --c-accent:  #06b6d4;
    --c-text:    #e2e8f0;
    --c-muted:   #475569;
    --c-label:   #94a3b8;
    --radius:    12px;

    /* Gradio 6.x theme variable overrides */
    --body-background-fill: #080812 !important;
    --body-text-color: #e2e8f0 !important;
    --background-fill-primary: #0e0e1c !important;
    --background-fill-secondary: #111120 !important;
    --input-background-fill: rgba(255,255,255,0.05) !important;
    --input-border-color: rgba(255,255,255,0.1) !important;
    --input-border-color-focus: rgba(124,58,237,0.55) !important;
    --input-placeholder-color: #4b5568 !important;
    --border-color-primary: rgba(255,255,255,0.08) !important;
    --border-color-accent: rgba(124,58,237,0.4) !important;
    --block-background-fill: rgba(255,255,255,0.02) !important;
    --block-border-color: rgba(255,255,255,0.07) !important;
    --block-label-text-color: #64748b !important;
    --block-title-text-color: #94a3b8 !important;
    --panel-background-fill: #0c0c1a !important;
    --panel-border-color: rgba(255,255,255,0.07) !important;
    --color-accent: #7c3aed !important;
    --button-primary-background-fill: linear-gradient(135deg, #7c3aed, #0891b2) !important;
    --button-primary-text-color: #fff !important;
    --button-secondary-background-fill: rgba(255,255,255,0.05) !important;
    --button-secondary-text-color: #94a3b8 !important;
    --table-even-background-fill: rgba(255,255,255,0.01) !important;
    --table-odd-background-fill: rgba(124,58,237,0.03) !important;
    --table-row-focus: rgba(124,58,237,0.06) !important;
    --stat-background-fill: #0e0e1c !important;
    --checkbox-background-color: rgba(255,255,255,0.05) !important;
    --checkbox-border-color: rgba(255,255,255,0.15) !important;
    --checkbox-background-color-selected: #7c3aed !important;
    --accordion-text-color: #64748b !important;
    --section-header-text-color: #64748b !important;
}

body, .dark, html {
    background: var(--c-bg) !important;
}

.gradio-container {
    background: var(--c-bg) !important;
    max-width: 1500px !important;
    padding: 0 !important;
    font-family: 'Inter', 'Noto Sans KR', system-ui, -apple-system, sans-serif !important;
}

/* Remove default Gradio padding */
.gradio-container > .main {
    padding: 0 !important;
}

/* ═══════════════════════════════════════
   TYPOGRAPHY
═══════════════════════════════════════ */
* {
    font-family: 'Inter', 'Noto Sans KR', system-ui, sans-serif !important;
}

/* Label text */
label span, .block > label > span, .svelte-1gfkn6j {
    color: var(--c-label) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ═══════════════════════════════════════
   FORM INPUTS
═══════════════════════════════════════ */
textarea, input[type="text"], input[type="password"], input[type="number"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

textarea:focus, input[type="text"]:focus, input[type="password"]:focus {
    border-color: rgba(124,58,237,0.55) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
    outline: none !important;
}

/* Placeholder */
textarea::placeholder, input::placeholder {
    color: #4b5568 !important;
}

/* ═══════════════════════════════════════
   DROPDOWNS / SELECT
═══════════════════════════════════════ */
select, .gr-dropdown select {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
    font-size: 13px !important;
}

/* ═══════════════════════════════════════
   BUTTONS
═══════════════════════════════════════ */
button.primary, .gr-button.primary {
    background: linear-gradient(135deg, #7c3aed 0%, #0891b2 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35), 0 1px 0 rgba(255,255,255,0.1) inset !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}

button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(124,58,237,0.5), 0 1px 0 rgba(255,255,255,0.1) inset !important;
}

button.secondary, .gr-button.secondary {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-size: 13px !important;
    transition: background 0.15s, color 0.15s !important;
}

button.secondary:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #e2e8f0 !important;
}

button.stop, .gr-button.stop {
    background: rgba(239,68,68,0.08) !important;
    border: 1px solid rgba(239,68,68,0.2) !important;
    border-radius: 10px !important;
    color: #f87171 !important;
    font-size: 13px !important;
}

button.stop:hover {
    background: rgba(239,68,68,0.15) !important;
}

/* ═══════════════════════════════════════
   CHECKBOX
═══════════════════════════════════════ */
input[type="checkbox"] {
    accent-color: #7c3aed !important;
    width: 16px !important;
    height: 16px !important;
}

/* ═══════════════════════════════════════
   TABS
═══════════════════════════════════════ */
.tab-nav {
    background: rgba(255,255,255,0.02) !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    padding: 0 16px !important;
    gap: 4px !important;
}

.tab-nav button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: #475569 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 14px 16px 12px !important;
    letter-spacing: 0.01em !important;
    transition: color 0.15s !important;
    margin-bottom: -1px !important;
}

.tab-nav button:hover {
    color: #94a3b8 !important;
    background: rgba(255,255,255,0.02) !important;
}

.tab-nav button.selected {
    color: #a78bfa !important;
    border-bottom: 2px solid #7c3aed !important;
    font-weight: 600 !important;
    background: transparent !important;
}

/* ═══════════════════════════════════════
   BLOCKS & PANELS
═══════════════════════════════════════ */
.block, .gr-block {
    background: transparent !important;
    border: none !important;
}

.panel {
    background: transparent !important;
    border: none !important;
}

/* ═══════════════════════════════════════
   ACCORDION
═══════════════════════════════════════ */
.accordion {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

.accordion > .label-wrap {
    padding: 12px 16px !important;
    cursor: pointer !important;
}

.accordion > .label-wrap:hover {
    background: rgba(255,255,255,0.02) !important;
}

.accordion > .label-wrap > span {
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: none !important;
    letter-spacing: 0.01em !important;
}

/* ═══════════════════════════════════════
   MARKDOWN
═══════════════════════════════════════ */
.prose, .gr-markdown, .md {
    color: #cbd5e1 !important;
}

.prose p, .gr-markdown p {
    color: #94a3b8 !important;
    font-size: 13px !important;
}

/* italic / emphasis text (status messages) */
.prose em, .gr-markdown em, .md em {
    color: #64748b !important;
    font-style: italic !important;
}

.prose table, .gr-markdown table {
    border-collapse: collapse !important;
    width: 100% !important;
    font-size: 13px !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

.prose th, .gr-markdown th {
    background: rgba(124,58,237,0.08) !important;
    color: #a78bfa !important;
    font-weight: 700 !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid rgba(124,58,237,0.15) !important;
    white-space: nowrap !important;
}

.prose td, .gr-markdown td {
    color: #cbd5e1 !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid rgba(255,255,255,0.04) !important;
    font-size: 13px !important;
}

.prose tr:hover td, .gr-markdown tr:hover td {
    background: rgba(255,255,255,0.015) !important;
}

/* ═══════════════════════════════════════
   GROUP
═══════════════════════════════════════ */
.gr-group {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    padding: 20px !important;
    transition: border-color 0.2s !important;
}

.gr-group:hover {
    border-color: rgba(124,58,237,0.2) !important;
}

/* ═══════════════════════════════════════
   FILE COMPONENT
═══════════════════════════════════════ */
.file-preview, .gr-file {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
}

/* ═══════════════════════════════════════
   SCROLLBAR
═══════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,0.4); }

/* ═══════════════════════════════════════
   HR
═══════════════════════════════════════ */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.07) !important;
    margin: 20px 0 !important;
}

/* ═══════════════════════════════════════
   SECTION DIVIDER LABELS
═══════════════════════════════════════ */
.section-label {
    font-size: 10px !important;
    font-weight: 700 !important;
    color: #64748b !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 16px 4px 8px !important;
    display: block !important;
}

/* Range input */
input[type="range"] {
    accent-color: #7c3aed !important;
}

/* Gradio row gap fix */
.gap-4 { gap: 12px !important; }

/* Row padding fix */
.gr-row { gap: 12px !important; }
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

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


# ── Processing ─────────────────────────────────────────────────────────────────

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


# ── Voice Library ──────────────────────────────────────────────────────────────

def _parse_time(s: str) -> float:
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
    voice_name = (voice_name or "").strip()
    if not voice_name:
        yield "⚠️ 목소리 이름을 입력하세요.", *list_voice_library()
        return
    if not url or not url.strip():
        yield "⚠️ YouTube URL을 입력하세요.", *list_voice_library()
        return

    voice_dir = os.path.join(VOICES_DIR, voice_name)
    if os.path.exists(voice_dir):
        yield f"⚠️ '{voice_name}' 이름이 이미 존재합니다.", *list_voice_library()
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


# ── History ────────────────────────────────────────────────────────────────────

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


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="팟캐스트 한국어 더빙",
) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="
        background: linear-gradient(135deg, #0c0818 0%, #0d1220 50%, #080c18 100%);
        border-bottom: 1px solid rgba(124,58,237,0.2);
        padding: 28px 36px 24px;
        position: relative;
        overflow: hidden;
    ">
      <!-- Background glow orbs -->
      <div style="position:absolute;top:-60px;right:-60px;width:220px;height:220px;
          background:radial-gradient(circle,rgba(124,58,237,0.12) 0%,transparent 65%);
          border-radius:50%;pointer-events:none;"></div>
      <div style="position:absolute;bottom:-40px;left:25%;width:160px;height:160px;
          background:radial-gradient(circle,rgba(6,182,212,0.08) 0%,transparent 65%);
          border-radius:50%;pointer-events:none;"></div>

      <!-- Logo + Title -->
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px;position:relative;">
        <div style="
            width: 48px; height: 48px;
            border-radius: 14px;
            background: linear-gradient(135deg, #7c3aed, #0891b2);
            display: flex; align-items: center; justify-content: center;
            font-size: 24px;
            flex-shrink: 0;
            box-shadow: 0 6px 24px rgba(124,58,237,0.45), 0 1px 0 rgba(255,255,255,0.15) inset;
        ">🎙️</div>
        <div>
          <h1 style="
              margin: 0;
              font-size: 28px;
              font-weight: 800;
              background: linear-gradient(135deg, #c4b5fd 0%, #67e8f9 100%);
              -webkit-background-clip: text;
              -webkit-text-fill-color: transparent;
              background-clip: text;
              line-height: 1.2;
              letter-spacing: -0.02em;
          ">팟캐스트 한국어 더빙</h1>
          <p style="margin: 5px 0 0; color: #64748b; font-size: 13px; font-weight: 400; letter-spacing: 0.01em;">
            YouTube URL → AI 자막 추출 → 한국어 번역 → TTS → MP3
          </p>
        </div>
      </div>

      <!-- Feature badges -->
      <div style="display:flex;gap:8px;flex-wrap:wrap;position:relative;">
        <span style="background:rgba(124,58,237,0.12);border:1px solid rgba(124,58,237,0.25);color:#a78bfa;font-size:10px;font-weight:700;padding:4px 10px;border-radius:20px;letter-spacing:0.04em;">🤖 Claude / Google 번역</span>
        <span style="background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.2);color:#67e8f9;font-size:10px;font-weight:700;padding:4px 10px;border-radius:20px;letter-spacing:0.04em;">🔊 Edge TTS / Fish Speech</span>
        <span style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);color:#6ee7b7;font-size:10px;font-weight:700;padding:4px 10px;border-radius:20px;letter-spacing:0.04em;">⚡ GPU Whisper STT</span>
        <span style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);color:#fcd34d;font-size:10px;font-weight:700;padding:4px 10px;border-radius:20px;letter-spacing:0.04em;">🎭 화자 구분 더빙</span>
      </div>
    </div>
    """)

    # ── Main Content ──────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # Left column: Input + Settings
        with gr.Column(scale=2, min_width=380):
            gr.HTML('<span class="section-label">📥 URL 입력</span>')

            url_input = gr.Textbox(
                label="YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...\nhttps://youtu.be/...",
                lines=3,
                show_label=False,
            )

            with gr.Row():
                whisper_model = gr.Dropdown(
                    label="Whisper 모델",
                    choices=["large-v3", "medium", "small", "base"],
                    value="large-v3",
                )
                whisper_device = gr.Dropdown(
                    label="처리 장치",
                    choices=["cuda", "cpu"],
                    value="cuda",
                )
            with gr.Row():
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

            with gr.Accordion("🔧 고급 설정", open=False):
                voice_clone_cb = gr.Checkbox(
                    label="🎤 원본 화자 목소리로 더빙 (Fish Speech — checkpoints 필요)",
                    value=False,
                )
                ref_voice_selector = gr.Dropdown(
                    label="📚 라이브러리 목소리 (비어있으면 영상에서 자동 추출)",
                    choices=_list_ref_voices(),
                    value=None,
                )
                claude_key_input = gr.Textbox(
                    label="🤖 Claude API 키 (없으면 Google 번역 사용)",
                    placeholder="sk-ant-...",
                    type="password",
                    value=DEFAULT_CLAUDE_KEY,
                )
                diarization_cb = gr.Checkbox(
                    label="🎙️ 화자 구분 더빙 (AssemblyAI — 인터뷰/게스트 목소리 분리)",
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

            start_btn = gr.Button("🚀  처리 시작", variant="primary", size="lg")

        # Right column: Tabs (처리 결과 / 목소리 라이브러리 / 처리 히스토리)
        with gr.Column(scale=3):
            with gr.Tabs():
                # ── 처리 결과 ──────────────────────────────────────────────
                with gr.Tab("🎵 처리 결과"):
                    slot_players  = []
                    slot_ko_texts = []
                    slot_en_texts = []
                    slot_groups   = []

                    for i in range(MAX_SLOTS):
                        with gr.Group(visible=False) as grp:
                            player = gr.HTML("")
                            with gr.Accordion("📄 한국어 번역 보기", open=False):
                                ko_text = gr.Textbox(label="한국어", lines=8, max_lines=20, interactive=False)
                            with gr.Accordion("📄 영어 원문 보기", open=False):
                                en_text = gr.Textbox(label="English", lines=8, max_lines=20, interactive=False)

                        slot_players.append(player)
                        slot_ko_texts.append(ko_text)
                        slot_en_texts.append(en_text)
                        slot_groups.append(grp)

                # ── 목소리 라이브러리 ──────────────────────────────────────
                with gr.Tab("🎤 목소리 라이브러리"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.HTML('<span class="section-label">새 목소리 추가</span>')
                            lib_url = gr.Textbox(
                                label="YouTube URL",
                                placeholder="https://www.youtube.com/watch?v=...",
                                show_label=False,
                            )
                            with gr.Row():
                                lib_start = gr.Textbox(label="시작 시간 (MM:SS)", placeholder="0:30", scale=1)
                                lib_end   = gr.Textbox(label="끝 시간 (MM:SS)", placeholder="1:00", scale=1)
                                lib_name  = gr.Textbox(label="목소리 이름", placeholder="나레이터", scale=2)
                            lib_device = gr.Dropdown(label="STT 장치", choices=["cuda", "cpu"], value="cuda")
                            with gr.Row():
                                lib_add_btn    = gr.Button("🎙️ 목소리 추출 & 등록", variant="primary")
                                lib_delete_btn = gr.Button("🗑️ 선택 삭제", variant="stop")
                            lib_status = gr.Markdown("_URL과 구간을 입력하고 등록하세요._")

                        with gr.Column(scale=3):
                            gr.HTML('<span class="section-label">등록된 목소리</span>')
                            lib_table      = gr.Markdown()
                            lib_selector   = gr.Dropdown(label="삭제할 목소리 선택", choices=[])
                            lib_refresh_btn = gr.Button("🔄 새로고침", variant="secondary")

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

                # ── 처리 히스토리 ──────────────────────────────────────────
                with gr.Tab("📁 처리 히스토리"):
                    history_md     = gr.Markdown()
                    history_player = gr.HTML()
                    with gr.Row():
                        file_selector   = gr.Dropdown(label="파일 선택", choices=[], scale=4)
                        open_folder_btn = gr.Button("📂 폴더 열기", variant="secondary", scale=1)
                        delete_btn      = gr.Button("🗑️ 선택 삭제", variant="stop", scale=1)
                        delete_all_btn  = gr.Button("🗑️ 전체 삭제", variant="stop", scale=1)
                        refresh_btn     = gr.Button("🔄 새로고침", variant="secondary", scale=1)
                    download_file = gr.File(label="⬇️ 다운로드", interactive=False, visible=False)

                    history_outputs = [history_md, history_player, file_selector, download_file]
                    refresh_btn.click(fn=list_history, outputs=history_outputs)
                    delete_btn.click(fn=delete_file, inputs=[file_selector], outputs=history_outputs)
                    delete_all_btn.click(fn=delete_all_files, outputs=history_outputs)
                    file_selector.change(fn=_get_download_path, inputs=[file_selector], outputs=[download_file])
                    open_folder_btn.click(fn=open_output_folder)
                    demo.load(fn=list_history, outputs=history_outputs)

    # ── 처리 상태 (하단) ──────────────────────────────────────────────────────
    gr.HTML('<div style="height:1px;background:rgba(255,255,255,0.06);margin:4px 0 0;"></div>')
    gr.HTML('<span class="section-label" style="padding:12px 4px 6px;">📊 처리 상태</span>')
    status_md = gr.Markdown("_URL을 입력하고 처리 시작을 클릭하세요._")

    # Wire all outputs
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
        css=CUSTOM_CSS,
    )
