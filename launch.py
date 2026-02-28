"""
팟캐스트 한국어 더빙 — 원클릭 런처
pythonw.exe 로 실행 → 터미널/검은창 없음
"""
import os
import socket
import subprocess
import sys
import time
import webbrowser

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_URL  = "http://localhost:7860"
PORT        = 7860

# python.exe 경로 (pythonw.exe 와 같은 폴더에 있음)
PYTHON_EXE = os.path.join(os.path.dirname(sys.executable), "python.exe")


def is_ready() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=1):
            return True
    except OSError:
        return False


def main():
    # 이미 서버 켜져 있으면 바로 열기
    if is_ready():
        webbrowser.open(SERVER_URL)
        return

    # 서버 시작 — python.exe + CREATE_NO_WINDOW (창 없음)
    subprocess.Popen(
        [PYTHON_EXE, os.path.join(PROJECT_DIR, "app.py")],
        cwd=PROJECT_DIR,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    # 준비될 때까지 1초마다 확인 (최대 45초)
    for _ in range(45):
        time.sleep(1)
        if is_ready():
            time.sleep(1)
            webbrowser.open(SERVER_URL)
            return

    # 타임아웃 → 그냥 열기
    webbrowser.open(SERVER_URL)


if __name__ == "__main__":
    main()
