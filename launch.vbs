' 팟캐스트 한국어 더빙 — 원클릭 런처

Set oShell = CreateObject("WScript.Shell")
Set oFSO   = CreateObject("Scripting.FileSystemObject")

projectDir = "C:\Users\sungh\OneDrive\Desktop\팟케스트 번역"
serverUrl  = "http://localhost:7860"
tmpFile    = oShell.ExpandEnvironmentStrings("%TEMP%") & "\pc_ns.txt"

' ── 포트 확인: 파이프 없이 파일 리디렉트만 사용 ─────────────────
' 파이프(|)가 있으면 cmd 창이 추가로 뜨므로 파이프 금지
Function IsServerReady()
    On Error Resume Next
    If oFSO.FileExists(tmpFile) Then oFSO.DeleteFile tmpFile, True
    ' 파이프 없이 netstat 전체를 파일로 저장
    oShell.Run "cmd /c netstat -an > """ & tmpFile & """", 0, True
    If oFSO.FileExists(tmpFile) Then
        Dim f : Set f = oFSO.OpenTextFile(tmpFile, 1)
        Dim s : s = f.ReadAll()
        f.Close
        ' VBScript에서 직접 :7860 + LISTENING 검색
        IsServerReady = (InStr(s, ":7860") > 0)
    Else
        IsServerReady = False
    End If
End Function

' ── 이미 실행 중이면 바로 브라우저 ─────────────────────────────
If IsServerReady() Then
    oShell.Run serverUrl, 1, False
    WScript.Quit
End If

' ── 서버 백그라운드 시작 ────────────────────────────────────────
oShell.Run "cmd /c cd /d """ & projectDir & """ && start /B python app.py > nul 2>&1", 0, False

' ── 준비될 때까지 2초마다 확인 (최대 40초) ──────────────────────
Dim i
For i = 1 To 20
    WScript.Sleep 2000
    If IsServerReady() Then
        WScript.Sleep 1500
        oShell.Run serverUrl, 1, False
        WScript.Quit
    End If
Next

oShell.Run serverUrl, 1, False
