import ctypes
import subprocess
import time
from agent.kinematics import Point2D, build_cursor_trajectory
from agent.typing_model import TypingModel

user32 = ctypes.windll.user32

class _Pt(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

pt = _Pt()
user32.GetCursorPos(ctypes.byref(pt))
start = Point2D(float(pt.x), float(pt.y))
target = Point2D(float(min(pt.x + 180, 1100)), float(min(pt.y + 140, 650)))

samples = build_cursor_trajectory(start=start, target=target, dt=0.015)
prev_t = 0.0
for s in samples:
    user32.SetCursorPos(int(round(s.x)), int(round(s.y)))
    delta = max(0.0, float(s.t) - prev_t)
    prev_t = float(s.t)
    if delta > 0:
        time.sleep(delta)

subprocess.Popen(["notepad.exe"])
time.sleep(1.3)

tm = TypingModel(wpm=75.0)
events = tm.plan_text("Bot live desktop typing active.", allow_typos=False)

send_parts = []
for ev in events:
    if ev.kind == "char" and ev.value and ev.value not in {"'", '"', "{", "}", "+", "^", "%", "~", "(", ")"}:
        ms = int(max(15, float(ev.delay_sec) * 1000))
        send_parts.append(f"[System.Windows.Forms.SendKeys]::SendWait('{ev.value}'); Start-Sleep -Milliseconds {ms}")
    elif ev.kind == "key" and ev.value.lower() == "space":
        ms = int(max(15, float(ev.delay_sec) * 1000))
        send_parts.append(f"[System.Windows.Forms.SendKeys]::SendWait(' '); Start-Sleep -Milliseconds {ms}")

if send_parts:
    ps_script = "Add-Type -AssemblyName System.Windows.Forms; " + "; ".join(send_parts)
    subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=False)

print(f"DONE: {len(samples)} cursor moves, {len(send_parts)} key events sent.")
