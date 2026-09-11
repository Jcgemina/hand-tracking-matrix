# Hand Tracking Matrix

This Python app opens your webcam, detects up to two hands, and draws lines between the wrist, palm, and every finger joint in real time.

## Setup

Use Python 3.10 or 3.11 (MediaPipe compatibility is most reliable on those versions), then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python hand_tracking.py
```

Allow camera access if Windows asks. Press `Q` or `Esc` to quit.

If your webcam is not the default camera, change `cv2.VideoCapture(0)` in `hand_tracking.py` to `cv2.VideoCapture(1)`.

## MediaPipe version note

This project intentionally uses `mediapipe==0.10.21`: newer MediaPipe releases removed the older `mp.solutions` interface used by this simple example. If you previously installed a newer version, reinstall the project dependencies with:

```powershell
pip uninstall mediapipe -y
pip install -r requirements.txt
```
