# Libraries

## Selection Criteria
- Actively maintained (releases within last 12 months)
- Widely used (GitHub stars, PyPI downloads)
- Permissive license (MIT, Apache-2.0, BSD)
- Cross-platform where possible
- Minimal dependencies
- Good documentation

---

## Voice/Audio Capture

### Recommended: `sounddevice` + `soundfile`

| Aspect | Detail |
|--------|--------|
| **Library** | `sounddevice` (PortAudio bindings), `soundfile` (libsndfile bindings) |
| **Purpose** | Microphone capture, audio I/O, WAV writing |
| **Version** | sounddevice ≥ 0.4.6, soundfile ≥ 0.12.1 |
| **License** | MIT (sounddevice), BSD-3-Clause (soundfile) |
| **Platform** | Windows, macOS, Linux |
| **Reason** | - Direct PortAudio binding, low latency<br>- Callback-based streaming for real-time capture<br>- `soundfile` handles WAV/FLAC/OGG reliably<br>- Both are pure Python wrappers, no heavy deps<br>- Used in production audio applications |

**Alternatives Considered:**
- `pyaudio`: Older, less maintained, harder to install on Windows
- `pydub`: Requires ffmpeg, higher-level, not for real-time capture
- `python-sounddevice` (same as sounddevice): This is the correct package

**Usage Pattern:**
```python
import sounddevice as sd
import soundfile as sf

# Callback streaming
def callback(indata, frames, time, status):
    # indata is numpy array, write to queue/file
    pass

stream = sd.InputStream(callback=callback, samplerate=16000, channels=1)
stream.start()
```

---

## Keyboard/Keystroke Capture

### Recommended: `pynput`

| Aspect | Detail |
|--------|--------|
| **Library** | `pynput` |
| **Purpose** | Global keyboard (and mouse) listeners |
| **Version** | ≥ 1.7.6 |
| **License** | LGPL-3.0 |
| **Platform** | Windows, macOS, Linux (X11) |
| **Reason** | - Cross-platform global keyboard hook<br>- Supports key_down/key_up events with timestamps<br>- Pure Python with platform-specific backends<br>- Handles modifiers, special keys<br>- Actively maintained |

**Alternatives Considered:**
- `keyboard`: Windows-only, requires admin, less maintained
- `pyhook`: Windows-only, deprecated
- `keyboard` (boppreh/keyboard): Good but Windows-focused, admin needed for global hook
- `pynput` LGPL: Requires dynamic linking consideration for distribution; acceptable for this use case

**Platform-Specific Notes:**
- **Windows**: Uses `SetWindowsHookEx` — works without admin for low-level hook
- **macOS**: Requires Accessibility permission (TCC)
- **Linux (X11)**: Uses XRecord extension; Wayland not supported for global capture

**Usage Pattern:**
```python
from pynput import keyboard

def on_press(key):
    # key.char for printable, key.name for special
    pass

def on_release(key):
    pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
```

**Privacy Filtering:**
```python
SENSITIVE_FIELDS = {"password", "secret", "token", "credit", "cvv", "ssn"}
# Filter by checking active window title/class or field type where possible
```

---

## Webcam and Facial Feature/Expression Processing

### Recommended: `opencv-python` + `mediapipe`

| Aspect | Detail |
|--------|--------|
| **Library** | `opencv-python` (OpenCV), `mediapipe` |
| **Purpose** | Webcam capture, face detection, landmarks, expression features |
| **Version** | opencv-python ≥ 4.8.0, mediapipe ≥ 0.10.0 |
| **License** | Apache-2.0 (both) |
| **Platform** | Windows, macOS, Linux |
| **Reason** | - OpenCV: Standard for camera capture, frame processing<br>- MediaPipe: Google's production-ready face mesh (468 landmarks), face detection, iris tracking<br>- MediaPipe Face Mesh runs real-time on CPU (no GPU required)<br>- Provides blendshapes (52 coefficients) for expression analysis<br>- Well-documented, used in production apps<br>- Apache-2.0 license is permissive |

**MediaPipe Face Capabilities:**
- Face Detection: BlazeFace (fast, accurate)
- Face Mesh: 468 3D landmarks
- Face Landmarks: 52 blendshape coefficients (expression features)
- Iris Tracking: Eye gaze direction
- All run at 30+ FPS on modern CPUs

**Alternatives Considered:**
- `dlib`: Excellent landmarks (68-point) but slower, C++ build issues on Windows
- `face_recognition`: Wrapper around dlib, same issues
- `deepface`: Heavy (TensorFlow), multiple backends, overkill for real-time
- `insightface`: Good but larger, more complex setup
- `openface`: Older, Torch-based, not maintained
- Custom CNN: Violates "existing library first" principle

**Usage Pattern:**
```python
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # includes iris
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        # Extract blendshapes for expression
        blendshapes = results.face_blendshapes[0] if results.face_blendshapes else None
```

**Fallback for Expression Classification (if needed):**
- `fer` (Facial Expression Recognition): Lightweight, 7 emotions, FER2013 trained
- `deepface`: If heavier analysis needed, but adds TF dependency
- **Recommendation**: Start with MediaPipe blendshapes (52 coefficients) as raw features; add classifier only if requirement demands categorical emotions

---

## Desktop UI Framework

### Recommended: `PySide6` (Qt for Python)

| Aspect | Detail |
|--------|--------|
| **Library** | `PySide6` |
| **Purpose** | Desktop UI, native look, cross-platform |
| **Version** | ≥ 6.6.0 |
| **License** | LGPL-3.0 (commercial option available) |
| **Platform** | Windows, macOS, Linux |
| **Reason** | - Official Qt for Python binding (Qt Company)<br>- Mature, complete Qt6 API coverage<br>- Native widgets, high DPI support<br>- Qt's signal/slot fits event-driven architecture<br>- Better licensing than PyQt (LGPL vs GPL)<br>- Excellent threading support (QThread, QRunnable)<br>- QWebEngineView if web-based UI needed later |

**Alternatives Considered:**
- `tkinter` (stdlib): No external dep, but limited widgets, dated look, weaker threading
- `customtkinter`: Modern look on tkinter, but still tkinter underneath
- `PyQt6`: Same Qt binding, but GPL license (viral) or paid commercial
- `flet` / `nicegui`: Web-based, adds browser overhead
- `wxPython`: Native but smaller community, less active
- `kivy`: Custom renderer, non-native look, mobile-focused

**LGPL Consideration:**
- Dynamically link to Qt libraries (default with PySide6)
- Provide attribution
- Allow user to swap Qt libraries
- Acceptable for this application type

**Usage Pattern:**
```python
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtCore import QThread, Signal, QTimer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # UI setup
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(100)  # 10 FPS UI updates
```

---

## Data Processing / Numerical / Signal Processing

### Recommended: `numpy` + `scipy`

| Aspect | Detail |
|--------|--------|
| **Library** | `numpy`, `scipy` |
| **Purpose** | Numerical arrays, signal processing, audio feature extraction |
| **Version** | numpy ≥ 1.24, scipy ≥ 1.11 |
| **License** | BSD-3-Clause |
| **Platform** | All |
| **Reason** | - Foundation for all numerical work<br>- scipy.signal for audio processing (resample, filter, spectrogram)<br>- Required by MediaPipe, OpenCV internally<br>- Standard, battle-tested |

---

## Configuration

### Recommended: `pyyaml`

| Aspect | Detail |
|--------|--------|
| **Library** | `pyyaml` |
| **Purpose** | YAML configuration loading |
| **Version** | ≥ 6.0 |
| **License** | MIT |
| **Reason** | - Standard for YAML in Python<br>- Simple API, no surprises<br>- Supports YAML 1.2 |

**Alternative:** `tomli` / `tomllib` (stdlib in 3.11+) for TOML — but SPEC specifies YAML

---

## Logging

### Recommended: `structlog` + stdlib `logging`

| Aspect | Detail |
|--------|--------|
| **Library** | `structlog` |
| **Purpose** | Structured logging with context |
| **Version** | ≥ 23.1 |
| **License** | MIT/Apache-2.0 |
| **Reason** | - Structured key-value logging<br>- Integrates with stdlib logging<br>- Context binding (session_id, module)<br>- JSON output for log aggregation<br>- No heavy dependencies |

**Alternative:** stdlib `logging` with custom formatter — more boilerplate

---

## Storage / Serialization

### Recommended: `orjson` + stdlib `json`

| Aspect | Detail |
|--------|--------|
| **Library** | `orjson` |
| **Purpose** | Fast JSON serialization |
| **Version** | ≥ 3.9 |
| **License** | MIT/Apache-2.0 |
| **Reason** | - 2-10x faster than stdlib json<br>- Handles numpy, datetime, UUID natively<br>- Strict JSON output<br>- Rust backend, no runtime deps |

**Usage:** JSONL lines for events, orjson for performance

---

## Testing

### Recommended: `pytest` + `pytest-asyncio` + `pytest-mock`

| Aspect | Detail |
|--------|--------|
| **Library** | `pytest`, `pytest-asyncio`, `pytest-mock` |
| **Purpose** | Test framework, async support, mocking |
| **Version** | pytest ≥ 7.4, pytest-asyncio ≥ 0.21, pytest-mock ≥ 3.11 |
| **License** | MIT |
| **Reason** | - Industry standard<br>- Fixtures, parametrization<br>- Async test support<br>- Mock integration |

---

## Dependency Summary

| Category | Primary | Alternative | License |
|----------|---------|-------------|---------|
| Audio Capture | sounddevice + soundfile | pyaudio | MIT / BSD |
| Keyboard | pynput | keyboard (Windows) | LGPL-3.0 |
| Camera | opencv-python | - | Apache-2.0 |
| Face Processing | mediapipe | dlib, deepface | Apache-2.0 |
| UI | PySide6 | tkinter, PyQt6 | LGPL-3.0 |
| Numerical | numpy + scipy | - | BSD |
| Config | pyyaml | - | MIT |
| Logging | structlog | stdlib logging | MIT/Apache |
| JSON | orjson | stdlib json | MIT/Apache |
| Testing | pytest + plugins | - | MIT |

---

## Installation Notes

### Platform-Specific Considerations

**Windows:**
- `sounddevice` requires PortAudio DLL (included in wheel)
- `opencv-python` wheels include FFmpeg
- `mediapipe` has Windows wheels
- `PySide6` has Windows wheels
- Global keyboard hook works without admin

**macOS:**
- Requires Camera/Microphone/Accessibility permissions (TCC)
- `mediapipe` has macOS wheels (Apple Silicon + Intel)
- May need `brew install portaudio` for sounddevice

**Linux:**
- `pip install` works but may need system deps:
  - `libportaudio2` for sounddevice
  - `libglib2.0-0`, `libgl1` for OpenCV/Qt
  - `v4l2` for camera
  - X11 for global keyboard (Wayland not supported by pynput)

### Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install sounddevice soundfile pynput opencv-python mediapipe PySide6 numpy scipy pyyaml structlog orjson pytest pytest-asyncio pytest-mock
```

### Requirements File (for documentation)
```
# Core
sounddevice>=0.4.6
soundfile>=0.12.1
pynput>=1.7.6
opencv-python>=4.8.0
mediapipe>=0.10.0
PySide6>=6.6.0
numpy>=1.24
scipy>=1.11
pyyaml>=6.0
structlog>=23.1
orjson>=3.9

# Dev
pytest>=7.4
pytest-asyncio>=0.21
pytest-mock>=3.11
```

---

## License Compliance Checklist

- [ ] All licenses identified and compatible
- [ ] LGPL libraries (pynput, PySide6) dynamically linked
- [ ] Attribution notices included in distribution
- [ ] No GPL dependencies that would viral-ize the project
- [ ] License files bundled with distribution