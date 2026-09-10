# HANDOFF — Multimodal Human State Capture Application

## 1. Project Purpose & Architecture

**Purpose**: Build a desktop application that captures and synchronizes three modalities:
1. **Voice/Audio** — microphone input
2. **Keyboard/Keystroke** — global key events
3. **Facial Expressions** — webcam + MediaPipe face landmarks

**Architecture** (modular, per SPEC.md):
```
Application
├── UI (PySide6)
├── Session Manager (lifecycle, state machine)
├── Voice Capture Module (AudioCapture, AudioProcessor, AudioStorage)
├── Keystroke Capture Module (KeyboardListener, KeyboardProcessor, KeyboardStorage)
├── Facial Capture Module (CameraCapture, FaceDetector, FaceProcessor, FaceStorage)
├── Timestamp/Synchronization Module (monotonic + wall-clock, EventSynchronizer)
├── Processing Pipeline (validation, normalization, batch)
├── Storage Module (session repo, file writers, index)
├── Configuration Module (YAML, defaults, runtime overrides)
└── Logging/Error Handling (structured JSON, sensitive data filtering)
```

**Data Flow**:
- Each modality runs in isolated thread(s)
- All events get dual timestamps (monotonic + ISO8601 wall-clock)
- EventSynchronizer buffers and orders events → writes `events.jsonl`
- Per-modality storage writes JSONL (`facial_features.jsonl`, `keystrokes.jsonl`, audio WAV chunks)

**Key Design Principles**:
- Privacy-first: never log/store passwords, sensitive keys, raw pixels, raw audio
- Graceful degradation: one modality failure doesn't kill session
- Spec-driven: SPEC.md is source of truth

---

## 2. Current Implementation Status

| Component | Status |
|-----------|--------|
| Project structure & config | ✅ Complete |
| Logging (structured JSON) | ✅ Complete |
| Session Manager (lifecycle, state machine) | ✅ Complete |
| Timestamp/Sync infrastructure | ✅ Complete |
| Keyboard capture (listener, processor, storage) | ✅ Complete |
| Audio capture (listener, processor, storage) | ✅ Complete |
| Camera capture (OpenCV) | ✅ Complete |
| Face detection (MediaPipe FaceLandmarker) | ✅ Complete |
| Face analysis (features, baseline, state estimation) | ✅ Complete |
| Keyboard analysis (features, baseline, state estimation) | ✅ Complete |
| Audio analysis (features, baseline, state estimation) | ✅ Complete |
| Unified event pipeline + synchronizer | ✅ Complete |
| Storage (session dirs, JSONL, WAV chunks) | ✅ Complete |
| UI (PySide6 main window, panels, timers) | ✅ Complete |
| Unit tests (241 tests) | ✅ All passing |
| Config (config.yaml) | ✅ Complete |
| Documentation (docs/) | ✅ Complete |

**All Sprint 1–8 work is complete** per SPEC.md development order.

---

## 3. Test Results (Current)

```bash
python -m pytest tests/unit/ -v
# 241 passed in ~30s
```

All unit tests pass across:
- test_analysis_foundation.py (24 tests)
- test_audio.py (11 tests)
- test_audio_analysis.py (22 tests)
- test_config.py / test_config_loading.py (14 tests)
- test_dependencies.py (10 tests)
- test_face.py (18 tests)
- test_face_analysis.py (26 tests)
- test_keyboard.py (19 tests)
- test_session_manager.py (22 tests)
- test_storage.py (13 tests)
- test_structure.py (3 tests)
- test_ui.py (10 tests)

**Mypy**: Shows 1 error from numpy's type stubs (Python 3.13 incompatibility with numpy<2.x), **not from our code**:
```
.venv/Lib/site-packages/numpy/__init__.pyi:737: error: Type statement is only supported in Python 3.12+
```

---

## 4. Important Files & Responsibilities

### Core / Infrastructure
| File | Responsibility |
|------|----------------|
| `src/main.py` | Entry point, launches PySide6 app |
| `src/core/config.py` | YAML config loading, validation, defaults |
| `src/core/events.py` | Unified Event schema, ModalityType, EventType, factory functions |
| `src/core/timestamps.py` | TimestampGenerator (monotonic + wall-clock), Timestamp dataclass |
| `src/core/logging.py` | Structured JSON logger setup |
| `src/core/interfaces.py` | ModalityInterface, FaceModalityInterface, etc. |
| `src/processing/synchronization.py` | EventSynchronizer (priority queue, flush thread, JSONL writer) |

### Session
| File | Responsibility |
|------|----------------|
| `src/session/manager.py` | SessionManager: lifecycle, modality coordination, state machine |
| `src/session/state.py` | SessionState enum, SessionStateMachine (READY→RECORDING↔PAUSED→STOPPED) |
| `src/session/metadata.py` | SessionMetadata, ModalityStatusInfo, session.json persistence |

### Face Modality
| File | Responsibility |
|------|----------------|
| `src/face/camera.py` | FaceCamera: OpenCV capture thread, FPS control, frame callback |
| `src/face/detector.py` | FaceDetector: MediaPipe FaceLandmarker async, 468 landmarks, 52 blendshapes, iris, head pose |
| `src/face/processor.py` | FaceProcessor: validates observations, emits face_found/lost/observation events |
| `src/face/storage.py` | FaceStorage: async JSONL writer for facial_features.jsonl |
| `src/face/capture.py` | FaceCapture: orchestrates camera→detector→processor→analysis→storage |

### Keyboard Modality
| File | Responsibility |
|------|----------------|
| `src/keyboard/listener.py` | KeyboardListener: pynput global hook, key_down/key_up, sensitive context filtering |
| `src/keyboard/processor.py` | KeyboardProcessor: queues events, enriches (duration, modifiers, window), callbacks |
| `src/keyboard/storage.py` | KeyboardStorage: async JSONL writer for keystrokes.jsonl |
| `src/keyboard/capture.py` | KeyboardCapture: orchestrates listener→processor→analysis→storage |

### Audio Modality
| File | Responsibility |
|------|----------------|
| `src/audio/listener.py` | AudioListener: sounddevice input stream, chunk callback |
| `src/audio/processor.py` | AudioProcessor: validates chunks, optional feature extraction |
| `src/audio/storage.py` | AudioStorage: WAV chunk writer + audio_index.jsonl |
| `src/audio/capture.py` | AudioCapture: orchestrates listener→processor→analysis→storage |

### Analysis (Shared + Per-Modality)
| File | Responsibility |
|------|----------------|
| `src/analysis/features.py` | BaseFeatureExtractor, TemporalWindow, statistical/rate/trend utils |
| `src/analysis/interfaces.py` | FeatureExtractor, FeatureVector, DataQuality, StateEstimator interfaces |
| `src/analysis/results.py` | FeatureVector, StateEstimate, AnalysisResult dataclasses |
| `src/analysis/temporal.py` | TemporalWindow (sliding time window, mean/std/cv) |
| `src/analysis/face/features.py` | FaceFeatureExtractor: EAR, MAR, blink rate, head pose, symmetry, expressivity |
| `src/analysis/face/state.py` | FaceStateEstimator: neutral/positive/surprise/negative/tension states |
| `src/analysis/face/baseline.py` | FaceBaselineManager: EMA warmup, per-feature normalization |
| `src/analysis/keyboard/features.py` | KeyboardFeatureExtractor: KPM, IKI, corrections, pauses, rhythm scores |
| `src/analysis/keyboard/state.py` | KeyboardStateEstimator: typing/uncertain/focused states |
| `src/analysis/keyboard/baseline.py` | KeyboardBaselineManager: similar to face |
| `src/analysis/audio/features.py` | AudioFeatureExtractor: pitch, energy, ZCR, spectral centroid, VAD |
| `src/analysis/audio/state.py` | AudioStateEstimator: silent/voice/elevated/calm states |
| `src/analysis/audio/baseline.py` | AudioBaselineManager: per-feature-group baseline |

### UI
| File | Responsibility |
|------|----------------|
| `src/ui/main_window.py` | MainWindow: session controls, modality panels, polling timers (200ms) |
| `src/ui/panels.py` | OverallHumanStatePanel, ExplanationPanel, MultimodalContributionPanel |
| `src/ui/keyboard_analysis_panel.py` | KeyboardAnalysisPanel: typing speed, IKI, corrections, rhythm |

---

## 5. Known Limitations

1. **Mypy/numpy version conflict**: numpy<2.x type stubs incompatible with Python 3.13. Workaround: ignore or upgrade numpy when available.
2. **MediaPipe model file**: Requires `face_landmarker.task` in working directory (present in repo).
3. **Windows-only global keyboard hook**: pynput works on Windows; macOS/Linux need accessibility permissions.
4. **Camera permission**: Windows privacy settings must allow camera access.
5. **UI polling timers**: 200ms interval may miss rapid changes; analysis results only available when `_last_analysis_result` is set.

---

## 6. Unresolved Issues — Live Face Detection & Keyboard Capture

### The Core Problem
**User reports**: On physical Windows laptop, GUI opens, session starts, but:
- **Face panel**: "Face: Not Detected", "State: --", "Confidence: --", "Baseline: Warming Up (0%)"
- **Keyboard panel**: "Keyboard Not Active", all metrics "--"
- Camera LED turns ON, user is in front of camera, actively typing
- Session shows "Recording"

### Root Causes Discovered & Fixed in This Session

| Issue | Location | Fix Applied |
|-------|----------|-------------|
| **1. Undefined `timestamp_monotonic` in face analysis** | `src/face/capture.py:_extract_raw_features_from_detection()` | Added `timestamp_monotonic` and `timestamp_wallclock` parameters; updated call site in `_run_face_analysis()` |
| **2. numpy.float64 serialization crash** | `src/face/detector.py:process_frame()` | Convert MediaPipe landmark `x,y,z` to `float()` before storing in `landmarks_3d` and `iris` dicts |
| **3. Keyboard analysis callback never wired** | `src/keyboard/capture.py:start()` | Added `self._processor.set_analysis_callback(self._on_analysis_event)` (line 470) |

### Evidence Fixes Work
- **All 241 unit tests pass** after fixes
- **Log analysis** (logs/app.log): 
  - Before fix: Repeated `"error": "name 'timestamp_monotonic' is not defined", "event": "face_feature_extraction_failed"`
  - After fix: No such errors in new session logs (only modality registration + start events)
  - No `keyboard_feature_extraction_failed` or `keyboard_analysis_callback_failed` errors
  - `keyboard_capture_started`, `keyboard_listener_started`, `keyboard_processor_started`, `keyboard_analysis_started` all present
- **UI polling mechanism** (`main_window.py:_poll_face_analysis`, `_poll_keyboard_analysis`) reads `_last_analysis_result` from capture objects — this is now populated by `_store_analysis_results()` (face) and `_on_analysis_event()` (keyboard)

### What Still Needs Verification
The fixes address the **code-level bugs** preventing analysis results from being generated/stored. However, the **live GUI test** (user sitting in front of camera + typing) has not been re-run in this environment because:
- This is a headless/CI-like environment (no physical camera/keyboard hook visible)
- The application requires a physical Windows laptop with camera + keyboard to fully verify

---

## 7. Debugging Work Already Attempted

1. **Traced full data pipelines** for face and keyboard:
   - Camera → Frame → Detector → Processor → Analysis → Storage
   - OS Hook → Listener → Processor → Analysis Callback → Storage

2. **Identified 3 critical bugs** via log analysis (`logs/app.log`):
   - `face_feature_extraction_failed: name 'timestamp_monotonic' is not defined` (repeated)
   - `flush_event_failed: Type is not JSON serializable: numpy.float64` (repeated)
   - Keyboard analysis callback never registered

3. **Applied minimal fixes**:
   - Face capture: pass timestamps to feature extractor
   - Face detector: convert numpy types to Python `float`/`int`
   - Keyboard capture: wire analysis callback to processor

4. **Verified fixes don't break tests**: 241/241 pass

5. **Verified application imports & starts** without crash

---

## 8. GUI Startup Issue Discovery

**Original user report**: GUI opens but shows no live data.

**Investigation findings**:
- Session state machine works: READY → RECORDING transitions confirmed in logs
- Modalities register and start: `modality_registered`, `modality_started` events in logs
- Camera initializes: `camera_started` with actual_resolution [640,480], actual_fps 30.0
- **Analysis pipeline was silently failing** due to the 3 bugs above
- UI polling timers (`_face_analysis_timer`, `_keyboard_analysis_timer`) run at 200ms but `_last_analysis_result` was never populated → UI never updates

**No GUI framework issue** — the bug was entirely in the analysis pipeline data flow.

---

## 9. What Still Needs To Be Done Next

### Immediate (Required for Acceptance)
1. **Run on physical Windows laptop** with camera + keyboard:
   ```bash
   python -m src.main
   ```
2. **Start session**, sit in front of camera, type for 10–15 seconds
3. **Verify GUI updates**:
   - Face: "Face: Detected" → State (neutral/positive/etc) + Confidence % + Baseline progress
   - Keyboard: "Keyboard Active" → Typing Speed (KPM), IKI, Corrections, Rhythm
4. **If still failing**: Add temporary debug logging in `_poll_face_analysis` / `_poll_keyboard_analysis` to trace `_last_analysis_result` presence

### Short-term
- Investigate why `analysis.get("frames_processed")` in UI polling always returns 0 (stats dict key mismatch — stats uses `frames_processed` but analysis dict may use different key)
- Consider adding `_last_analysis_result` directly to `get_stats()["analysis"]` for cleaner UI polling
- Run integration tests if any exist beyond unit tests

### Long-term (Post-Acceptance)
- Fix mypy/numpy compatibility (upgrade numpy or add type ignores)
- Add integration test for full session lifecycle with all 3 modalities
- Consider async storage writers for analysis files (currently sync `flush()`)

---

## 10. Important Commands

| Task | Command |
|------|---------|
| Run all unit tests | `python -m pytest tests/unit/ -v` |
| Run specific test file | `python -m pytest tests/unit/test_face.py -v` |
| Type check (our code only) | `python -m mypy src/core/ src/analysis/ src/session/ --ignore-missing-imports` |
| Start application | `python -m src.main` |
| View logs | `type logs\app.log` or `Get-Content logs\app.log -Tail 50` |
| Search logs for errors | `Select-String -Path logs\app.log -Pattern "error"` |

---

## 11. Key Privacy & Design Requirements (from SPEC.md)

- **Never store passwords/sensitive keys** — keyboard listener filters sensitive contexts (password fields)
- **Never log raw data** — no key values, audio samples, pixel data in logs
- **Facial expressions ≠ mental state** — state estimates are model predictions, labeled as such
- **Graceful degradation** — one modality failure continues others
- **Externalized config** — `config/config.yaml` for all settings
- **Structured JSON logging** — timestamp, level, module, message, session_id
- **Session data deletable** — user can delete recorded sessions
- **Monotonic clock for sync** — `time.monotonic()` for alignment, wall-clock for metadata

---

## 12. Repository State Snapshot

```
app_work/
├── config/config.yaml          # All modality configs
├── data/sessions/              # Auto-created session directories
├── docs/                       # 8 architecture/requirement docs
├── logs/app.log                # Structured JSON logs (large)
├── face_landmarker.task        # MediaPipe model (required)
├── pyproject.toml              # Dependencies, pytest config
├── requirements.txt            # pip dependencies
├── src/                        # All source code (modular)
│   ├── analysis/               # Feature extraction, baselines, state estimation
│   ├── audio/                  # Audio capture + analysis
│   ├── core/                   # Events, config, timestamps, logging, interfaces
│   ├── face/                   # Camera, detector, processor, storage, capture
│   ├── keyboard/               # Listener, processor, storage, capture
│   ├── processing/             # Event synchronizer
│   ├── session/                # Manager, state machine, metadata
│   ├── storage/                # Session store, loader
│   └── ui/                     # PySide6 main window, panels
└── tests/unit/                 # 241 tests, all passing
```

---

## 13. Next Session — Quick Start

```bash
cd C:\Users\theat\OneDrive\Desktop\BE_Project\my_app\app_work
python -m pytest tests/unit/ -v          # Verify tests still pass (241)
python -m src.main                       # Launch GUI on physical Windows laptop
# Click "Start Session", verify face + keyboard panels update live
```

If GUI still shows "Not Detected"/"Not Active":
- Check `logs/app.log` for new `face_feature_extraction_failed` or `keyboard_analysis_callback_failed`
- Add `logger.debug("polling face", result=face_capture._last_analysis_result)` in `_poll_face_analysis`
- Verify `_on_analysis_event` is called by adding log at entry

---

**Last Updated**: 2026-09-10 (after fixing 3 critical bugs, all 241 tests passing)
**Status**: Code complete, requires physical hardware validation