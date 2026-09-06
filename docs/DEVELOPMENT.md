# Development Workflow

## Overview

This document defines the development workflow for the Multimodal Human State Capture Application following Spec-Driven Development (SDD). All implementation must follow the specification in `SPEC.md` and the architecture defined in `docs/ARCHITECTURE.md`.

## Development Phases (from SPEC.md §25)

| Step | Phase | Deliverable |
|------|-------|-------------|
| 1 | Project Structure | Directory layout, pyproject.toml, config.yaml |
| 2 | Configuration System | Config loader, validation, defaults |
| 3 | Logging | Structured logger, file + console output |
| 4 | Session Manager | Session lifecycle, state machine, metadata |
| 5 | Timestamp/Sync Infrastructure | Time source, timestamp generator, sync engine |
| 6 | Keyboard Capture | Listener, processor, storage, privacy filter |
| 7 | Audio Capture | Stream capture, chunking, WAV writing |
| 8 | Camera Capture | Frame capture, face detection |
| 9 | Facial Feature Processing | Landmarks, blendshapes, observations |
| 10 | Unified Event Pipeline | Normalization, event bus, unified JSONL |
| 11 | Storage | Session repository, file writers, index |
| 12 | UI | Main window, status, controls, session list |
| 13 | Testing | Unit, integration, failure tests |
| 14 | Documentation | Update all docs, README |

**Rule**: Do not attempt to build everything simultaneously. Complete each step before moving to the next.

---

## Environment Setup

### Prerequisites
- Python 3.11+
- Git
- Virtual environment tool (venv, conda, or uv)

### Initial Setup
```bash
# Clone / navigate to project
cd C:\Users\theat\OneDrive\Desktop\BE_Project\my_app\app_work

# Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies (after LIBRARIES.md finalized)
pip install -r requirements.txt

# Install dev dependencies
pip install -r requirements-dev.txt
```

### Project Structure (Target)
```
app_work/
├── SPEC.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── config/
│   └── config.yaml
├── docs/
│   ├── REQUIREMENTS.md
│   ├── ARCHITECTURE.md
│   ├── LIBRARIES.md
│   ├── DATA_SCHEMA.md
│   ├── PRIVACY.md
│   └── DEVELOPMENT.md
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── timestamps.py
│   │   ├── events.py
│   │   └── interfaces.py
│   ├── session/
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   ├── state.py
│   │   └── metadata.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── capture.py
│   │   ├── processor.py
│   │   └── storage.py
│   ├── keyboard/
│   │   ├── __init__.py
│   │   ├── listener.py
│   │   ├── processor.py
│   │   └── storage.py
│   ├── face/
│   │   ├── __init__.py
│   │   ├── capture.py
│   │   ├── detector.py
│   │   ├── processor.py
│   │   └── storage.py
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   └── synchronization.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── repository.py
│   │   └── writers.py
│   └── ui/
│       ├── __init__.py
│       ├── main_window.py
│       ├── status_panel.py
│       ├── session_panel.py
│       └── log_panel.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_timestamps.py
│   │   ├── test_events.py
│   │   ├── test_session.py
│   │   └── test_storage.py
│   ├── integration/
│   │   ├── test_session_lifecycle.py
│   │   ├── test_audio_capture.py
│   │   ├── test_keyboard_capture.py
│   │   ├── test_camera_capture.py
│   │   ├── test_face_processing.py
│   │   ├── test_storage.py
│   │   └── test_synchronization.py
│   └── failure/
│       ├── test_camera_failure.py
│       ├── test_microphone_failure.py
│       ├── test_config_errors.py
│       └── test_storage_failure.py
└── data/
    └── sessions/
```

---

## Coding Standards

### Python Style
- **Formatter**: `ruff format` (or `black`)
- **Linter**: `ruff check` (or `flake8` + `mypy`)
- **Type Hints**: Required for all public functions/classes
- **Docstrings**: Google style for public APIs
- **Line Length**: 100 characters

### Code Organization
- One class per file (generally)
- Private helpers prefixed with `_`
- No circular imports (enforce with `import-linter` or layer architecture)
- Stdlib imports first, then third-party, then local

### Configuration
- All config via `config.yaml` loaded at startup
- No hardcoded values in capture/processing code
- Config validated with `pydantic` or manual schema check

### Error Handling
- Custom exceptions per module (`AudioCaptureError`, `KeyboardCaptureError`, etc.)
- Never catch bare `Exception` — catch specific exceptions
- Log errors with full context (session_id, modality, operation)
- Fail gracefully: modality failure ≠ session failure

### Threading
- Use `threading` + `queue.Queue` for inter-thread communication
- UI updates via Qt signals/slots (thread-safe)
- No shared mutable state between threads
- Locks only for brief critical sections

### Logging
```python
import structlog

logger = structlog.get_logger(__name__)

# In functions:
logger.info("microphone_initialized", sample_rate=16000, session_id=session_id)
logger.error("capture_failed", error=str(e), modality="audio", session_id=session_id)
```

---

## Testing Requirements

### Test Organization
- **Unit tests**: Fast, isolated, mock external dependencies
- **Integration tests**: Real components, may need hardware (marked `@pytest.mark.hardware`)
- **Failure tests**: Simulate errors, verify graceful degradation

### Running Tests
```bash
# All tests
pytest

# Unit only (fast)
pytest tests/unit -v

# Integration (may need camera/mic)
pytest tests/integration -v -m "not hardware"

# With hardware
pytest tests/integration -v -m hardware

# Coverage
pytest --cov=src --cov-report=term-missing
```

### Test Coverage Targets
| Component | Target |
|-----------|--------|
| Core (config, timestamps, events) | 90%+ |
| Session Manager | 85%+ |
| Capture Modules | 70%+ (hardware-dependent) |
| Processing | 80%+ |
| Storage | 85%+ |
| UI | 50%+ (UI testing harder) |

### Test Patterns

**Unit Test Example:**
```python
# tests/unit/test_timestamps.py
import pytest
from src.core.timestamps import TimestampGenerator

def test_monotonic_increases():
    gen = TimestampGenerator()
    t1 = gen.now_monotonic_ms()
    t2 = gen.now_monotonic_ms()
    assert t2 >= t1

def test_wallclock_conversion():
    gen = TimestampGenerator(session_start_wallclock=1700000000.0)
    monotonic = 5000.0  # 5 seconds
    wallclock = gen.monotonic_to_wallclock(monotonic)
    assert abs(wallclock - 1700000005.0) < 0.001
```

**Integration Test Example:**
```python
# tests/integration/test_audio_capture.py
import pytest
from src.audio.capture import AudioCapture

@pytest.mark.hardware
def test_audio_capture_real_device(tmp_path):
    capture = AudioCapture(sample_rate=16000, channels=1)
    capture.start(tmp_path / "test.wav")
    # ... wait ...
    capture.stop()
    assert (tmp_path / "test.wav").exists()
```

**Failure Test Example:**
```python
# tests/failure/test_microphone_failure.py
from unittest.mock import patch
from src.audio.capture import AudioCapture

def test_microphone_unavailable_graceful():
    with patch('sounddevice.query_devices', side_effect=OSError("No device")):
        capture = AudioCapture()
        result = capture.start("test.wav")
        assert result is False  # Failed gracefully
        assert capture.status == "error"
```

---

## Git Workflow

### Branching
- `main`: Stable, documented, tested
- Feature branches: `feature/<name>` from main
- No long-lived branches

### Commits
- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- One logical change per commit
- Reference issue/task if applicable

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

---

## Code Review Checklist

Before merging any PR:

- [ ] Follows SPEC.md requirements
- [ ] Matches ARCHITECTURE.md design
- [ ] Uses libraries from LIBRARIES.md (no new deps without approval)
- [ ] Data schema matches DATA_SCHEMA.md
- [ ] Privacy controls implemented (PRIVACY.md)
- [ ] Tests added/updated for new functionality
- [ ] All tests pass
- [ ] Linting passes (`ruff check`, `mypy`)
- [ ] Documentation updated if API changed
- [ ] No hardcoded config values
- [ ] No sensitive data logged
- [ ] Error handling follows graceful degradation
- [ ] Thread safety verified

---

## Dependency Management

### Adding a New Dependency
1. Check LIBRARIES.md — is there an existing library?
2. Evaluate: maintenance, license, compatibility, size
3. Document in LIBRARIES.md with rationale
4. Add to `requirements.txt` and `requirements-dev.txt`
5. Update `pyproject.toml`
6. Run `pip install` and verify

### Updating Dependencies
- Monthly review for security updates
- Pin versions in requirements.txt
- Test after updates

---

## Building and Distribution

### Development Run
```bash
python src/main.py
```

### Package Build (Future)
```bash
pip install build
python -m build
```

### Standalone Executable (Future)
```bash
pip install pyinstaller
pyinstaller --clean --windowed --name "MultimodalCapture" src/main.py
```

---

## Debugging Tips

### Audio Issues
- Check `sounddevice.query_devices()` for available devices
- Verify sample rate supported by device
- Monitor callback underruns (status.flags)

### Keyboard Issues
- Windows: Test with admin if global hook fails
- macOS: Grant Accessibility permission
- Linux: Ensure X11 session, not Wayland

### Camera Issues
- `cv2.VideoCapture(0)` — try indices 0, 1, 2
- Check backend: `cv2.CAP_DSHOW` (Windows), `cv2.CAP_V4L2` (Linux)
- Verify resolution/FPS supported

### Face Processing Issues
- MediaPipe requires RGB input (`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`)
- `static_image_mode=False` for video
- `refine_landmarks=True` for iris

### Synchronization Issues
- Log monotonic timestamps from each modality at start
- Verify conversion to wallclock matches
- Check for clock drift (monotonic vs wallclock)

---

## Performance Profiling

### CPU Profiling
```bash
python -m cProfile -o profile.stats src/main.py
# Analyze with snakeviz or pstats
```

### Memory Profiling
```bash
pip install memray
python -m memray run src/main.py
memray flamegraph memray-results.bin
```

### Key Metrics to Monitor
- Audio callback latency (< 10ms)
- Face processing time per frame (< 33ms for 30 FPS)
- UI update latency (< 16ms for 60 FPS)
- Disk write throughput (async, non-blocking)

---

## Documentation Maintenance

### When to Update Docs
- New module added → ARCHITECTURE.md
- New dependency → LIBRARIES.md
- Schema change → DATA_SCHEMA.md (bump version)
- Privacy impact → PRIVACY.md
- Process change → DEVELOPMENT.md
- User-facing change → README.md

### Documentation Standards
- Markdown format
- Code examples must be runnable
- Diagrams in Mermaid (rendered in GitHub/GitLab)
- Cross-reference with `[link](file.md#section)`

---

## Release Criteria (from SPEC.md §26)

Application is complete when ALL are verified:

- [ ] Application can start and stop a session
- [ ] Microphone input can be captured
- [ ] Keyboard events can be captured
- [ ] Webcam frames can be captured
- [ ] Facial features/expressions extracted using appropriate library
- [ ] All observations contain synchronized timestamps
- [ ] Three modalities associated with common session
- [ ] Data can be stored and retrieved
- [ ] One failed modality does not terminate session
- [ ] UI displays capture status
- [ ] Configuration externalized
- [ ] Errors logged
- [ ] Sensitive keyboard information not intentionally stored
- [ ] Tests exist for important components
- [ ] Documentation explains installation, architecture, dependencies, data format, privacy

---

## Troubleshooting Common Issues

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| `ModuleNotFoundError` | Venv not activated | Activate `.venv` |
| `sounddevice.PortAudioError` | No microphone / wrong device | Check `query_devices()`, set `device=` |
| `cv2.VideoCapture` returns False | Camera in use / wrong index | Try indices 0-3, close other apps |
| `pynput` listener doesn't fire | macOS: no Accessibility permission | Grant in System Settings |
| `mediapipe` import fails | Version mismatch / arch | Reinstall, check Python version |
| UI freezes during capture | Blocking call in main thread | Move to QThread/worker |
| Timestamp drift | Wallclock vs monotonic mix | Use monotonic for alignment |
| JSONL parsing fails | Partial line written | Use atomic writes or verify complete lines |

---

## Continuous Integration (Future)

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: ruff check src tests
      - run: mypy src
      - run: pytest tests/unit -v
      - run: pytest tests/integration -v -m "not hardware"
```

---

## Getting Help

- Check SPEC.md for requirements
- Check ARCHITECTURE.md for design
- Check LIBRARIES.md for library usage
- Check existing tests for patterns
- Search codebase for similar implementations
- Ask before adding new dependencies or changing architecture