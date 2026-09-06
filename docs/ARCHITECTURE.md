# Architecture

## Overview

The application follows a modular architecture with clearly separated concerns. Each modality (voice, keyboard, face) has its own capture module, processor, and storage path, all coordinated by a central Session Manager and synchronized through a Timestamp/Synchronization module.

## System Components

```
Application
│
├── UI (Desktop Application)
│   ├── Status Display
│   ├── Modality Indicators
│   ├── Session Controls
│   └── Session Info Panel
│
├── Session Manager
│   ├── Session Lifecycle (start/stop/pause/resume)
│   ├── Session Metadata Management
│   ├── Modality Coordination
│   └── State Machine
│
├── Voice Capture Module
│   ├── AudioCapture (microphone interface)
│   ├── AudioProcessor (feature extraction, chunking)
│   └── AudioStorage (WAV file writing)
│
├── Keystroke Capture Module
│   ├── KeyboardListener (global/local hook)
│   ├── KeyboardProcessor (filtering, enrichment)
│   └── KeyboardStorage (JSONL writing)
│
├── Facial Capture Module
│   ├── CameraCapture (webcam interface)
│   ├── FaceDetector (face detection)
│   ├── FaceProcessor (landmarks, expressions)
│   └── FaceStorage (JSONL writing)
│
├── Timestamp/Synchronization Module
│   ├── TimeSource (monotonic + wall-clock)
│   ├── TimestampGenerator
│   └── SynchronizationEngine
│
├── Processing Pipeline
│   ├── Validation
│   ├── Event Normalization
│   └── Batch Processing
│
├── Storage Module
│   ├── Session Repository
│   ├── File Writers (per modality)
│   └── Index Management
│
├── Configuration Module
│   ├── Config Loader (YAML)
│   ├── Defaults Management
│   └── Runtime Overrides
│
└── Logging/Error Handling
    ├── Structured Logger
    ├── Error Aggregator
    └── Recovery Strategies
```

## Data Flow

### Voice Data Flow
```
Microphone → AudioCapture (raw frames)
    → AudioProcessor (chunking, optional feature extraction)
    → TimestampedAudio (monotonic timestamp + wall-clock)
    → AudioStorage (WAV chunks + metadata)
    → Session Storage
```

### Keystroke Data Flow
```
OS Keyboard Hook → KeyboardListener (key_down/key_up)
    → KeyboardProcessor (filter sensitive, add context, duration)
    → TimestampedKeyboardEvent
    → KeyboardStorage (JSONL append)
    → Session Storage
```

### Facial Data Flow
```
Webcam → CameraCapture (frames at configured FPS)
    → FaceDetector (bounding box, confidence)
    → FaceProcessor (landmarks, action units, expression features)
    → TimestampedFacialObservation
    → FaceStorage (JSONL append)
    → Session Storage
```

### Unified Event Pipeline
```
All Modalities → Validation → Normalization → Event Bus
    → SynchronizationEngine (align by timestamp)
    → Unified Events (JSONL)
    → Storage
```

## Timestamp Synchronization

### Time Sources
- **Monotonic Clock** (`time.monotonic()` / `QueryPerformanceCounter`): Used for elapsed session time, inter-modality alignment, and duration measurements. Immune to system clock adjustments.
- **Wall-Clock** (`time.time()` / `DateTime.UtcNow`): Used for session metadata (start_time, end_time) and absolute timestamps in stored data.

### Synchronization Strategy
1. At session start, record `session_start_monotonic` and `session_start_wallclock`
2. Each modality captures using monotonic timestamps relative to session start
3. For storage, convert to absolute wall-clock: `wallclock = session_start_wallclock + monotonic_elapsed`
4. All events store both monotonic (for precision alignment) and wall-clock (for human readability)
4. SynchronizationEngine aligns events by monotonic timestamp during export/analysis

### Event Timestamp Structure
```json
{
  "timestamp_monotonic": 12345.678901,
  "timestamp_wallclock": "2026-09-05T12:34:56.789Z",
  "session_elapsed_ms": 12345.678901
}
```

## Session Management

### Session States
```
READY → RECORDING ↔ PAUSED → STOPPED
                ↑         ↓
                └─────────┘ (resume)
```

### Session Metadata (`session.json`)
```json
{
  "session_id": "uuid-v4",
  "start_time": "2026-09-05T12:00:00.000Z",
  "end_time": "2026-09-05T12:30:00.000Z",
  "duration_ms": 1800000,
  "enabled_modalities": ["voice", "keyboard", "face"],
  "application_version": "1.0.0",
  "config_snapshot": { ... },
  "status": "completed"
}
```

### Session Lifecycle
1. **Start**: Generate session_id, record start times, initialize enabled modalities, create session directory
2. **Record**: All enabled modalities capture; UI shows RECORDING
3. **Pause**: Signal modalities to pause capture; UI shows PAUSED
4. **Resume**: Signal modalities to resume; UI shows RECORDING
5. **Stop**: Signal modalities to stop, flush buffers, write session.json, close files; UI shows STOPPED

## Storage

### Directory Structure
```
data/
└── sessions/
    └── <session_id>/
        ├── session.json              # Session metadata
        ├── audio/
        │   ├── chunk_000001.wav      # Audio chunks (configurable duration)
        │   ├── chunk_000002.wav
        │   └── audio_index.jsonl     # Chunk metadata (timestamp, duration, sample_rate)
        ├── keystrokes.jsonl          # One line per key event
        ├── facial_features.jsonl     # One line per facial observation
        └── events.jsonl              # Unified events (all modalities)
```

### Storage Format
- **Audio**: WAV files (PCM), chunked by configurable duration (e.g., 10s chunks)
- **Keystrokes**: JSONL (one event per line)
- **Facial Features**: JSONL (one observation per line)
- **Unified Events**: JSONL (normalized schema for cross-modal analysis)

### File Writers
Each modality has a dedicated async file writer with:
- Buffered writes (configurable flush interval)
- Automatic rotation/flushing on pause/stop
- Error handling with retry and fallback

## Error Handling

### Modality Failure Modes
| Modality | Failure | Response |
|----------|---------|----------|
| Voice | No microphone / permission denied | Log error, disable voice, continue others |
| Voice | Capture error mid-session | Log error, attempt restart once, then disable |
| Keyboard | Hook installation failed | Log error, disable keyboard, continue others |
| Keyboard | Event processing error | Log error, drop event, continue |
| Face | No camera / permission denied | Log error, disable face, continue others |
| Face | Detection/processing error | Log error, skip frame, continue |

### Error Recovery
- Each modality runs in isolated thread/process
- Supervisor pattern: Session Manager monitors modality health
- Graceful degradation: session continues with available modalities
- Errors logged with session_id for traceability

## Module Communication

### Event Bus
- In-process pub/sub for loose coupling
- Modalities publish raw events → Processing Pipeline subscribes
- UI subscribes to status events

### Interfaces
```python
# Common Modality Interface
class Modality:
    def start(session_id: str, config: dict) -> None
    def stop() -> None
    def pause() -> None
    def resume() -> None
    def get_status() -> ModalityStatus
    def on_error(error: Exception) -> None
```

### Configuration Flow
```
config.yaml → Config Module → validated Config object
    → Session Manager (passes to modalities on start)
    → Each Modality (uses relevant section)
```

## UI Architecture

### Technology: Tkinter (stdlib) / CustomTkinter / PySide6
- Main thread: UI event loop
- Background threads: modality capture, processing, storage
- Thread-safe communication via queue + UI update callbacks

### UI Components
- **StatusBar**: Application state, modality indicators
- **SessionPanel**: Controls, session info, duration timer
- **LogPanel**: Real-time structured logs (filtered, no sensitive data)

## Extensibility

### Adding a New Modality
1. Implement `Modality` interface
2. Register in Session Manager
3. Add config section
4. Add storage writer
5. Add UI indicator

### Plugin Architecture (Future)
- Entry points for modality plugins
- Shared interfaces in `src/core/interfaces.py`
- Dynamic discovery via `importlib.metadata`

## Threading Model

```
Main Thread (UI)
    │
    ├── Session Manager (coordinates)
    │       │
    │       ├── Audio Thread (capture + process)
    │       ├── Keyboard Thread (event loop)
    │       ├── Camera Thread (capture)
    │       ├── Face Processing Thread (detection + features)
    │       ├── Storage Thread (async writes)
    │       └── Sync Thread (timestamp alignment)
    │
    └── Logging (thread-safe handler)
```

- Capture threads: Real-time priority where possible
- Processing threads: Standard priority
- Storage thread: Batch writes, low priority
- All inter-thread communication via thread-safe queues

## Configuration

### Config Schema
```yaml
audio:
  enabled: true
  sample_rate: 16000
  channels: 1
  format: "wav"
  chunk_duration_ms: 10000

keyboard:
  enabled: true
  capture_modifiers: true
  capture_context: true
  exclude_sensitive: true

camera:
  enabled: true
  fps: 15
  resolution: [640, 480]

face:
  enabled: true
  detection_model: "mediapipe"
  landmark_model: "mediapipe"
  expression_model: "none"  # or "fer", "deepface"

storage:
  output_directory: "data/sessions"
  audio_chunk_duration_ms: 10000
  flush_interval_ms: 5000

logging:
  level: "INFO"
  format: "json"
  file: "logs/app.log"
```

### Config Loading
- Load at startup
- Validate against schema
- Snapshot saved with each session
- Runtime overrides via UI (persist to config)

## Logging

### Structured Format (JSON)
```json
{
  "timestamp": "2026-09-05T12:34:56.789Z",
  "level": "INFO",
  "module": "AudioCapture",
  "message": "Microphone initialized",
  "session_id": "abc-123",
  "metadata": { "sample_rate": 16000 }
}
```

### Log Levels
- DEBUG: Detailed diagnostic info
- INFO: General operational events
- WARNING: Recoverable issues
- ERROR: Modality failures, data loss
- CRITICAL: Session termination

### Sensitive Data Filtering
- Keyboard events: Never log key values
- Audio: Never log raw samples
- Face: Never log raw pixels

## Deployment Considerations

### Platform Support
- Primary: Windows 10/11
- Secondary: macOS, Linux (X11/Wayland)

### Distribution
- PyInstaller / cx_Freeze for standalone executable
- No external runtime dependencies beyond Python stdlib + selected packages

### Permissions
- Windows: Microphone/Camera privacy settings
- macOS: TCC permissions (microphone, camera, accessibility for global keyboard)
- Linux: PulseAudio/PipeWire, V4L2, uinput/evdev