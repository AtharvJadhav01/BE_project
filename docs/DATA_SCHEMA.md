# Data Schema

## Overview

All data is organized around sessions. Each session has a unique ID and contains metadata, modality-specific data files, and a unified event log.

## Directory Structure

```
data/
└── sessions/
    └── <session_id>/
        ├── session.json              # Session metadata
        ├── audio/
        │   ├── chunk_000001.wav      # Audio chunks (PCM WAV)
        │   ├── chunk_000002.wav
        │   └── audio_index.jsonl     # Chunk metadata
        ├── keystrokes.jsonl          # Keyboard events (one per line)
        ├── typed_text.txt            # Text entered in the session's keyboard analysis field
        ├── facial_features.jsonl     # Facial observations (one per line)
        └── events.jsonl              # Unified event log (all modalities)
```

---

## 1. Session Metadata (`session.json`)

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "application_version": "1.0.0",
  "start_time_wallclock": "2026-09-05T12:00:00.000Z",
  "end_time_wallclock": "2026-09-05T12:30:00.000Z",
  "start_time_monotonic": 0.0,
  "end_time_monotonic": 1800000.0,
  "duration_ms": 1800000,
  "enabled_modalities": ["voice", "keyboard", "face"],
  "config_snapshot": {
    "audio": { "sample_rate": 16000, "channels": 1, "chunk_duration_ms": 10000 },
    "keyboard": { "capture_modifiers": true, "exclude_sensitive": true },
    "camera": { "fps": 15, "resolution": [640, 480] },
    "face": { "detection_model": "mediapipe", "landmark_model": "mediapipe" }
  },
  "status": "completed",
  "modality_status": {
    "voice": "completed",
    "keyboard": "completed",
    "face": "completed"
  },
  "error_count": 0,
  "events_file": "events.jsonl",
  "audio_index_file": "audio/audio_index.jsonl"
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| session_id | string (UUIDv4) | Yes | Unique session identifier |
| application_version | string | Yes | Semantic version of the application |
| start_time_wallclock | string (ISO 8601) | Yes | Session start in UTC |
| end_time_wallclock | string (ISO 8601) | Yes | Session end in UTC |
| start_time_monotonic | number | Yes | Monotonic time at start (ms) |
| end_time_monotonic | number | Yes | Monotonic time at end (ms) |
| duration_ms | integer | Yes | Session duration in milliseconds |
| enabled_modalities | array[string] | Yes | List of active modalities |
| config_snapshot | object | Yes | Full config used for this session |
| status | enum | Yes | `ready`, `recording`, `paused`, `stopped`, `completed`, `failed` |
| modality_status | object | Yes | Per-modality status |
| error_count | integer | Yes | Total errors during session |
| events_file | string | Yes | Relative path to unified events |
| audio_index_file | string | No | Relative path to audio index |

---

## 2. Audio Data

### Audio Chunks (`audio/chunk_XXXXXX.wav`)
- Format: WAV, PCM, 16-bit
- Sample rate: Configurable (default 16000 Hz)
- Channels: Configurable (default 1 mono)
- Chunk duration: Configurable (default 10000 ms)

### Audio Index (`audio/audio_index.jsonl`)

Each line is a JSON object:

```json
{
  "chunk_index": 1,
  "filename": "chunk_000001.wav",
  "start_time_monotonic": 0.0,
  "end_time_monotonic": 10000.0,
  "start_time_wallclock": "2026-09-05T12:00:00.000Z",
  "duration_ms": 10000,
  "sample_rate": 16000,
  "channels": 1,
  "sample_count": 160000,
  "bytes": 320000,
  "rms_db": -24.5
}
```

| Field | Type | Description |
|-------|------|-------------|
| chunk_index | integer | Sequential chunk number (1-based) |
| filename | string | WAV filename |
| start_time_monotonic | number | Session-relative start (ms) |
| end_time_monotonic | number | Session-relative end (ms) |
| start_time_wallclock | string | Absolute start time (ISO 8601) |
| duration_ms | integer | Chunk duration (ms) |
| sample_rate | integer | Samples per second |
| channels | integer | Number of channels |
| sample_count | integer | Total samples in chunk |
| bytes | integer | File size in bytes |
| rms_db | number | RMS level in dBFS (optional, for voice activity) |

---

## 3. Keyboard Events (`keystrokes.jsonl`)

Each line is a JSON object:

```json
{
  "event_id": "kbd_000001",
  "timestamp_monotonic": 1234.567,
  "timestamp_wallclock": "2026-09-05T12:00:01.234Z",
  "event_type": "key_down",
  "key": "a",
  "key_code": 65,
  "modifiers": ["ctrl", "shift"],
  "duration_ms": null,
  "is_printable": true,
  "is_sensitive_context": false,
  "window_title": "Notepad",
  "process_name": "notepad.exe"
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| event_id | string | Yes | Unique event ID (`kbd_<seq>`) |
| timestamp_monotonic | number | Yes | Session-relative timestamp (ms, float) |
| timestamp_wallclock | string | Yes | Absolute timestamp (ISO 8601) |
| event_type | enum | Yes | `key_down`, `key_up` |
| key | string | Yes | Key identifier (e.g., `"a"`, `"Enter"`, `"Shift_L"`) |
| key_code | integer | No | Platform-specific key code |
| modifiers | array[string] | No | Active modifiers: `ctrl`, `alt`, `shift`, `cmd`, `win` |
| duration_ms | number | No | For `key_up`: time since matching `key_down` |
| is_printable | boolean | Yes | Whether key produces printable character |
| is_sensitive_context | boolean | Yes | True if detected in password/sensitive field |
| window_title | string | No | Active window title (redacted if sensitive) |
| process_name | string | No | Active process name |

### Key Naming Convention
- Printable: `"a"`, `"1"`, `"@"`, `" "` (space)
- Special: `"Enter"`, `"Tab"`, `"Backspace"`, `"Escape"`, `"Space"`
- Modifiers: `"Shift_L"`, `"Shift_R"`, `"Control_L"`, `"Control_R"`, `"Alt_L"`, `"Alt_R"`, `"Meta_L"`, `"Meta_R"`
- Function: `"F1"`..`"F12"`
- Navigation: `"Up"`, `"Down"`, `"Left"`, `"Right"`, `"Home"`, `"End"`, `"Page_Up"`, `"Page_Down"`
- Numpad: `"Num_0"`..`"Num_9"`, `"Num_Enter"`, `"Num_Add"`, etc.

### Sensitive Context Detection
- `is_sensitive_context: true` when:
  - Active window title contains: `password`, `login`, `signin`, `credential`, `secret`, `token`, `pin`, `cvv`, `ssn`
  - Active control is a password field (OS accessibility API where available)
- When `is_sensitive_context: true`:
  - `key` is replaced with `"[REDACTED]"`
  - `window_title` is replaced with `"[REDACTED]"`
  - Event still recorded for timing analysis but content protected

---

### Dedicated Session Text (`typed_text.txt`)

This UTF-8 plain-text file contains text entered in the dedicated keyboard analysis field. It is rewritten as the text changes, so the latest content is available during the session. It is created for a session when the field is used. The field is disabled while paused and after the session stops.

Unlike `keystrokes.jsonl`, this file contains the user's actual text. Users should not enter passwords or other sensitive information in the analysis field.

---

## 4. Facial Observations (`facial_features.jsonl`)

Each line is a JSON object (one per processed frame where face detected):

```json
{
  "observation_id": "face_000001",
  "timestamp_monotonic": 1234.567,
  "timestamp_wallclock": "2026-09-05T12:00:01.234Z",
  "frame_index": 185,
  "face_detected": true,
  "detection_confidence": 0.98,
  "bounding_box": {
    "x": 245,
    "y": 120,
    "width": 150,
    "height": 150
  },
  "landmarks_3d": [
    {"x": 0.45, "y": 0.38, "z": -0.02},
    ...
  ],
  "landmarks_2d": [
    {"x": 287, "y": 185},
    ...
  ],
  "blendshapes": {
    "browDownLeft": 0.12,
    "browDownRight": 0.11,
    "browInnerUp": 0.05,
    "browOuterUpLeft": 0.03,
    "browOuterUpRight": 0.02,
    "eyeBlinkLeft": 0.01,
    "eyeBlinkRight": 0.01,
    "eyeLookDownLeft": 0.0,
    "eyeLookDownRight": 0.0,
    "eyeLookInLeft": 0.0,
    "eyeLookInRight": 0.0,
    "eyeLookOutLeft": 0.0,
    "eyeLookOutRight": 0.0,
    "eyeLookUpLeft": 0.0,
    "eyeLookUpRight": 0.0,
    "eyeSquintLeft": 0.05,
    "eyeSquintRight": 0.04,
    "eyeWideLeft": 0.0,
    "eyeWideRight": 0.0,
    "jawForward": 0.0,
    "jawLeft": 0.0,
    "jawOpen": 0.02,
    "jawRight": 0.0,
    "mouthClose": 0.95,
    "mouthDimpleLeft": 0.0,
    "mouthDimpleRight": 0.0,
    "mouthFrownLeft": 0.0,
    "mouthFrownRight": 0.0,
    "mouthFunnel": 0.0,
    "mouthLeft": 0.0,
    "mouthLowerDownLeft": 0.0,
    "mouthLowerDownRight": 0.0,
    "mouthPressLeft": 0.0,
    "mouthPressRight": 0.0,
    "mouthPucker": 0.0,
    "mouthRight": 0.0,
    "mouthRollLower": 0.0,
    "mouthRollUpper": 0.0,
    "mouthShrugLower": 0.0,
    "mouthShrugUpper": 0.0,
    "mouthSmileLeft": 0.15,
    "mouthSmileRight": 0.14,
    "mouthStretchLeft": 0.0,
    "mouthStretchRight": 0.0,
    "mouthUpperUpLeft": 0.0,
    "mouthUpperUpRight": 0.0,
    "noseSneerLeft": 0.0,
    "noseSneerRight": 0.0
  },
  "iris": {
    "left": {"x": 0.42, "y": 0.36, "z": -0.01},
    "right": {"x": 0.55, "y": 0.36, "z": -0.01}
  },
  "head_pose": {
    "pitch": 2.1,
    "yaw": -1.5,
    "roll": 0.3
  },
  "processing_time_ms": 12.4
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| observation_id | string | Yes | Unique ID (`face_<seq>`) |
| timestamp_monotonic | number | Yes | Session-relative timestamp (ms) |
| timestamp_wallclock | string | Yes | Absolute timestamp (ISO 8601) |
| frame_index | integer | Yes | Sequential frame number |
| face_detected | boolean | Yes | Whether face was found |
| detection_confidence | number | No | Face detection confidence (0-1) |
| bounding_box | object | No | Face bbox in pixel coordinates |
| landmarks_3d | array | No | 468 3D landmarks (normalized 0-1, z relative) |
| landmarks_2d | array | No | 468 2D landmarks (pixel coordinates) |
| blendshapes | object | No | 52 MediaPipe blendshape coefficients (0-1) |
| iris | object | No | Left/right iris centers (normalized) |
| head_pose | object | No | Euler angles in degrees |
| processing_time_ms | number | No | Time to process this frame |

### When Face Not Detected
```json
{
  "observation_id": "face_000002",
  "timestamp_monotonic": 1267.890,
  "timestamp_wallclock": "2026-09-05T12:00:01.267Z",
  "frame_index": 186,
  "face_detected": false,
  "detection_confidence": 0.0,
  "processing_time_ms": 3.1
}
```

### Blendshape Reference (MediaPipe Face Mesh)
52 coefficients representing facial action units. Values 0.0-1.0.
Key groups: brows, eyes, mouth, jaw, nose. See MediaPipe docs for full list.

---

## 5. Unified Events (`events.jsonl`)

Normalized schema for cross-modal analysis. Each modality maps its native events to this structure.

```json
{
  "event_id": "evt_000001",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp_monotonic": 1234.567,
  "timestamp_wallclock": "2026-09-05T12:00:01.234Z",
  "modality": "keyboard",
  "event_type": "key_down",
  "data": {
    "key": "a",
    "modifiers": ["ctrl"],
    "is_printable": true
  }
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| event_id | string | Yes | Unique across session (`evt_<seq>`) |
| session_id | string | Yes | Session UUID |
| timestamp_monotonic | number | Yes | Session-relative (ms, float) |
| timestamp_wallclock | string | Yes | Absolute (ISO 8601) |
| modality | enum | Yes | `voice`, `keyboard`, `face`, `system` |
| event_type | string | Yes | Modality-specific type |
| data | object | Yes | Modality-specific payload |

### Modality Event Types

#### Voice (`modality: "voice"`)
| event_type | data fields |
|------------|-------------|
| `audio_chunk_start` | `chunk_index`, `sample_rate`, `channels` |
| `audio_chunk_end` | `chunk_index`, `duration_ms`, `rms_db` |
| `voice_activity` | `chunk_index`, `is_speech`, `confidence` |

#### Keyboard (`modality: "keyboard"`)
| event_type | data fields |
|------------|-------------|
| `key_down` | `key`, `key_code`, `modifiers`, `is_printable`, `is_sensitive_context` |
| `key_up` | `key`, `key_code`, `modifiers`, `duration_ms`, `is_printable` |

#### Face (`modality: "face"`)
| event_type | data fields |
|------------|-------------|
| `face_observation` | `frame_index`, `face_detected`, `detection_confidence`, `blendshapes`, `head_pose`, `iris` |
| `face_lost` | `last_frame_index`, `duration_ms` |
| `face_found` | `frame_index`, `detection_confidence` |

#### System (`modality: "system"`)
| event_type | data fields |
|------------|-------------|
| `session_start` | `config_snapshot` |
| `session_end` | `duration_ms`, `modality_status` |
| `session_pause` | `paused_modalities` |
| `session_resume` | `resumed_modalities` |
| `modality_error` | `modality`, `error_type`, `message`, `recovered` |
| `modality_status_change` | `modality`, `status` |

---

## 6. Timestamp Structure (All Events)

Every timestamped event contains both time references:

```json
{
  "timestamp_monotonic": 1234567.890123,
  "timestamp_wallclock": "2026-09-05T12:00:01.234567Z"
}
```

### Conversion
```
wallclock = session_start_wallclock + (monotonic / 1000)
monotonic = (wallclock - session_start_wallclock) * 1000
```

### Precision Requirements
- Monotonic: Microsecond precision (float64)
- Wallclock: Microsecond precision (ISO 8601 with 6 decimal places)
- Synchronization accuracy target: < 5ms between modalities

---

## 7. Schema Versioning

Each session stores schema version in `session.json`:
```json
{
  "schema_version": "1.0",
  ...
}
```

Future changes:
- Minor version: Additive fields only (backward compatible)
- Major version: Breaking changes, migration script provided

---

## 8. Data Validation

### Required Fields by Modality

| Modality | Required Fields |
|----------|-----------------|
| All | `event_id`, `timestamp_monotonic`, `timestamp_wallclock`, `modality`, `event_type`, `data` |
| Voice | `chunk_index`, `sample_rate` |
| Keyboard | `key`, `event_type` (key_down/key_up) |
| Face | `frame_index`, `face_detected` |

### Validation Rules
1. `timestamp_monotonic` must be non-decreasing within a modality stream
2. `event_id` must be unique within session
3. `session_id` in events must match `session.json`
4. Audio chunks must have contiguous, non-overlapping time ranges
5. Keyboard `key_up` should have matching `key_down` (allow orphan key_up for keys held at session start)
6. Face observations at configured FPS interval (±10%)

---

## 9. Example Complete Session

```
data/sessions/550e8400-e29b-41d4-a716-446655440000/
├── session.json
├── audio/
│   ├── chunk_000001.wav
│   ├── chunk_000002.wav
│   └── audio_index.jsonl
├── keystrokes.jsonl
├── facial_features.jsonl
└── events.jsonl
```

### Cross-Reference Example

**Audio chunk 1**: 0-10000ms monotonic
**Keystroke at 1234.567ms**: `key_down` for "a"
**Face observation at 1234.567ms**: blendshapes showing smile
**All three appear in events.jsonl with same timestamp_monotonic**

Analysis tool can join on `timestamp_monotonic` to align modalities.