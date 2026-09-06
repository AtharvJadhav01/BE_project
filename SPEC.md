# Multimodal Human State Capture Application — Specification

## 1. Project Overview

Build a desktop application that captures and synchronizes three types of human input:

1. **Voice / Audio**
2. **Keyboard / Keystroke activity**
3. **Facial expressions / Face-related visual features**

The application should collect these modalities continuously during a user session, process them independently, synchronize them using timestamps, and store the resulting data in a structured format for later analysis.

The project must be developed using **Spec-Driven Development (SDD)**.

This specification is the source of truth for the project.

---

# 2. Development Principles

The implementation MUST follow these principles:

* Follow this specification before writing application code.
* Do not introduce functionality that is not required by this specification without documenting it.
* Prefer existing, mature libraries over implementing functionality from scratch.
* Before implementing a component, check whether a suitable open-source or standard library already exists.
* Reuse existing libraries when they satisfy the requirements.
* Avoid unnecessary dependencies.
* Keep components modular and independently testable.
* Keep the application extensible so additional modalities can be added later.
* Do not mix data collection, processing, storage, and UI logic unnecessarily.
* All captured data must have reliable timestamps.
* The application must continue functioning if one modality temporarily becomes unavailable.

---

# 3. Existing Library / Framework First

Before implementing any functionality, inspect the ecosystem for suitable existing libraries.

The developer/agent should search for and evaluate libraries for:

### Audio

Possible areas:

* microphone capture
* audio streaming
* audio recording
* speech/audio feature extraction

### Keystrokes

Possible areas:

* keyboard event capture
* global/local keyboard listeners
* key press/release timestamps

### Facial Expressions

Possible areas:

* webcam capture
* face detection
* facial landmarks
* facial expression / emotion-related feature extraction

### Application

Possible areas:

* desktop application framework
* UI
* logging
* configuration
* database/storage
* asynchronous processing

### Data Processing

Possible areas:

* numerical processing
* signal processing
* computer vision
* audio processing

The implementation should prefer well-maintained existing libraries.

Do NOT recreate functionality that can reliably be provided by an appropriate existing library.

However, do not blindly install libraries.

For every important dependency:

1. Identify the library.
2. Determine whether it satisfies the requirement.
3. Check compatibility with the selected language/runtime.
4. Prefer actively maintained and widely used libraries.
5. Document the reason for selecting it.

---

# 4. Spec-Driven Development Workflow

The project must follow this workflow.

## Phase 1 — Specification Analysis

Before coding:

* Read this entire specification.
* Identify functional requirements.
* Identify non-functional requirements.
* Identify required components.
* Identify dependencies.
* Identify risks and platform-specific limitations.

Create:

`docs/REQUIREMENTS.md`

containing the interpreted requirements.

---

## Phase 2 — Architecture

Create:

`docs/ARCHITECTURE.md`

The architecture should define:

* application components
* data flow
* modality collectors
* processing pipeline
* synchronization mechanism
* storage layer
* UI
* configuration
* error handling
* logging

Do not start implementation until the architecture is consistent with this specification.

---

# 5. Application Architecture

Use a modular architecture.

The application should contain at least the following logical components:

```text
Application
│
├── UI
│
├── Session Manager
│
├── Voice Capture Module
│
├── Keystroke Capture Module
│
├── Facial Capture Module
│
├── Timestamp / Synchronization Module
│
├── Processing Pipeline
│
├── Storage Module
│
├── Configuration Module
│
└── Logging / Error Handling
```

Each module should have a clearly defined responsibility.

---

# 6. Session Management

The application operates using sessions.

A session represents one period during which the user is being monitored.

Each session must have:

* unique session ID
* start timestamp
* end timestamp
* configuration information
* enabled modalities
* application version

Example:

```text
session_id
start_time
end_time
enabled_modalities
application_version
```

The user must be able to:

* Start a session
* Stop a session
* Pause/resume capture where technically feasible
* View session status

---

# 7. Voice Capture

The application must be capable of capturing microphone audio.

## Requirements

The Voice Capture Module should:

* request microphone permission where required
* detect microphone availability
* start audio capture when the session begins
* stop audio capture when the session ends
* associate audio data with timestamps
* save audio in a standard format
* handle microphone errors gracefully

The implementation should support configurable:

* sample rate
* number of channels
* audio format
* recording duration/chunk size

The audio pipeline should preferably use an existing audio capture library.

Do not implement low-level microphone drivers.

---

# 8. Keystroke Capture

The application must capture keyboard activity.

The system should record keyboard events such as:

```text
key_down
key_up
```

Each event should contain at minimum:

```text
timestamp
event_type
key
```

Where possible, also record:

```text
duration
modifier_keys
application/window context
```

However, the system must NOT record sensitive information unnecessarily.

## Privacy Requirement

The application must not intentionally store plaintext passwords or sensitive credential information.

If the operating system or application environment provides a way to exclude sensitive input fields/windows, use it.

The user must clearly know that keyboard activity is being captured.

---

# 9. Facial Capture

The application must use a webcam to capture facial information.

The system should:

1. Detect webcam availability.
2. Request camera permission where required.
3. Capture frames during an active session.
4. Detect the user's face.
5. Extract facial information using an appropriate existing computer-vision library.
6. Generate timestamped facial observations.

The preferred processing pipeline is:

```text
Webcam
   ↓
Frame Capture
   ↓
Face Detection
   ↓
Face Landmark / Feature Extraction
   ↓
Facial Expression Feature Extraction
   ↓
Timestamped Observation
```

The application should prefer extracting useful facial features rather than unnecessarily storing every raw video frame.

The exact facial-expression model/library should be selected based on availability, licensing, compatibility, and performance.

---

# 10. Facial Expression Representation

Facial expression data should be represented as structured observations.

For example:

```text
timestamp
face_detected
landmarks/features
expression_features
confidence
```

If an emotion/expression classifier is used, its output must be treated as a model prediction rather than an objective measurement of the user's internal emotional state.

Do not claim that facial expressions definitively determine a person's mental state.

---

# 11. Timestamp Synchronization

This is a core requirement.

All modalities must use a common time reference.

For example:

```text
Voice
timestamp ────────┐
                  │
Keystroke         │
timestamp ────────┼──> Unified Timeline
                  │
Face              │
timestamp ────────┘
```

The system must be able to align:

* audio segments
* keystroke events
* facial observations

using timestamps.

Each event must contain a timestamp with sufficient precision for synchronization.

Prefer a monotonic clock for measuring elapsed session time and a wall-clock timestamp for session metadata where appropriate.

---

# 12. Data Model

The data should be organized around a session.

Example:

```text
data/
└── sessions/
    └── <session_id>/
        ├── session.json
        ├── audio/
        ├── keystrokes.jsonl
        ├── facial_features.jsonl
        └── events.jsonl
```

The exact storage implementation can be changed if a better design is justified.

The system must maintain clear relationships between:

```text
Session
    ↓
Timestamp
    ↓
Modality
    ↓
Observation/Event
```

---

# 13. Event Schema

Use a unified event structure where practical.

Example:

```json
{
  "session_id": "session_001",
  "timestamp": 12345.678,
  "modality": "keyboard",
  "event_type": "key_down",
  "data": {
    "key": "A"
  }
}
```

Possible modality values:

```text
voice
keyboard
face
system
```

Possible event types may include:

```text
audio_chunk
key_down
key_up
face_observation
session_start
session_end
error
```

The schema should remain extensible.

---

# 14. Processing Pipeline

The application should separate:

```text
Capture
   ↓
Validation
   ↓
Timestamping
   ↓
Processing
   ↓
Synchronization
   ↓
Storage
```

Capture modules should not directly implement unrelated business logic.

For example:

```text
VoiceCapture
     ↓
AudioProcessor
     ↓
TimestampedAudio
     ↓
Storage
```

Similarly:

```text
KeyboardCapture
     ↓
KeyboardProcessor
     ↓
TimestampedKeyboardEvent
     ↓
Storage
```

And:

```text
CameraCapture
     ↓
FaceProcessor
     ↓
TimestampedFacialObservation
     ↓
Storage
```

---

# 15. User Interface

The initial UI should be simple.

It should display:

### Application status

```text
● Ready
● Recording
● Paused
● Stopped
```

### Modality status

```text
Microphone: Connected
Camera: Connected
Keyboard: Active
```

### Session controls

```text
[ Start Session ]

[ Pause ]

[ Resume ]

[ Stop Session ]
```

### Session information

Display:

* session ID
* session duration
* enabled modalities
* capture status
* errors/warnings

The UI should not expose unnecessary raw sensitive data.

---

# 16. Configuration

Configuration should not be hard-coded.

Provide a configuration file such as:

```text
config/config.yaml
```

Configuration may contain:

```yaml
audio:
  enabled: true
  sample_rate: ...
  channels: ...

keyboard:
  enabled: true

camera:
  enabled: true
  fps: ...

storage:
  output_directory: ...

logging:
  level: INFO
```

The exact configuration values should depend on the selected implementation.

---

# 17. Error Handling

The application must gracefully handle:

### Microphone unavailable

The application should:

* report the error
* disable audio capture
* continue other enabled modalities if possible

### Camera unavailable

The application should:

* report the error
* disable facial capture
* continue other modalities if possible

### Keyboard capture unavailable

The application should:

* report the error
* continue other modalities if possible

### Processing failure

A processing error in one modality should not unnecessarily terminate the entire session.

Errors must be logged.

---

# 18. Logging

Implement structured logging.

Logs should contain:

```text
timestamp
severity
module
message
session_id
```

Example:

```text
INFO  SessionManager  Session started
INFO  AudioCapture    Microphone initialized
INFO  CameraCapture   Camera initialized
ERROR FaceProcessor   Face detection failed
```

Do not log sensitive keyboard content.

---

# 19. Privacy and Security

Privacy is a core requirement.

The application must:

* clearly indicate when recording is active
* request required permissions
* avoid unnecessary storage of raw data
* avoid storing passwords
* avoid collecting unrelated information
* provide a way to delete recorded sessions
* document what information is captured

If raw audio/video is stored, clearly document it.

If possible, prefer derived features for analysis when raw data is not required.

---

# 20. Performance Requirements

The application should operate continuously without causing excessive system load.

The implementation should consider:

* CPU usage
* RAM usage
* GPU availability
* camera processing FPS
* audio buffering
* asynchronous processing
* disk I/O

Use asynchronous/background processing where appropriate.

Do not block the UI thread with:

* audio processing
* camera processing
* facial analysis
* disk operations

---

# 21. Extensibility

The system must allow future modalities to be added.

For example:

```text
Voice
Keyboard
Face
   ↓
Future:
Mouse
Eye Tracking
EEG
Heart Rate
```

The architecture should allow a new modality to implement a common interface such as:

```text
start()
stop()
pause()
resume()
get_status()
```

The exact interface may be adapted to the selected language/framework.

---

# 22. Testing Requirements

Implement tests for:

### Unit tests

* session creation
* timestamp generation
* event creation
* configuration loading
* data serialization
* error handling

### Integration tests

* session lifecycle
* audio capture integration
* keyboard capture integration
* camera integration
* storage integration
* synchronization

### Failure tests

Test:

* camera disconnected
* microphone unavailable
* invalid configuration
* storage failure
* processing failure

---

# 23. Documentation

Create:

```text
README.md
docs/
    REQUIREMENTS.md
    ARCHITECTURE.md
    LIBRARIES.md
    DATA_SCHEMA.md
    PRIVACY.md
    DEVELOPMENT.md
```

`LIBRARIES.md` should explain:

```text
Library
Purpose
Version
Reason for selection
Alternative considered
License
```

---

# 24. Dependency Selection Rule

Before adding a dependency:

Ask:

> "Does an established library already provide this functionality?"

If yes:

* evaluate it
* use it when appropriate
* document it

If no:

* implement the minimum required functionality
* keep it modular
* document why a custom implementation was necessary

Do not add dependencies simply because they are popular.

---

# 25. Development Order

Implement in this order:

### Step 1

Project structure

### Step 2

Configuration system

### Step 3

Logging

### Step 4

Session manager

### Step 5

Timestamp/synchronization infrastructure

### Step 6

Keyboard capture

### Step 7

Audio capture

### Step 8

Camera capture

### Step 9

Facial feature/expression processing

### Step 10

Unified event pipeline

### Step 11

Storage

### Step 12

UI

### Step 13

Testing

### Step 14

Documentation

Do not attempt to build everything simultaneously.

---

# 26. Acceptance Criteria

The project is considered complete when:

* The application can start and stop a session.
* Microphone input can be captured.
* Keyboard events can be captured.
* Webcam frames can be captured.
* Facial features/expressions can be extracted using an appropriate library.
* All observations contain synchronized timestamps.
* The three modalities can be associated with a common session.
* Data can be stored and retrieved.
* One failed modality does not unnecessarily terminate the entire session.
* The UI displays capture status.
* Configuration is externalized.
* Errors are logged.
* Sensitive keyboard information is not intentionally stored.
* Tests exist for important components.
* Documentation explains installation, architecture, dependencies, data format, and privacy.

---

# 27. OpenCode Agent Instructions

When working on this project:

1. Read `SPEC.md` before making changes.
2. Treat `SPEC.md` as the primary source of truth.
3. Inspect the existing repository before creating files.
4. Do not overwrite existing functionality without understanding it.
5. Search for existing libraries before implementing functionality yourself.
6. Prefer stable, maintained libraries.
7. Keep dependencies minimal.
8. Implement one module at a time.
9. After implementing a module, test it before proceeding.
10. Update documentation when architecture or dependencies change.
11. Do not silently change requirements.
12. If a requirement is technically impossible or ambiguous, explain the issue and propose the smallest change required.
13. Do not fabricate library APIs or configuration options.
14. Verify imports, APIs, and dependency versions before using them.
15. Keep the application modular.
16. Preserve synchronization between modalities.
17. Never intentionally store passwords or other sensitive keyboard input.
18. Do not describe facial-expression predictions as definitive measurements of a person's internal emotional or mental state.
19. Before considering the project complete, run the available tests and verify the acceptance criteria.

---

# 28. Final Development Rule

Do not optimize for writing the maximum amount of code.

Optimize for:

```text
Correctness
    ↓
Reliability
    ↓
Privacy
    ↓
Maintainability
    ↓
Performance
    ↓
Extensibility
```

The goal is to build a clean, modular, spec-driven multimodal data-capture application rather than a collection of disconnected scripts.
