# Requirements

## Functional Requirements

### Session Management
| ID | Requirement |
|----|-------------|
| FR-SM-001 | The system shall allow the user to start a new capture session |
| FR-SM-002 | The system shall allow the user to stop a capture session |
| FR-SM-003 | The system shall allow the user to pause and resume capture where technically feasible |
| FR-SM-004 | The system shall assign a unique session ID to each session |
| FR-SM-005 | The system shall record session start timestamp, end timestamp, enabled modalities, and application version |
| FR-SM-006 | The system shall display session status (Ready, Recording, Paused, Stopped) |

### Voice Capture
| ID | Requirement |
|----|-------------|
| FR-VC-001 | The system shall capture microphone audio during an active session |
| FR-VC-002 | The system shall request microphone permission where required by the OS |
| FR-VC-003 | The system shall detect microphone availability before capture |
| FR-VC-004 | The system shall associate audio data with timestamps |
| FR-VC-005 | The system shall save audio in a standard format (e.g., WAV) |
| FR-VC-006 | The system shall support configurable sample rate, channels, audio format, and chunk size |
| FR-VC-007 | The system shall handle microphone errors gracefully (report error, disable audio, continue other modalities) |

### Keystroke Capture
| ID | Requirement |
|----|-------------|
| FR-KC-001 | The system shall capture keyboard events (key_down, key_up) during an active session |
| FR-KC-002 | Each keyboard event shall contain: timestamp, event_type, key |
| FR-KC-003 | Each keyboard event should contain where possible: duration, modifier keys, application/window context |
| FR-KC-004 | The system shall NOT intentionally store plaintext passwords or sensitive credential information |
| FR-KC-005 | The system shall use OS-provided mechanisms to exclude sensitive input fields/windows where available |
| FR-KC-006 | The system shall clearly indicate to the user that keyboard activity is being captured |

### Facial Capture
| ID | Requirement |
|----|-------------|
| FR-FC-001 | The system shall detect webcam availability |
| FR-FC-002 | The system shall request camera permission where required by the OS |
| FR-FC-003 | The system shall capture frames during an active session |
| FR-FC-004 | The system shall detect the user's face in captured frames |
| FR-FC-005 | The system shall extract facial landmarks/features using a computer-vision library |
| FR-FC-006 | The system shall generate timestamped facial observations |
| FR-FC-007 | The system shall prefer extracting derived facial features over storing raw video frames |
| FR-FC-008 | The system shall handle camera errors gracefully (report error, disable facial capture, continue other modalities) |

### Timestamp Synchronization
| ID | Requirement |
|----|-------------|
| FR-TS-001 | All modalities shall use a common time reference |
| FR-TS-002 | The system shall align audio segments, keystroke events, and facial observations using timestamps |
| FR-TS-003 | Each event shall contain a timestamp with sufficient precision for synchronization (millisecond or better) |
| FR-TS-004 | The system shall use a monotonic clock for elapsed session time and wall-clock timestamp for session metadata |

### Data Storage
| ID | Requirement |
|----|-------------|
| FR-DS-001 | Data shall be organized around sessions with structure: `data/sessions/<session_id>/` |
| FR-DS-002 | Each session shall contain: session.json, audio/, keystrokes.jsonl, facial_features.jsonl, events.jsonl |
| FR-DS-003 | The system shall maintain clear relationships: Session → Timestamp → Modality → Observation/Event |
| FR-DS-004 | The system shall use a unified event structure where practical |
| FR-DS-005 | The schema shall remain extensible for future modalities |

### Processing Pipeline
| ID | Requirement |
|----|-------------|
| FR-PP-001 | The system shall separate: Capture → Validation → Timestamping → Processing → Synchronization → Storage |
| FR-PP-002 | Voice capture flow: VoiceCapture → AudioProcessor → TimestampedAudio → Storage |
| FR-PP-003 | Keyboard capture flow: KeyboardCapture → KeyboardProcessor → TimestampedKeyboardEvent → Storage |
| FR-PP-004 | Facial capture flow: CameraCapture → FaceProcessor → TimestampedFacialObservation → Storage |

### User Interface
| ID | Requirement |
|----|-------------|
| FR-UI-001 | The UI shall display application status (Ready, Recording, Paused, Stopped) |
| FR-UI-002 | The UI shall display modality status (Microphone: Connected/Disconnected, Camera: Connected/Disconnected, Keyboard: Active/Inactive) |
| FR-UI-003 | The UI shall provide session controls: Start Session, Pause, Resume, Stop Session |
| FR-UI-004 | The UI shall display session information: session ID, session duration, enabled modalities, capture status, errors/warnings |
| FR-UI-005 | The UI shall not expose unnecessary raw sensitive data |

### Configuration
| ID | Requirement |
|----|-------------|
| FR-CF-001 | Configuration shall not be hard-coded |
| FR-CF-002 | Configuration shall be provided via a config file (config/config.yaml) |
| FR-CF-003 | Configuration shall include: audio settings, keyboard settings, camera settings, storage location, logging level |

### Error Handling
| ID | Requirement |
|----|-------------|
| FR-EH-001 | The system shall gracefully handle microphone unavailable |
| FR-EH-002 | The system shall gracefully handle camera unavailable |
| FR-EH-003 | The system shall gracefully handle keyboard capture unavailable |
| FR-EH-004 | A processing error in one modality shall not terminate the entire session |
| FR-EH-005 | Errors must be logged |

### Logging
| ID | Requirement |
|----|-------------|
| FR-LG-001 | The system shall implement structured logging |
| FR-LG-002 | Logs shall contain: timestamp, severity, module, message, session_id |
| FR-LG-003 | The system shall not log sensitive keyboard content |

### Privacy and Security
| ID | Requirement |
|----|-------------|
| FR-PR-001 | The application shall clearly indicate when recording is active |
| FR-PR-002 | The application shall request required permissions |
| FR-PR-003 | The application shall avoid unnecessary storage of raw data |
| FR-PR-004 | The application shall avoid storing passwords |
| FR-PR-005 | The application shall avoid collecting unrelated information |
| FR-PR-006 | The application shall provide a way to delete recorded sessions |
| FR-PR-007 | The application shall document what information is captured |
| FR-PR-008 | Facial-expression predictions must be treated as model outputs, not definitive measurements |

### Extensibility
| ID | Requirement |
|----|-------------|
| FR-EX-001 | The architecture shall allow future modalities to be added (Mouse, Eye Tracking, EEG, Heart Rate, etc.) |
| FR-EX-002 | New modalities shall implement a common interface: start(), stop(), pause(), resume(), get_status() |

---

## Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-001 | The application shall operate continuously without causing excessive system load |
| NFR-002 | The application shall consider CPU usage, RAM usage, GPU availability |
| NFR-003 | The application shall maintain camera processing FPS appropriate for facial analysis |
| NFR-004 | The application shall use asynchronous/background processing for audio, camera, facial analysis, and disk operations |
| NFR-005 | The application shall not block the UI thread with processing tasks |
| NFR-006 | The application shall be modular with independently testable components |
| NFR-007 | The application shall prefer existing, mature libraries over custom implementations |
| NFR-008 | The application shall keep dependencies minimal |
| NFR-009 | The application shall implement unit tests for: session creation, timestamp generation, event creation, configuration loading, data serialization, error handling |
| NFR-010 | The application shall implement integration tests for: session lifecycle, audio capture, keyboard capture, camera integration, storage, synchronization |
| NFR-011 | The application shall implement failure tests for: camera disconnected, microphone unavailable, invalid configuration, storage failure, processing failure |
| NFR-012 | Documentation shall explain installation, architecture, dependencies, data format, and privacy |