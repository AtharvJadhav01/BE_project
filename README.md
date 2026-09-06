# Multimodal Human State Capture Application

## Overview

A desktop application designed to capture and synchronize multiple human input modalities during a user session:

* Voice / audio
* Keyboard / keystroke activity
* Facial expressions and facial features

The collected data is timestamped and stored in a structured format for further analysis and research.

## Features

* 🎙️ Voice/audio capture
* ⌨️ Keystroke event capture
* 📷 Webcam capture
* 🙂 Facial feature and expression extraction
* 🕒 Timestamp synchronization across modalities
* 📊 Structured session-based data storage
* ⚙️ Configurable capture settings
* 🖥️ Desktop user interface
* 🧩 Modular architecture
* 🧪 Automated testing
* 🔒 Privacy-aware data handling

## Development Approach

This project follows **Spec-Driven Development (SDD)**.

The primary source of truth for the project is:

```text
SPEC.md
```

All implementation decisions should be consistent with the requirements defined in `SPEC.md`.

Before implementing a feature, the development process should:

1. Understand the requirement.
2. Check for existing suitable libraries.
3. Evaluate available libraries for compatibility and maintenance.
4. Select the most appropriate library when available.
5. Implement the feature according to the specification.
6. Test the implementation.
7. Update the relevant documentation.

## Project Structure

```text
your-project/
│
├── SPEC.md
├── README.md
│
├── config/
│   └── config.yaml
│
├── docs/
│   ├── REQUIREMENTS.md
│   ├── ARCHITECTURE.md
│   ├── LIBRARIES.md
│   ├── DATA_SCHEMA.md
│   ├── PRIVACY.md
│   └── DEVELOPMENT.md
│
├── src/
│   ├── main.py
│   ├── session/
│   ├── audio/
│   ├── keyboard/
│   ├── face/
│   ├── processing/
│   ├── synchronization/
│   ├── storage/
│   └── ui/
│
├── tests/
│
└── data/
    └── sessions/
```

## Data Flow

```text
Voice ───────────┐
                 │
Keyboard ────────┼──> Timestamping ──> Processing ──> Storage
                 │
Face ────────────┘
```

All modalities are associated with a common session and timestamp system so that events from different sources can be aligned on a unified timeline.

## Privacy

The application must clearly indicate when data capture is active.

The system should:

* Request required microphone and camera permissions.
* Avoid intentionally storing passwords or authentication credentials.
* Minimize unnecessary collection of personal information.
* Provide a mechanism to delete captured sessions.
* Clearly document what data is captured and stored.
* Prefer derived facial features when raw video is not required.
* Keep captured data local unless a future specification explicitly requires cloud processing.

Facial-expression predictions must be treated as model outputs and must not be presented as definitive measurements of a person's internal emotional or mental state.

## Installation

Installation instructions will be added after the technology stack and required libraries have been finalized.

## Running the Application

The application will be started through the project's configured entry point.

Example:

```bash
python src/main.py
```

The exact command may change based on the final technology stack.

## Configuration

Application settings are maintained in:

```text
config/config.yaml
```

Configuration may include:

* Audio settings
* Camera settings
* Keyboard capture settings
* Storage location
* Logging level
* Processing settings

## Testing

Tests are located in:

```text
tests/
```

Run the project's test suite using the test framework selected during implementation.

Testing should cover:

* Session management
* Audio capture
* Keyboard capture
* Camera/face processing
* Timestamp synchronization
* Data storage
* Error handling

## Future Scope

The architecture is designed to allow additional modalities to be added later, such as:

* Mouse activity
* Eye tracking
* Physiological signals
* EEG
* Other behavioral or biometric signals

New modalities should follow the same modular capture, timestamping, processing, and storage architecture.

## License

License information will be added when the project license is finalized.
