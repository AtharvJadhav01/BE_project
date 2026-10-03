# Privacy and Security

## Privacy Principles

This application is designed with privacy as a core requirement, not an afterthought. The following principles guide all design and implementation decisions:

1. **Transparency**: User must always know when capture is active
2. **Minimization**: Collect only what is necessary for the stated purpose
3. **Local-First**: All data stays on the user's device unless explicitly configured otherwise
4. **Control**: User can start, stop, pause, and delete sessions at any time
5. **No Surprises**: No hidden data collection, no telemetry without consent

---

## Data Captured

### Voice / Audio
| Data | Stored | Purpose |
|------|--------|---------|
| Raw audio (WAV chunks) | Yes (configurable) | Acoustic analysis, speech research |
| Audio metadata (timestamps, RMS) | Yes | Synchronization, voice activity detection |
| Derived features (spectrograms, MFCCs) | No (computed on demand) | Future analysis |

**Privacy Controls:**
- Microphone permission requested explicitly
- Visual indicator when recording (UI + system tray)
- Configurable: can disable audio entirely
- No speech-to-text by default (no transcription stored)

### Keyboard / Keystrokes
| Data | Stored | Purpose |
|------|--------|---------|
| Key down/up events | Yes | Typing dynamics, interaction patterns |
| Key identities (A, Enter, Shift, etc.) | Yes (filtered) | Key-level timing analysis |
| Modifier keys (Ctrl, Alt, Shift) | Yes | Shortcut/command analysis |
| Key durations | Yes | Hold time analysis |
| Window/process context | Yes (redacted) | Application context |
| Text entered in the dedicated keyboard analysis field | Yes, in `typed_text.txt` during a session | Analyze the user's entered text and typing behavior |

**Privacy Controls - MANDATORY:**
- **Password/sensitive field detection**: OS accessibility APIs used to detect password fields; keys in these contexts are REDACTED (`"[REDACTED]"`)
- Window titles containing sensitive terms are REDACTED
- Keyboard data outside the dedicated text field is redacted in detected sensitive contexts. Text entered in the dedicated analysis field is intentionally saved as plaintext; users must not enter passwords or other sensitive information there.
- User notified that keyboard is being captured (UI indicator)
- Dedicated analysis field clearly indicates that its text is saved locally; users must not enter passwords or sensitive information there
- Configurable: can disable keyboard capture entirely

**Sensitive Context Detection:**
- Active window title patterns: `password`, `login`, `signin`, `credential`, `secret`, `token`, `pin`, `cvv`, `ssn`, `credit`, `card`, `auth`, `2fa`, `mfa`
- Password field detection via OS accessibility (Windows UIA, macOS AX, Linux AT-SPI)
- When detected: `key` → `"[REDACTED]"`, `window_title` → `"[REDACTED]"`, `is_sensitive_context: true`

### Facial / Webcam
| Data | Stored | Purpose |
|------|--------|---------|
| Raw video frames | No (by default) | Not stored - processed in memory only |
| Face detection bounding boxes | Yes | Face presence, position |
| Facial landmarks (468 points) | Yes | Geometry, pose |
| Blendshapes (52 expression coefficients) | Yes | Expression dynamics |
| Iris positions | Yes | Gaze direction |
| Head pose (pitch, yaw, roll) | Yes | Attention, orientation |

**Privacy Controls:**
- Camera permission requested explicitly
- Visual indicator when camera active (UI + LED where available)
- **Raw frames NOT stored by default** - only derived features
- Configurable: can disable facial capture entirely
- Blendshapes are model outputs, NOT emotional state measurements

**Important Disclaimer:**
> Facial expression predictions (blendshapes, action units, emotion classifiers) are **model outputs** based on statistical patterns in training data. They do NOT definitively measure a person's internal emotional or mental state. Results should be interpreted as "facial configuration consistent with X" not "person is feeling X".

---

## Data Not Captured

The following are explicitly NOT captured:
- Clipboard contents
- Screen content / screenshots
- Mouse movements (unless future modality added)
- Network activity
- File system access
- Browser history
- Application content, except text deliberately entered in the dedicated keyboard analysis field during a session
- Audio from other applications (only microphone input)
- Video from other sources (only webcam)

---

## Data Storage

### Location
- Default: `./data/sessions/<session_id>/`
- User-configurable via `config.yaml`
- No cloud sync, no network transmission

### Format
- Open, documented formats (JSONL, WAV, JSON)
- No proprietary encryption (user controls filesystem encryption)
- Session metadata includes config snapshot for reproducibility

### Retention
- Sessions persist until user deletes them
- No automatic expiration
- User can delete individual sessions via UI or file system

### Deletion
- UI provides "Delete Session" action
- Removes entire session directory
- No recovery (unless filesystem backup exists)

---

## Permissions

### Required OS Permissions
| Platform | Microphone | Camera | Keyboard (Global) |
|----------|------------|--------|-------------------|
| Windows 10/11 | Privacy Settings → Microphone | Privacy Settings → Camera | Low-level hook (no admin) |
| macOS | System Settings → Privacy → Microphone | System Settings → Privacy → Camera | Accessibility permission |
| Linux | PulseAudio/PipeWire | V4L2 / PipeWire | X11 (XRecord), not Wayland |

### Permission Handling
- Application requests permissions on first use
- Graceful degradation if denied (modality disabled, session continues)
- Clear error messages guiding user to settings
- No repeated prompts after denial

---

## Security Considerations

### Data at Rest
- Files stored with OS default permissions (user-readable)
- Recommend: Full-disk encryption (BitLocker, FileVault, LUKS)
- No application-level encryption (keys would need storage)

### Data in Transit
- No network communication in base application
- All processing local
- Future network features would require explicit user consent and TLS

### Memory Handling
- Raw audio frames: Processed in buffers, released after chunk write
- Raw video frames: Processed in memory, NOT stored, released after feature extraction
- Keyboard events: Written to disk immediately (JSONL append)
- No persistent in-memory caches of sensitive data

### Process Isolation
- Each modality runs in separate thread
- No shared memory between modalities except event bus
- Event bus passes only timestamped, structured data

---

## User Consent & Control

### Session Start
- Explicit "Start Session" button required
- Modalities shown with status before start
- Configuration review available

### During Session
- Real-time status indicators for each modality
- Pause/Resume available
- Stop ends session and flushes data

### After Session
- Session list with metadata
- Per-session delete action
- Export option (future)

### Configuration
- All modalities independently enable/disable
- Granular settings per modality
- Config file human-readable YAML

---

## Compliance Considerations

### GDPR (EU)
- Personal data: Yes (biometric: face, voice; behavioral: keystrokes)
- Lawful basis: Consent (explicit session start)
- Data subject rights: Access (export), Deletion (delete session), Portability (JSONL)
- DPIA recommended for production deployment
- Data Processing Agreement not needed (local-only)

### CCPA (California)
- Personal information: Yes
- Sale: No (no data leaves device)
- Opt-out: Disable modalities or don't use
- Deletion: Delete session action

### HIPAA (Healthcare)
- Not designed for HIPAA compliance
- No BAA offered
- Do not use for clinical data without separate assessment

### Biometric Data Laws (BIPA Illinois, etc.)
- Facial geometry = biometric identifier
- Written consent required before capture
- Retention policy needed
- This application: Consent via explicit session start, retention until user deletion

---

## Audit Checklist for Release

- [ ] Microphone permission requested before capture
- [ ] Camera permission requested before capture
- [ ] Keyboard capture disclosed in UI
- [ ] Password field detection implemented and tested
- [ ] Sensitive window title patterns configured
- [ ] Raw video frames NOT written to disk
- [ ] Raw audio configurable (default: stored as WAV)
- [ ] Session delete removes all modality data
- [ ] Config allows disabling any modality
- [ ] No network requests in codebase
- [ ] No telemetry/analytics
- [ ] License headers on all files
- [ ] Privacy policy document references this spec

---

## Threat Model

| Threat | Mitigation |
|--------|------------|
| Malware reads session data | OS file permissions, disk encryption |
| Shoulder surfing UI | Minimal UI, no raw data display |
| Unauthorized session start | Explicit button, no auto-start |
| Modality re-enabled silently | Config persists, UI shows status |
| Data exfiltration | No network code, audit dependencies |
| Keylogger misuse | Open source, visible indicators, no stealth mode |

---

## Privacy by Design Summary

| Principle | Implementation |
|-----------|----------------|
| Proactive not Reactive | Privacy controls built into architecture |
| Privacy as Default | Modalities opt-in via config, explicit start |
| Privacy Embedded | Modular design isolates modalities |
| Full Functionality | All features work with any modality subset |
| End-to-End Security | Local-only, no network, user controls files |
| Visibility & Transparency | Real-time indicators, documented data schema |
| Respect for User | User controls start/stop/delete, no dark patterns |

---

## Contact

For privacy questions or concerns about this application, refer to the project documentation or contact the maintainers.