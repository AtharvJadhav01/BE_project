# Webcam Expression Dataset Collection Protocol

## Purpose

This document describes the procedure for collecting a labeled facial expression dataset from real webcam video. The dataset will be used to evaluate the facial expression recognition system's real-world performance (domain shift from FER2013 to webcam images).

## Expression Classes

Collect samples for all seven basic emotions (matching FER2013):

| Class | Description |
|-------|-------------|
| angry | Furrowed brows, narrowed eyes, pressed lips |
| disgust | Wrinkled nose, raised upper lip |
| fear | Wide eyes, raised brows, open mouth |
| happy | Smile, raised cheeks, crow's feet |
| sad | Drooping mouth, raised inner brows |
| surprised | Wide eyes, raised brows, open mouth |
| neutral | Relaxed face, no expression |

## Recording Procedure

### Setup
1. Position webcam at eye level, ~60 cm from face
2. Ensure adequate lighting (avoid strong backlighting)
3. Run the collector tool:
   ```bash
   python tools/webcam_dataset_collector.py
   ```

### Per-Expression Recording
For EACH expression class:

1. **Select target expression** (press key 1-7)
2. **Start recording** (press SPACE)
3. **Perform the expression naturally** for 3-5 seconds
   - Vary head pose slightly (left, right, up, down ~15°)
   - Vary distance slightly (50-80 cm)
   - Vary expression intensity (subtle to pronounced)
4. **Stop recording** (press SPACE)
5. **Repeat** for 3-5 separate recording sessions per expression
   - Do NOT record one long continuous session
   - Take breaks between sessions
   - Change lighting/position between sessions

### Variation Guidelines
- **Head pose**: Record with slight yaw (±15°), pitch (±10°), roll (±5°)
- **Distance**: 50 cm, 65 cm, 80 cm
- **Lighting**: Front lighting, side lighting, diffuse lighting
- **Expression intensity**: Subtle, moderate, strong
- **Background**: Plain, cluttered, different colors

### Quality Checks During Recording
- Ensure face is detected (green box in UI)
- Avoid covering face (hands, hair, masks)
- Avoid extreme angles (>30° yaw/pitch)
- Ensure expression is recognizable to human observer

## Recommended Sample Counts

| Expression | Minimum | Recommended |
|------------|---------|-------------|
| angry      | 30      | 50+         |
| disgust    | 30      | 50+         |
| fear       | 30      | 50+         |
| happy      | 50      | 80+         |
| sad        | 30      | 50+         |
| surprised  | 30      | 50+         |
| neutral    | 50      | 80+         |
| **Total**  | **250** | **400+**    |

Higher counts for happy/neutral as they are typically easier to collect.

## Session Separation

**Critical**: Do NOT randomly split frames from the same recording session.

### If Multiple Participants Available:
- Split by PARTICIPANT
- Training: Participant A + B
- Validation: Participant C
- Test: Participant D

### If Single Participant:
- Split by RECORDING SESSION
- Each press of SPACE (start/stop) = one session
- Never mix frames from same session across splits

Example split for single participant:
- Training: Sessions 1, 2, 4, 5, 7, 8
- Validation: Session 3
- Test: Session 6, 9

## Labeling Procedure

**Ground truth labels come from MANUAL SELECTION during recording, NOT model predictions.**

1. User presses key 1-7 to select target expression
2. User presses SPACE to start recording
3. All frames recorded during that session get that label
4. Metadata records: session_id, target_expression, timestamp, frame_index

Do NOT use model predictions to assign or verify labels.

## Data Storage Structure

```
data/webcam_expression_dataset/
├── angry/
│   ├── angry_session_abc123_000001.jpg
│   └── ...
├── disgust/
├── fear/
├── happy/
├── sad/
├── surprised/
├── neutral/
└── metadata.jsonl
```

### Metadata Format (JSONL, one line per sample)

```json
{
  "sample_id": "happy_session_abc123_000001",
  "class": "happy",
  "timestamp": "2026-09-22T15:30:45.123Z",
  "timestamp_monotonic": 123456.789,
  "session_id": "session_20260922_153045_abc12345",
  "image_path": "happy/happy_session_abc123_000001.jpg",
  "camera_resolution": [640, 480],
  "frame_index": 1
}
```

## Privacy Considerations

- **No personal identifiers**: Do not store names, emails, or IDs
- **Session IDs are random**: Generated as `session_YYYYMMDD_HHMMSS_xxxxxxxx`
- **Raw video NOT stored**: Only individual frames saved as JPG
- **Local storage only**: Data remains on collection machine
- **Deletion policy**: Document when/how data will be deleted after evaluation
- **Consent**: If collecting from multiple participants, document informed consent

## Validation

After collection, run the validator:

```bash
python tools/validate_webcam_dataset.py --dataset data/webcam_expression_dataset
```

This checks:
- Corrupted images
- Duplicate files
- Image dimensions
- Face detection success rate
- Class distribution balance
- Metadata completeness

## Next Steps

After dataset collection and validation:

1. Run evaluation: `python training/evaluate_webcam.py`
2. Compare webcam accuracy vs FER2013 (49.14%)
3. Analyze error patterns
4. Document findings in `docs/real_world_face_validation.md`

---

**Note**: This protocol is designed to create a realistic, challenging test set that reveals domain shift between FER2013 (grayscale, 48×48, centered faces) and real webcam images (color, variable resolution, natural variation).