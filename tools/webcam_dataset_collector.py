"""
Webcam Expression Dataset Collector

Records labeled facial expression samples from webcam for real-world validation.
Uses the project's existing camera infrastructure (OpenCV + MediaPipe).

Ground-truth labels come from manual user selection, NOT model predictions.
"""

import os
import cv2
import json
import uuid
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.timestamps import TimestampGenerator


# Expression classes (matching FER2013)
EXPRESSION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprised",
    "neutral",
]


class WebcamDatasetCollector:
    """Collects labeled facial expression samples from webcam."""

    def __init__(
        self,
        output_dir: str = "data/webcam_expression_dataset",
        device: int = 0,
        fps: int = 15,
        resolution: tuple = (640, 480),
    ):
        self.output_dir = Path(output_dir)
        self.device = device
        self.fps = fps
        self.resolution = resolution
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.timestamp_generator = TimestampGenerator()
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_label: Optional[str] = None
        self.recording = False
        self.frame_count = 0
        self.samples_saved = 0
        self.metadata_file = self.output_dir / "metadata.jsonl"
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

        # Create class directories
        for cls in EXPRESSION_CLASSES:
            (self.output_dir / cls).mkdir(parents=True, exist_ok=True)

    def _init_camera(self) -> bool:
        """Initialize webcam capture."""
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            print(f"Error: Cannot open camera device {self.device}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
        return True

    def _release_camera(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def _save_frame(self, frame):
        """Save a single frame with metadata."""
        if self.current_label is None:
            return False

        # Generate unique filename
        timestamp = datetime.now(timezone.utc)
        timestamp_str = timestamp.isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        filename = f"{self.current_label}_{self.session_id}_{self.frame_count:06d}.jpg"
        filepath = self.output_dir / self.current_label / filename

        # Save image
        success = cv2.imwrite(str(filepath), frame)
        if not success:
            print(f"Warning: Failed to save {filepath}")
            return False

        # Save metadata
        metadata = {
            "sample_id": f"{self.current_label}_{self.session_id}_{self.frame_count:06d}",
            "class": self.current_label,
            "timestamp": timestamp_str,
            "timestamp_monotonic": self.timestamp_generator.now_monotonic(),
            "session_id": self.session_id,
            "image_path": str(filepath.relative_to(self.output_dir)),
            "camera_resolution": [actual_width, actual_height] if self.cap else [self.resolution[0], self.resolution[1]],
            "frame_index": self.frame_count,
        }

        with open(self.metadata_file, "a") as f:
            f.write(json.dumps(metadata) + "\n")

        self.samples_saved += 1
        return True

    def _draw_ui(self, frame):
        """Draw UI overlay on frame."""
        h, w = frame.shape[:2]

        # Status bar at top
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

        # Session info
        cv2.putText(
            frame,
            f"Session: {self.session_id}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Current target expression
        label_color = (0, 255, 0) if self.current_label else (100, 100, 100)
        label_text = f"TARGET: {self.current_label.upper()}" if self.current_label else "TARGET: (not selected)"
        cv2.putText(
            frame,
            label_text,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            label_color,
            2,
        )

        # Recording status
        if self.recording:
            cv2.putText(
                frame,
                "RECORDING ●",
                (w - 200, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "IDLE",
                (w - 100, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 100, 100),
                1,
            )

        # Sample count
        cv2.putText(
            frame,
            f"Samples saved: {self.samples_saved}",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Key help
        help_text = "Keys: 1-7=Select expression | SPACE=Start/Stop recording | Q=Quit"
        cv2.putText(
            frame,
            help_text,
            (10, h - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1,
        )

        # Expression key mapping
        for i, cls in enumerate(EXPRESSION_CLASSES):
            y = h - 70 + (i // 4) * 20
            x = 10 + (i % 4) * 155
            color = (0, 255, 0) if cls == self.current_label else (150, 150, 150)
            cv2.putText(
                frame,
                f"{i+1}: {cls}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        return frame

    def run(self):
        """Main collection loop."""
        if not self._init_camera():
            return

        print("\n=== Webcam Expression Dataset Collector ===")
        print("Keys:")
        for i, cls in enumerate(EXPRESSION_CLASSES):
            print(f"  {i+1}: {cls}")
        print("  SPACE: Start/Stop recording")
        print("  Q: Quit")
        print("\nInstructions:")
        print("  1. Select target expression with keys 1-7")
        print("  2. Press SPACE to start recording")
        print("  3. Perform the expression naturally")
        print("  4. Press SPACE to stop recording")
        print("  4. Vary head pose, distance, lighting between recordings")
        print("  5. Press Q to quit\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Warning: Failed to read frame")
                    time.sleep(0.01)
                    continue

                self.frame_count += 1

                # Draw UI
                display_frame = self._draw_ui(frame.copy())

                # Save frame if recording
                if self.recording and self.current_label:
                    self._save_frame(frame)

                cv2.imshow("Webcam Expression Collector", display_frame)

                key = cv2.waitKey(1) & 0xFF

                # Number keys 1-7 for expression selection
                if ord('1') <= key <= ord('7'):
                    self.current_label = EXPRESSION_CLASSES[key - ord('1')]
                    print(f"Target expression set to: {self.current_label}")

                # SPACE to toggle recording
                elif key == ord(' '):
                    if self.current_label is None:
                        print("Select a target expression first (keys 1-7)")
                    else:
                        self.recording = not self.recording
                        if self.recording:
                            print(f"Started recording: {self.current_label}")
                        else:
                            print(f"Stopped recording: {self.current_label} ({self.samples_saved} total samples)")

                # Q to quit
                elif key == ord('q') or key == ord('Q'):
                    break

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self._release_camera()
            cv2.destroyAllWindows()
            print(f"\nSession complete. Total samples saved: {self.samples_saved}")
            print(f"Metadata saved to: {self.metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Collect labeled webcam expression samples")
    parser.add_argument(
        "--output",
        default="data/webcam_expression_dataset",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Camera device index",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Target FPS",
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        default=[640, 480],
        help="Camera resolution (width height)",
    )
    args = parser.parse_args()

    collector = WebcamDatasetCollector(
        output_dir=args.output,
        device=args.device,
        fps=args.fps,
        resolution=tuple(args.resolution),
    )
    collector.run()


if __name__ == "__main__":
    main()