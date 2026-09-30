"""
Keyboard Analysis Panel for displaying real-time typing behavior analysis.
"""

from typing import Optional, Dict
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QLineEdit
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from analysis.keyboard.features import KeyboardFeatureExtractor


class _TypingInput(QLineEdit):
    """Text field that reports key timing without exposing entered text."""

    def __init__(self, on_key_input, parent=None):
        super().__init__(parent)
        self._on_key_input = on_key_input

    def keyPressEvent(self, event) -> None:
        super().keyPressEvent(event)

        if event.isAutoRepeat():
            return

        modifiers = event.modifiers()
        if modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier):
            return

        if event.key() == Qt.Key.Key_Backspace:
            self._on_key_input("Backspace", False)
        elif event.key() == Qt.Key.Key_Delete:
            self._on_key_input("Delete", False)
        elif event.text() and event.text().isprintable():
            self._on_key_input(event.text(), True)


class KeyboardAnalysisPanel(QWidget):
    """Panel displaying real-time keyboard/typing analysis information."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Title
        title = QLabel("Keyboard Analysis")
        title.setFont(QFont("Segoe UI", 11, 600))  # Bold
        title.setStyleSheet("color: #0078d4;")
        self.layout().addWidget(title)

        # Keyboard detected status
        self.keyboard_detected_label = QLabel("Keyboard: Not Active")
        self.keyboard_detected_label.setFont(QFont("Segoe UI", 9))
        self.keyboard_detected_label.setStyleSheet("color: #cc0000;")
        self.layout().addWidget(self.keyboard_detected_label)

        # Current typing state
        self.state_label = QLabel("State: --")
        self.state_label.setFont(QFont("Segoe UI", 10, 600))  # Bold
        self.layout().addWidget(self.state_label)

        # Sample progress
        self.sample_progress_label = QLabel("Sample Progress: 0%")
        self.sample_progress_label.setFont(QFont("Segoe UI", 9))
        self.layout().addWidget(self.sample_progress_label)

        # Baseline status
        self.baseline_label = QLabel("Baseline: Warming Up")
        self.baseline_label.setFont(QFont("Segoe UI", 9))
        self.baseline_label.setStyleSheet("color: #ffaa00;")
        self.layout().addWidget(self.baseline_label)

        self.input_status_label = QLabel("Start a session to record text and analyze your typing.")
        self.input_status_label.setWordWrap(True)
        self.input_status_label.setFont(QFont("Segoe UI", 8))
        self.layout().addWidget(self.input_status_label)

        self._feature_extractor = KeyboardFeatureExtractor("keyboard_sample", {})
        self.input_field = _TypingInput(self._on_sample_key, self)
        self.input_field.setPlaceholderText("Start a session to enter text")
        self.input_field.setEnabled(False)
        self.input_field.textChanged.connect(self._persist_session_text)
        self.layout().addWidget(self.input_field)
        self._session_text_path: Optional[Path] = None
        self._session_text_recording_active = False

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        sep.setStyleSheet("color: #dddddd;")
        self.layout().addWidget(sep)

        # Real-time metrics
        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(4)

        # Typing speed
        self.typing_speed_label = QLabel("Typing Speed: --")
        self.typing_speed_label.setFont(QFont("Segoe UI", 8))
        self.layout().addWidget(self.typing_speed_label)

        # Inter-key interval
        self.iki_label = QLabel("Inter-Key Interval: --")
        self.iki_label.setFont(QFont("Segoe UI", 8))
        self.layout().addWidget(self.iki_label)

        # Correction rate
        self.correction_rate_label = QLabel("Correction Rate: --")
        self.correction_rate_label.setFont(QFont("Segoe UI", 8))
        self.layout().addWidget(self.correction_rate_label)

        # Pause duration
        self.pause_duration_label = QLabel("Pause Duration: --")
        self.pause_duration_label.setFont(QFont("Segoe UI", 8))
        self.layout().addWidget(self.pause_duration_label)

        # Rhythm regularity
        self.rhythm_label = QLabel("Rhythm Regularity: --")
        self.rhythm_label.setFont(QFont("Segoe UI", 8))
        self.rhythm_label.setToolTip("0 to 1; higher values mean more consistent inter-key timing.")
        self.layout().addWidget(self.rhythm_label)

        # Hesitation score
        self.hesitation_label = QLabel("Hesitation: --")
        self.hesitation_label.setFont(QFont("Segoe UI", 8))
        self.layout().addWidget(self.hesitation_label)

        self.layout().addLayout(metrics_layout)
        self.layout().addStretch()

    def _on_sample_key(self, key_name: str, is_printable: bool) -> None:
        result = self._feature_extractor.extract({
            "event_type": "key_down",
            "key": key_name,
            "is_printable": is_printable,
            "modifiers": [],
            "is_sensitive_context": False,
        })
        if result is None:
            return

        metrics = result.features
        key_count = int(metrics.get("total_keys", 0))
        self.update_keyboard_detected(True)
        self.update_metrics(metrics)

        enough_data = key_count >= 4
        state = "typing_pattern_observed" if enough_data else "collecting_sample"
        progress = min(1.0, key_count / 20)
        self.update_state(state)
        self.update_sample_progress(progress)
        if progress >= 1.0:
            self.baseline_label.setText("Baseline: Sample ready (not personalized)")
            self.baseline_label.setStyleSheet("color: #0078d4;")
        else:
            self.baseline_label.setText(f"Baseline: Collecting sample ({key_count}/20 keys)")
            self.baseline_label.setStyleSheet("color: #ffaa00;")

    def _reset_sample_analysis(self) -> None:
        self._feature_extractor.reset()
        self.update_keyboard_detected(False)
        self.update_state("--")
        self.update_sample_progress(0.0)
        self.update_baseline(False)
        self.typing_speed_label.setText("Typing Speed: --")
        self.iki_label.setText("Inter-Key Interval: --")
        self.correction_rate_label.setText("Correction Rate: --")
        self.pause_duration_label.setText("Pause Duration: --")
        self.rhythm_label.setText("Rhythm: --")
        self.hesitation_label.setText("Hesitation: --")

    @property
    def session_text_recording_active(self) -> bool:
        return self._session_text_recording_active

    def begin_session_text_recording(self, output_path: Path) -> None:
        self._session_text_path = output_path
        self.input_field.clear()
        self._reset_sample_analysis()
        self._session_text_recording_active = True
        self.input_field.setEnabled(True)
        self.input_status_label.setText(
            "Session text is being saved locally. Do not enter passwords or sensitive information."
        )
        self._persist_session_text()

    def set_session_text_recording_active(self, active: bool) -> None:
        if self._session_text_path is None:
            return
        self._session_text_recording_active = active
        self.input_field.setEnabled(active)
        if active:
            self.input_status_label.setText(
                "Session text is being saved locally. Do not enter passwords or sensitive information."
            )
        else:
            self.input_status_label.setText("Session paused. Text entry is temporarily disabled.")

    def finish_session_text_recording(self) -> None:
        if self._session_text_path is None:
            return
        self._persist_session_text()
        self._session_text_path = None
        self._session_text_recording_active = False
        self.input_field.setEnabled(False)
        self.input_status_label.setText("Session text saved to typed_text.txt.")

    def _persist_session_text(self, _text: Optional[str] = None) -> None:
        if self._session_text_path is None:
            return
        try:
            self._session_text_path.parent.mkdir(parents=True, exist_ok=True)
            self._session_text_path.write_text(self.input_field.text(), encoding="utf-8")
        except OSError as error:
            self.input_status_label.setText(f"Could not save session text: {error}")

    def update_keyboard_detected(self, detected: bool) -> None:
        """Update keyboard detected status."""
        if detected:
            self.keyboard_detected_label.setText("Keyboard: Active")
            self.keyboard_detected_label.setStyleSheet("color: #00aa00;")
        else:
            self.keyboard_detected_label.setText("Keyboard: Not Active")
            self.keyboard_detected_label.setStyleSheet("color: #cc0000;")

    def update_state(self, state: str) -> None:
        """Update the observed typing state."""
        self.state_label.setText(f"State: {state}")

    def update_sample_progress(self, progress: float) -> None:
        self.sample_progress_label.setText(f"Sample Progress: {progress:.0%}")

    def update_baseline(self, ready: bool, progress: float = 0.0) -> None:
        """Update baseline status."""
        if ready:
            self.baseline_label.setText("Baseline: Ready")
            self.baseline_label.setStyleSheet("color: #00aa00;")
        else:
            self.baseline_label.setText(f"Baseline: Warming Up ({progress:.0%})")
            self.baseline_label.setStyleSheet("color: #ffaa00;")

    def update_metrics(self, metrics: dict) -> None:
        """Update real-time keyboard metrics."""
        if "typing_speed_kpm" in metrics:
            self.typing_speed_label.setText(f"Typing Speed: {metrics['typing_speed_kpm']:.0f} keys/min")
        if "iki_mean" in metrics:
            self.iki_label.setText(f"Inter-Key Interval: {metrics['iki_mean']:.1f} ms")
        if "correction_rate" in metrics:
            self.correction_rate_label.setText(f"Correction Rate: {metrics['correction_rate']:.1f}/min")
        if "pause_duration_mean" in metrics:
            self.pause_duration_label.setText(f"Pause Duration: {metrics['pause_duration_mean']:.0f} ms")
        if "typing_rhythm_regularity" in metrics:
            if metrics.get("total_keys", 0) < 4:
                self.rhythm_label.setText("Rhythm Regularity: collecting timing")
            else:
                self.rhythm_label.setText(f"Rhythm Regularity: {metrics['typing_rhythm_regularity']:.2f}")
        if "hesitation_score" in metrics:
            self.hesitation_label.setText(f"Hesitation: {metrics['hesitation_score']:.2f}")

    def clear(self) -> None:
        """Reset all metrics."""
        self.input_field.clear()
        self._reset_sample_analysis()
        self.update_keyboard_detected(False)
        self.update_state("--")
        self.update_sample_progress(0.0)
        self.update_baseline(False)
        self.typing_speed_label.setText("Typing Speed: --")
        self.iki_label.setText("Inter-Key Interval: --")
        self.correction_rate_label.setText("Correction Rate: --")
        self.pause_duration_label.setText("Pause Duration: --")
        self.rhythm_label.setText("Rhythm: --")
        self.hesitation_label.setText("Hesitation: --")