"""
Keyboard Analysis Panel for displaying real-time typing behavior analysis.
"""

from typing import Optional, Dict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QProgressBar
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt


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

        # Confidence
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setFont(QFont("Segoe UI", 9))
        self.layout().addWidget(self.confidence_label)

        # Baseline status
        self.baseline_label = QLabel("Baseline: Warming Up")
        self.baseline_label.setFont(QFont("Segoe UI", 9))
        self.baseline_label.setStyleSheet("color: #ffaa00;")
        self.layout().addWidget(self.baseline_label)

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
        self.rhythm_label = QLabel("Rhythm: --")
        self.rhythm_label.setFont(QFont("Segoe UI", 8))
        self.layout().addWidget(self.rhythm_label)

        # Hesitation score
        self.hesitation_label = QLabel("Hesitation: --")
        self.hesitation_label.setFont(QFont("Segoe UI", 8))
        self.layout().addWidget(self.hesitation_label)

        self.layout().addLayout(metrics_layout)
        self.layout().addStretch()

    def update_keyboard_detected(self, detected: bool) -> None:
        """Update keyboard detected status."""
        if detected:
            self.keyboard_detected_label.setText("Keyboard: Active")
            self.keyboard_detected_label.setStyleSheet("color: #00aa00;")
        else:
            self.keyboard_detected_label.setText("Keyboard: Not Active")
            self.keyboard_detected_label.setStyleSheet("color: #cc0000;")

    def update_state(self, state: str, confidence: float) -> None:
        """Update typing state and confidence."""
        self.state_label.setText(f"State: {state}")
        self.confidence_label.setText(f"Confidence: {confidence:.0%}")

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
            self.rhythm_label.setText(f"Rhythm: {metrics['typing_rhythm_regularity']:.2f}")
        if "hesitation_score" in metrics:
            self.hesitation_label.setText(f"Hesitation: {metrics['hesitation_score']:.2f}")

    def clear(self) -> None:
        """Reset all metrics."""
        self.update_keyboard_detected(False)
        self.update_state("--", 0.0)
        self.update_baseline(False)
        self.typing_speed_label.setText("Typing Speed: --")
        self.iki_label.setText("Inter-Key Interval: --")
        self.correction_rate_label.setText("Correction Rate: --")
        self.pause_duration_label.setText("Pause Duration: --")
        self.rhythm_label.setText("Rhythm: --")
        self.hesitation_label.setText("Hesitation: --")