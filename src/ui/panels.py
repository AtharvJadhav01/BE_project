"""
Overall Human State Panel for displaying the fused multimodal human state estimate.
"""

from typing import Optional, Dict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QProgressBar
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt


class OverallHumanStatePanel(QWidget):
    """Panel displaying the overall fused human state estimate."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Title
        title = QLabel("Overall Human State")
        title.setFont(QFont("Segoe UI", 11, 600))  # Bold
        title.setStyleSheet("color: #0078d4;")
        self.layout().addWidget(title)

        # Current state
        self.state_label = QLabel("State: --")
        self.state_label.setFont(QFont("Segoe UI", 14, 600))  # Bold, larger
        self.state_label.setStyleSheet("color: #333333;")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout().addWidget(self.state_label)

        # Confidence
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setFont(QFont("Segoe UI", 10))
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout().addWidget(self.confidence_label)

        # Uncertainty
        self.uncertainty_label = QLabel("Uncertainty: --")
        self.uncertainty_label.setFont(QFont("Segoe UI", 9))
        self.uncertainty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout().addWidget(self.uncertainty_label)

        # Confidence progress bar
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFormat("Confidence: %p%")
        self.confidence_bar.setFixedHeight(20)
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 3px;
            }
        """)
        self.layout().addWidget(self.confidence_bar)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        sep.setStyleSheet("color: #dddddd;")
        self.layout().addWidget(sep)

        # Modality status
        modality_layout = QHBoxLayout()
        modality_layout.setSpacing(8)
        
        self.face_status = QLabel("Face: --")
        self.face_status.setFont(QFont("Segoe UI", 9))
        self.face_status.setStyleSheet("color: #cc0000;")
        modality_layout.addWidget(self.face_status)

        self.audio_status = QLabel("Audio: --")
        self.audio_status.setFont(QFont("Segoe UI", 9))
        self.audio_status.setStyleSheet("color: #cc0000;")
        modality_layout.addWidget(self.audio_status)

        self.keyboard_status = QLabel("Keyboard: --")
        self.keyboard_status.setFont(QFont("Segoe UI", 9))
        self.keyboard_status.setStyleSheet("color: #cc0000;")
        modality_layout.addWidget(self.keyboard_status)

        modality_layout.addStretch()
        self.layout().addLayout(modality_layout)

        self.layout().addStretch()

    def update_state(self, state: str, confidence: float, uncertainty: float) -> None:
        """Update the overall state and confidence."""
        self.state_label.setText(f"State: {state}")
        self.confidence_label.setText(f"Confidence: {confidence:.0%}")
        self.uncertainty_label.setText(f"Uncertainty: {uncertainty:.0%}")
        self.confidence_bar.setValue(int(confidence * 100))

    def update_modality_status(self, modality: str, available: bool, fresh: bool, state: str = "--", confidence: float = 0.0) -> None:
        """Update modality status."""
        label_map = {
            "face": self.face_status,
            "audio": self.audio_status,
            "keyboard": self.keyboard_status,
        }
        
        if modality not in label_map:
            return

        label = label_map[modality]
        
        if not available:
            label.setText(f"{modality.capitalize()}: Unavailable")
            label.setStyleSheet("color: #cc0000;")
        elif not fresh:
            label.setText(f"{modality.capitalize()}: Stale")
            label.setStyleSheet("color: #ffaa00;")
        else:
            label.setText(f"{modality.capitalize()}: {state} ({confidence:.0%})")
            label.setStyleSheet("color: #00aa00;")


class ExplanationPanel(QWidget):
    """Panel displaying the explanation for the fused state."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Title
        title = QLabel("Explanation")
        title.setFont(QFont("Segoe UI", 11, 600))  # Bold
        title.setStyleSheet("color: #0078d4;")
        self.layout().addWidget(title)

        # Explanation text
        self.explanation_label = QLabel("No explanation available.")
        self.explanation_label.setFont(QFont("Segoe UI", 9))
        self.explanation_label.setWordWrap(True)
        self.explanation_label.setStyleSheet("color: #333333; background-color: #f5f5f5; padding: 8px; border-radius: 4px;")
        self.explanation_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.layout().addWidget(self.explanation_label)
        self.layout().addStretch()

    def update_explanation(self, explanation: str) -> None:
        """Update the explanation text."""
        if explanation:
            self.explanation_label.setText(explanation)
        else:
            self.explanation_label.setText("No explanation available.")


class MultimodalContributionPanel(QWidget):
    """Panel showing modality contributions to the fused state."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Title
        title = QLabel("Multimodal Contribution")
        title.setFont(QFont("Segoe UI", 11, 600))  # Bold
        title.setStyleSheet("color: #0078d4;")
        self.layout().addWidget(title)

        # Contribution bars
        self.bars = {}

        for modality in ["face", "audio", "keyboard"]:
            bar_layout = QHBoxLayout()
            bar_layout.setSpacing(8)

            label = QLabel(modality.capitalize())
            label.setFont(QFont("Segoe UI", 9))
            label.setFixedWidth(70)
            bar_layout.addWidget(label)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFormat("%p%")
            bar.setFixedHeight(18)
            bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #cccccc;
                    border-radius: 3px;
                    text-align: center;
                    background-color: #f0f0f0;
                    font-size: 8px;
                }
                QProgressBar::chunk {
                    background-color: #0078d4;
                    border-radius: 2px;
                }
            """)
            bar_layout.addWidget(bar)
            
            self.bars[modality] = bar
            self.layout().addLayout(bar_layout)

        self.layout().addStretch()

    def update_contributions(self, contributions: dict) -> None:
        """Update modality contribution percentages."""
        for modality, value in contributions.items():
            if modality in self.bars:
                self.bars[modality].setValue(int(value * 100))
                self.bars[modality].setFormat(f"{int(value * 100)}%")

    def clear(self) -> None:
        """Reset all bars."""
        for bar in self.bars.values():
            bar.setValue(0)
            bar.setFormat("--")