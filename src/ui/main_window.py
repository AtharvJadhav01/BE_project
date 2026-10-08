"""
Main application window using PySide6.
"""

import sys
import time
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QProgressBar,
    QTextEdit,
    QSplitter,
    QSizePolicy,
    QMessageBox,
    QHBoxLayout,
)
from PySide6.QtCore import (
    Qt,
    QTimer,
    QThread,
    Signal,
    QObject,
    Slot,
    QSize,
)
from PySide6.QtGui import (
    QFont,
    QColor,
    QPalette,
    QIcon,
    QTextCursor,
)

from core.config import load_config
from core.logging import get_logger, setup_logging
from session.manager import SessionManager
from keyboard import KeyboardCapture
from audio import AudioCapture
from face import FaceCapture

from ui.keyboard_analysis_panel import KeyboardAnalysisPanel
from ui.panels import (
    OverallHumanStatePanel,
    ExplanationPanel,
    MultimodalContributionPanel,
)

logger = get_logger(__name__)


class FaceAnalysisPanel(QWidget):
    """Panel displaying real-time face analysis information."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # Title
        title = QLabel("Face Analysis")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title.setStyleSheet("color: #0078d4;")
        layout.addWidget(title)
        
        # Face detected status
        self.face_detected_label = QLabel("Face: Not Detected")
        self.face_detected_label.setFont(QFont("Segoe UI", 9))
        self.face_detected_label.setStyleSheet("color: #cc0000;")
        layout.addWidget(self.face_detected_label)
        
        # Current facial state
        self.state_label = QLabel("State: --")
        self.state_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(self.state_label)
        
        # Confidence
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setFont(QFont("Segoe UI", 9))
        layout.addWidget(self.confidence_label)
        
        # Baseline status
        self.baseline_label = QLabel("Baseline: Warming Up")
        self.baseline_label.setFont(QFont("Segoe UI", 9))
        self.baseline_label.setStyleSheet("color: #ffaa00;")
        layout.addWidget(self.baseline_label)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        sep.setStyleSheet("color: #dddddd;")
        layout.addWidget(sep)
        
        # Estimated Expression
        expr_title = QLabel("Estimated Expression")
        expr_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        expr_title.setStyleSheet("color: #0078d4;")
        layout.addWidget(expr_title)
        
        self.expr_label = QLabel("Estimated Expression: --")
        self.expr_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.expr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.expr_label)
        
        self.expr_conf_label = QLabel("Expression Confidence: --")
        self.expr_conf_label.setFont(QFont("Segoe UI", 9))
        self.expr_conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.expr_conf_label)
        
        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        sep2.setStyleSheet("color: #dddddd;")
        layout.addWidget(sep2)
        
        # ResNet-18 FER2013 Emotion (Image-based)
        resnet_expr_title = QLabel("ResNet-18 Emotion (FER2013)")
        resnet_expr_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        resnet_expr_title.setStyleSheet("color: #0078d4;")
        layout.addWidget(resnet_expr_title)
        
        self.resnet_expr_label = QLabel("ResNet Emotion: --")
        self.resnet_expr_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.resnet_expr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.resnet_expr_label)
        
        self.resnet_expr_conf_label = QLabel("ResNet Confidence: --")
        self.resnet_expr_conf_label.setFont(QFont("Segoe UI", 9))
        self.resnet_expr_conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.resnet_expr_conf_label)
        
        # Separator
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setFrameShadow(QFrame.Shadow.Sunken)
        sep3.setStyleSheet("color: #dddddd;")
        layout.addWidget(sep3)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        sep.setStyleSheet("color: #dddddd;")
        layout.addWidget(sep)
        
        # Real-time metrics
        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(4)
        
        # Eye metrics
        self.eye_openness_label = QLabel("Eye Openness: --")
        self.eye_openness_label.setFont(QFont("Segoe UI", 8))
        metrics_layout.addWidget(self.eye_openness_label)
        
        self.blink_rate_label = QLabel("Blink Rate: --")
        self.blink_rate_label.setFont(QFont("Segoe UI", 8))
        metrics_layout.addWidget(self.blink_rate_label)
        
        # Mouth metrics
        self.mouth_openness_label = QLabel("Mouth Openness: --")
        self.mouth_openness_label.setFont(QFont("Segoe UI", 8))
        metrics_layout.addWidget(self.mouth_openness_label)
        
        self.mouth_width_label = QLabel("Smile Score: --")
        self.mouth_width_label.setFont(QFont("Segoe UI", 8))
        metrics_layout.addWidget(self.mouth_width_label)
        
        # Facial movement
        self.movement_label = QLabel("Facial Movement: --")
        self.movement_label.setFont(QFont("Segoe UI", 8))
        metrics_layout.addWidget(self.movement_label)
        
        self.symmetry_label = QLabel("Symmetry: --")
        self.symmetry_label.setFont(QFont("Segoe UI", 8))
        metrics_layout.addWidget(self.symmetry_label)
        
        # Head pose
        self.head_pose_label = QLabel("Head Pose: --")
        self.head_pose_label.setFont(QFont("Segoe UI", 8))
        metrics_layout.addWidget(self.head_pose_label)
        
        layout.addLayout(metrics_layout)
        layout.addStretch()

    def update_face_detected(self, detected: bool) -> None:
        """Update face detected status."""
        if detected:
            self.face_detected_label.setText("Face: Detected")
            self.face_detected_label.setStyleSheet("color: #00aa00;")
        else:
            self.face_detected_label.setText("Face: Not Detected")
            self.face_detected_label.setStyleSheet("color: #cc0000;")

    def update_state(self, state: str, confidence: float) -> None:
        """Update facial state and confidence."""
        self.state_label.setText(f"State: {state}")
        self.confidence_label.setText(f"Confidence: {confidence:.0%}")

    def update_expression(self, expression: str, confidence: float) -> None:
        """Update estimated expression and confidence."""
        self.expr_label.setText(f"Estimated Expression: {expression}")
        self.expr_conf_label.setText(f"Expression Confidence: {confidence:.0%}")

    def update_resnet_emotion(self, emotion: str, confidence: float) -> None:
        """Update ResNet-18 FER2013 emotion and confidence."""
        self.resnet_expr_label.setText(f"ResNet Emotion: {emotion}")
        self.resnet_expr_conf_label.setText(f"ResNet Confidence: {confidence:.0%}")
        self.logger.debug("face_panel_resnet_update", emotion=emotion, confidence=f"{confidence:.0%}")

    def update_metrics(self, metrics: dict) -> None:
        """Update real-time facial metrics."""
        if "eye_openness_avg" in metrics:
            self.eye_openness_label.setText(f"Eye Openness: {metrics['eye_openness_avg']:.2f}")
        if "blink_rate" in metrics:
            self.blink_rate_label.setText(f"Blink Rate: {metrics['blink_rate']:.1f}/min")
        if "mouth_openness" in metrics:
            self.mouth_openness_label.setText(f"Mouth Openness: {metrics['mouth_openness']:.2f}")
        if "mouth_width" in metrics:
            self.mouth_width_label.setText(f"Smile Score: {metrics['mouth_width']:.2f}")
        if "facial_movement_velocity" in metrics:
            self.movement_label.setText(f"Facial Movement: {metrics['facial_movement_velocity']:.4f}")
        if "facial_symmetry" in metrics:
            self.symmetry_label.setText(f"Symmetry: {metrics['facial_symmetry']:.2f}")
        if "head_pitch" in metrics and "head_yaw" in metrics and "head_roll" in metrics:
            self.head_pose_label.setText(
                f"Head Pose: Pitch={metrics['head_pitch']:.1f}° Yaw={metrics['head_yaw']:.1f}° Roll={metrics['head_roll']:.1f}°"
            )

    def update_baseline(self, ready: bool, progress: float = 0.0) -> None:
        """Update baseline status."""
        if ready:
            self.baseline_label.setText("Baseline: Ready")
            self.baseline_label.setStyleSheet("color: #00aa00;")
        else:
            self.baseline_label.setText(f"Baseline: Warming Up ({progress:.0%})")
            self.baseline_label.setStyleSheet("color: #ffaa00;")

    def clear(self) -> None:
        """Reset all metrics."""
        self.update_face_detected(False)
        self.update_state("--", 0.0)
        self.update_baseline(False)
        self.update_expression("--", 0.0)
        self.update_resnet_emotion("--", 0.0)
        self.eye_openness_label.setText("Eye Openness: --")
        self.blink_rate_label.setText("Blink Rate: --")
        self.mouth_openness_label.setText("Mouth Openness: --")
        self.mouth_width_label.setText("Smile Score: --")
        self.movement_label.setText("Facial Movement: --")
        self.symmetry_label.setText("Symmetry: --")
        self.head_pose_label.setText("Head Pose: --")


class AudioAnalysisPanel(QWidget):
    """Panel displaying real-time audio analysis information."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # Title
        title = QLabel("Audio Analysis")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title.setStyleSheet("color: #0078d4;")
        layout.addWidget(title)
        
        # Voice activity status
        self.voice_activity_label = QLabel("Voice: Silent")
        self.voice_activity_label.setFont(QFont("Segoe UI", 9))
        self.voice_activity_label.setStyleSheet("color: #cc0000;")
        layout.addWidget(self.voice_activity_label)
        
        # Current vocal state
        self.state_label = QLabel("State: --")
        self.state_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(self.state_label)
        
        # Confidence
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setFont(QFont("Segoe UI", 9))
        layout.addWidget(self.confidence_label)
        
        # Baseline status
        self.baseline_label = QLabel("Baseline: Warming Up")
        self.baseline_label.setFont(QFont("Segoe UI", 9))
        self.baseline_label.setStyleSheet("color: #ffaa00;")
        layout.addWidget(self.baseline_label)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        sep.setStyleSheet("color: #dddddd;")
        layout.addWidget(sep)
        
        # Real-time metrics
        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(4)
        
        # Pitch
        self.pitch_label = QLabel("Pitch: --")
        self.pitch_label.setFont(QFont("Segoe UI", 8))
        metrics_layout.addWidget(self.pitch_label)
        
        self.pitch_var_label = QLabel("Pitch Variation: --")
        self.pitch_var_label.setFont(QFont("Segoe UI", 8))
        metrics_layout.addWidget(self.pitch_var_label)
        
        # Energy
        self.energy_label = QLabel("Energy: --")
        self.energy_label.setFont(QFont("Segoe UI", 8))
        metrics_layout.addWidget(self.energy_label)
        
        self.energy_var_label = QLabel("Energy Variation: --")
        self.energy_var_label.setFont(QFont("Segoe UI", 8))
        metrics_layout.addWidget(self.energy_var_label)
        
        # ZCR
        self.zcr_label = QLabel("ZCR: --")
        self.zcr_label.setFont(QFont("Segoe UI", 8))
        metrics_layout.addWidget(self.zcr_label)
        
        # Voice activity
        self.voiced_ratio_label = QLabel("Voiced Ratio: --")
        self.voiced_ratio_label.setFont(QFont("Segoe UI", 8))
        metrics_layout.addWidget(self.voiced_ratio_label)
        
        # Spectral
        self.spectral_centroid_label = QLabel("Spectral Centroid: --")
        self.spectral_centroid_label.setFont(QFont("Segoe UI", 8))
        metrics_layout.addWidget(self.spectral_centroid_label)
        
        layout.addLayout(metrics_layout)
        layout.addStretch()

    def update_voice_activity(self, active: bool) -> None:
        """Update voice activity status."""
        if active:
            self.voice_activity_label.setText("Voice: Active")
            self.voice_activity_label.setStyleSheet("color: #00aa00;")
        else:
            self.voice_activity_label.setText("Voice: Silent")
            self.voice_activity_label.setStyleSheet("color: #cc0000;")
    def update_state(self, state: str, confidence: float) -> None:
        """Update vocal state and confidence."""
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
        """Update real-time audio metrics."""
        if "pitch_hz" in metrics:
            self.pitch_label.setText(f"Pitch: {metrics['pitch_hz']:.1f} Hz")
        if "pitch_std" in metrics:
            self.pitch_var_label.setText(f"Pitch Variation: {metrics['pitch_std']:.1f} Hz")
        if "rms_energy_db" in metrics:
            self.energy_label.setText(f"Energy: {metrics['rms_energy_db']:.1f} dB")
        if "energy_std" in metrics:
            self.energy_var_label.setText(f"Energy Variation: {metrics['energy_std']:.1f} dB")
        if "zcr" in metrics:
            self.zcr_label.setText(f"ZCR: {metrics['zcr']:.3f}")
        if "voiced_ratio" in metrics:
            self.voiced_ratio_label.setText(f"Voiced Ratio: {metrics['voiced_ratio']:.2f}")
        if "spectral_centroid" in metrics:
            self.spectral_centroid_label.setText(f"Spectral Centroid: {metrics['spectral_centroid']:.0f} Hz")

    def clear(self) -> None:
        """Reset all metrics."""
        self.update_voice_activity(False)
        self.update_state("--", 0.0)
        self.update_baseline(False)
        self.pitch_label.setText("Pitch: --")
        self.pitch_var_label.setText("Pitch Variation: --")
        self.energy_label.setText("Energy: --")
        self.energy_var_label.setText("Energy Variation: --")
        self.zcr_label.setText("ZCR: --")
        self.voiced_ratio_label.setText("Voiced Ratio: --")
        self.spectral_centroid_label.setText("Spectral Centroid: --")


class StatusIndicator(QWidget):
    """A colored indicator widget for modality status."""

    def __init__(self, label: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self.led = QLabel()
        self.led.setFixedSize(12, 12)
        self.led.setStyleSheet("border-radius: 6px; background-color: #666666;")

        self.label = QLabel(label)
        self.label.setFont(QFont("Segoe UI", 9))

        self.status_label = QLabel("Stopped")
        self.status_label.setFont(QFont("Segoe UI", 9))
        self.status_label.setStyleSheet("color: #666666;")

        layout.addWidget(self.led)
        layout.addWidget(self.label)
        layout.addWidget(self.status_label)
        layout.addStretch()

        self._status = "stopped"

    def set_status(self, status: str) -> None:
        """Update the status indicator."""
        self._status = status.lower()
        colors = {
            "ready": "#666666",
            "starting": "#ffaa00",
            "running": "#00aa00",
            "recording": "#00aa00",
            "paused": "#ffaa00",
            "stopping": "#ff5500",
            "stopped": "#666666",
            "error": "#cc0000",
        }
        color = colors.get(self._status, "#666666")
        self.led.setStyleSheet(f"border-radius: 6px; background-color: {color};")
        self.status_label.setText(status.capitalize())
        self.status_label.setStyleSheet(f"color: {color};")


class SessionInfoPanel(QWidget):
    """Panel displaying session information."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Session ID
        self.session_id_label = QLabel("Session: Not started")
        self.session_id_label.setFont(QFont("Segoe UI", 9))
        self.session_id_label.setWordWrap(True)
        layout.addWidget(self.session_id_label)

        # Duration
        self.duration_label = QLabel("Duration: 00:00:00")
        self.duration_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.duration_label.setStyleSheet("color: #0066cc;")
        layout.addWidget(self.duration_label)

        # Enabled modalities
        self.modalities_label = QLabel("Modalities: None")
        self.modalities_label.setFont(QFont("Segoe UI", 9))
        layout.addWidget(self.modalities_label)

        layout.addStretch()

    def update_session_info(self, session_id: str, duration_ms: int, modalities: list) -> None:
        self.session_id_label.setText(f"Session: {session_id[:8]}...")
        self.duration_label.setText(f"Duration: {self._format_duration(duration_ms)}")
        self.modalities_label.setText(f"Modalities: {', '.join(modalities) if modalities else 'None'}")

    def reset(self) -> None:
        self.session_id_label.setText("Session: Not started")
        self.duration_label.setText("Duration: 00:00:00")
        self.modalities_label.setText("Modalities: None")

    @staticmethod
    def _format_duration(ms: int) -> str:
        total_seconds = ms // 1000
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class SessionControls(QWidget):
    """Session control buttons."""

    start_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        self.btn_start = QPushButton("Start Session")
        self.btn_start.setFixedHeight(36)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #106ebe; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)
        self.btn_start.clicked.connect(self.start_requested.emit)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setFixedHeight(36)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setStyleSheet("""
            QPushButton {
                background-color: #ffaa00;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #e69900; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)
        self.btn_pause.clicked.connect(self.pause_requested.emit)

        self.btn_resume = QPushButton("Resume")
        self.btn_resume.setFixedHeight(36)
        self.btn_resume.setEnabled(False)
        self.btn_resume.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #106ebe; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)
        self.btn_resume.clicked.connect(self.resume_requested.emit)

        self.btn_stop = QPushButton("Stop Session")
        self.btn_stop.setFixedHeight(36)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #cc0000;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #aa0000; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)
        self.btn_stop.clicked.connect(self.stop_requested.emit)

        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_pause)
        layout.addWidget(self.btn_resume)
        layout.addWidget(self.btn_stop)

    def set_state(self, state: str) -> None:
        """Update button states based on session state."""
        self.btn_start.setEnabled(state in ("ready", "stopped"))
        self.btn_pause.setEnabled(state == "recording")
        self.btn_resume.setEnabled(state == "paused")
        self.btn_stop.setEnabled(state in ("recording", "paused"))


class LogPanel(QTextEdit):
    """Log panel for displaying structured logs."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 8))
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #333333;
                border-radius: 4px;
            }
        """)
        self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.document().setMaximumBlockCount(1000)

    def append_log(self, level: str, module: str, message: str, timestamp: str = "") -> None:
        """Append a log entry with color coding."""
        colors = {
            "DEBUG": "#6a9955",
            "INFO": "#9cdcfe",
            "WARNING": "#ffcc00",
            "ERROR": "#f44747",
            "CRITICAL": "#ff0000",
        }
        color = colors.get(level.upper(), "#d4d4d4")

        if not timestamp:
            from datetime import datetime, timezone
            timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]

        html = f'<span style="color: #888888;">[{timestamp}]</span> ' \
               f'<span style="color: {color}; font-weight: bold;">{level}</span> ' \
               f'<span style="color: #9cdcfe;">{module}</span>: ' \
               f'<span style="color: #d4d4d4;">{message}</span>'

        self.append(html)
        # Auto-scroll to bottom
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)

    def clear_logs(self) -> None:
        self.clear()


class SessionStateWorker(QObject):
    """Worker to monitor session state and emit updates."""

    state_changed = Signal(str)  # state
    session_updated = Signal(str, int, list)  # session_id, duration_ms, modalities
    modality_status_changed = Signal(str, str)  # modality, status
    log_received = Signal(str, str, str, str)  # level, module, message, timestamp
    error_occurred = Signal(str)

    def __init__(self, session_manager):
        super().__init__()
        self.session_manager = session_manager
        self._running = False
        self._timer = QTimer()
        self._timer.setInterval(500)  # Update every 500ms
        self._timer.timeout.connect(self._poll_state)

    def start(self) -> None:
        self._running = True
        self._timer.start()

    def stop(self) -> None:
        self._running = False
        self._timer.stop()

    def _poll_state(self) -> None:
        if not self._running or not self.session_manager:
            return

        # Emit state changes
        state = self.session_manager.state.value
        self.state_changed.emit(state)

        if self.session_manager.metadata:
            self.session_updated.emit(
                self.session_manager.metadata.session_id,
                self.session_manager.metadata.duration_ms or 0,
                self.session_manager.metadata.enabled_modalities
            )

            # Check modality status changes
            for name, modality in self.session_manager.modalities.items():
                # We need to track previous status to detect changes
                pass


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimodal Human State Capture")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        # Load configuration
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        self.config = load_config(config_path)

        # Setup logging
        setup_logging(self.config.logging)

        # Initialize session manager
        self.session_manager = SessionManager(config=self.config, application_version="0.1.0")

        # Register modalities
        self._register_modalities()

        # Initialize UI
        self._init_ui()

        # State monitoring
        self._state_worker = SessionStateWorker(self.session_manager)
        self._state_worker.state_changed.connect(self._on_state_changed)
        self._state_worker.session_updated.connect(self._on_session_updated)
        self._state_worker.error_occurred.connect(self._on_error)

        # Duration timer
        self._duration_timer = QTimer()
        self._duration_timer.setInterval(1000)
        self._duration_timer.timeout.connect(self._update_duration)

        # Register modality event callbacks
        self._register_event_callbacks()

    def _register_modalities(self) -> None:
        """Register real modality implementations."""
        # Register keyboard capture
        if self.session_manager.config.keyboard.enabled:
            keyboard_capture = KeyboardCapture(self.session_manager.config.keyboard.__dict__)
            keyboard_capture.set_unified_event_callback(
                lambda event: self.session_manager._emit_event(event)
            )
            self.session_manager.register_modality(keyboard_capture)
            logger.info("keyboard_modality_registered")

        # Register audio capture
        if self.session_manager.config.audio.enabled:
            audio_capture = AudioCapture(self.session_manager.config.audio.__dict__)
            audio_capture.set_unified_event_callback(
                lambda event: self.session_manager._emit_event(event)
            )
            self.session_manager.register_modality(audio_capture)
            logger.info("audio_modality_registered")

        # Register face capture
        if self.session_manager.config.camera.enabled and self.session_manager.config.face.enabled:
            face_config = self.session_manager.config.analysis.face
            logger.info("resnet_emotion_config", 
                       enabled=face_config.resnet_emotion_enabled,
                       model_path=face_config.resnet_emotion_model_path,
                       device=face_config.resnet_emotion_device,
                       confidence_threshold=face_config.resnet_emotion_confidence_threshold)
            face_capture = FaceCapture(face_config.__dict__)
            face_capture.set_unified_event_callback(
                lambda event: self.session_manager._emit_event(event)
            )
            self.session_manager.register_modality(face_capture)
            logger.info("face_modality_registered")

    def _register_event_callbacks(self) -> None:
        """Register callbacks for modality events."""
        def on_event(event):
            # Update modality status indicators
            pass

        self.session_manager.add_event_callback(on_event)

    def _init_ui(self) -> None:
        """Initialize the UI components."""
        self.setWindowTitle("Multimodal Human State Capture")

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top status bar
        self._create_status_bar(main_layout)

        # Main content area with splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter, 1)

        # Top panel: Session info, controls, face analysis, and audio analysis
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(12, 12, 12, 12)
        top_layout.setSpacing(12)

        # Left column: Session info and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Session info panel
        self.session_info = SessionInfoPanel()
        left_layout.addWidget(self.session_info)

        # Session controls
        self.session_controls = SessionControls()
        self.session_controls.start_requested.connect(self._on_start_session)
        self.session_controls.pause_requested.connect(self._on_pause_session)
        self.session_controls.resume_requested.connect(self._on_resume_session)
        self.session_controls.stop_requested.connect(self._on_stop_session)
        left_layout.addWidget(self.session_controls)

        left_layout.addStretch()

        # Middle column: Face analysis panel
        self.face_analysis = FaceAnalysisPanel()
        middle_widget = QWidget()
        middle_layout = QVBoxLayout(middle_widget)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.addWidget(self.face_analysis)
        middle_layout.addStretch()

        # Right column: Audio analysis panel
        self.audio_analysis = AudioAnalysisPanel()
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.audio_analysis)
        right_layout.addStretch()

        top_layout.addWidget(left_widget, 1)
        top_layout.addWidget(middle_widget, 1)
        top_layout.addWidget(right_widget, 1)

        # Add keyboard analysis panel
        self.keyboard_analysis = KeyboardAnalysisPanel()
        keyboard_widget = QWidget()
        keyboard_layout = QVBoxLayout(keyboard_widget)
        keyboard_layout.setContentsMargins(0, 0, 0, 0)
        keyboard_layout.addWidget(self.keyboard_analysis)
        keyboard_layout.addStretch()
        top_layout.addWidget(keyboard_widget, 1)

        splitter.addWidget(top_widget)

        # Middle panel: Overall human state, explanation, and multimodal contribution
        middle_panel = QWidget()
        middle_layout = QHBoxLayout(middle_panel)
        middle_layout.setContentsMargins(12, 12, 12, 12)
        middle_layout.setSpacing(12)

        # Overall human state panel
        self.overall_human_state = OverallHumanStatePanel()
        middle_layout.addWidget(self.overall_human_state, 1)

        # Explanation panel
        self.explanation_panel = ExplanationPanel()
        middle_layout.addWidget(self.explanation_panel, 1)

        # Multimodal contribution panel
        self.multimodal_contribution = MultimodalContributionPanel()
        middle_layout.addWidget(self.multimodal_contribution, 1)

        splitter.addWidget(middle_panel)

        # Bottom panel: Log panel
        self.log_panel = LogPanel()
        splitter.addWidget(self.log_panel)

        # Set splitter sizes
        splitter.setSizes([400, 250, 300])

        # Status bar at bottom
        self.statusBar().showMessage("Ready")
        self.statusBar().setStyleSheet("QStatusBar { background-color: #f0f0f0; }")

        # Face analysis polling timer
        self._face_analysis_timer = QTimer()
        self._face_analysis_timer.setInterval(200)  # Update every 200ms
        self._face_analysis_timer.timeout.connect(self._poll_face_analysis)

        # Audio analysis polling timer
        self._audio_analysis_timer = QTimer()
        self._audio_analysis_timer.setInterval(200)  # Update every 200ms
        self._audio_analysis_timer.timeout.connect(self._poll_audio_analysis)

        # Keyboard analysis polling timer
        self._keyboard_analysis_timer = QTimer()
        self._keyboard_analysis_timer.setInterval(200)  # Update every 200ms
        self._keyboard_analysis_timer.timeout.connect(self._poll_keyboard_analysis)

    def _create_status_bar(self, layout: QVBoxLayout) -> None:
        """Create the top status bar with application state."""
        status_frame = QFrame()
        status_frame.setFixedHeight(50)
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border-bottom: 1px solid #dddddd;
            }
        """)

        layout_h = QHBoxLayout(status_frame)
        layout_h.setContentsMargins(16, 8, 16, 8)
        layout_h.setSpacing(16)

        # Application title
        title_label = QLabel("Multimodal Human State Capture")
        title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #0078d4;")
        layout_h.addWidget(title_label)

        layout_h.addStretch()

        # Overall session status
        self.overall_status = StatusIndicator("Session")
        self.overall_status.set_status("ready")
        layout_h.addWidget(self.overall_status)

        layout_h.addWidget(self._create_separator())

        # Modality indicators
        self.mic_status = StatusIndicator("Microphone")
        self.mic_status.set_status("ready")
        layout_h.addWidget(self.mic_status)

        layout_h.addWidget(self._create_separator())

        self.camera_status = StatusIndicator("Camera")
        self.camera_status.set_status("ready")
        layout_h.addWidget(self.camera_status)

        layout_h.addWidget(self._create_separator())

        self.keyboard_status = StatusIndicator("Keyboard")
        self.keyboard_status.set_status("ready")
        layout_h.addWidget(self.keyboard_status)

        layout_h.addStretch()

        # Add the status frame to the main layout
        layout.addWidget(status_frame)

    def _create_separator(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        sep.setStyleSheet("color: #dddddd;")
        sep.setFixedWidth(1)
        return sep

    def _on_state_changed(self, state: str) -> None:
        """Handle session state changes."""
        self.session_controls.set_state(state)

        # Update overall status
        if state == "recording":
            self.overall_status.set_status("recording")
        elif state == "paused":
            self.overall_status.set_status("paused")
        elif state == "stopped":
            self.overall_status.set_status("stopped")
        elif state == "ready":
            self.overall_status.set_status("ready")
        elif state == "error":
            self.overall_status.set_status("error")

        # Update status bar
        self.statusBar().showMessage(f"State: {state.capitalize()}")

    def _on_session_updated(self, session_id: str, duration_ms: int, modalities: list) -> None:
        """Update session info panel."""
        self.session_info.update_session_info(session_id, duration_ms, modalities)

    def _on_error(self, error_msg: str) -> None:
        """Handle errors from the session manager."""
        self.log_panel.append_log("ERROR", "Application", error_msg)
        QMessageBox.critical(self, "Error", error_msg)

    def _update_duration(self) -> None:
        """Update the duration display."""
        if self.session_manager and self.session_manager.metadata:
            duration = self.session_manager.metadata.duration_ms or 0
            self.session_info.update_session_info(
                self.session_manager.metadata.session_id,
                duration,
                self.session_manager.metadata.enabled_modalities
            )

    def _poll_face_analysis(self) -> None:
        """Poll face analysis data from the face capture modality."""
        if not self.session_manager or self.session_manager.state.value != "recording":
            return
        
        face_capture = self.session_manager.modalities.get("face")
        if not face_capture:
            return
        
        try:
            stats = face_capture.get_stats()
            analysis = stats.get("analysis", {})
            
            if not analysis:
                # No analysis data yet
                self.face_analysis.update_face_detected(False)
                return
            
            # Update face detected status
            # We can infer from analysis data
            last_ts = analysis.get("last_analysis_timestamp")
            if last_ts:
                # Check if recent (within 1 second)
                current_time = time.monotonic()
                if current_time - last_ts < 1.0:
                    self.face_analysis.update_face_detected(True)
                else:
                    self.face_analysis.update_face_detected(False)
            
            # Get baseline status
            # We need to access the baseline manager directly
            # For now, use the stats
            baseline_ready = analysis.get("frames_processed", 0) > 0
            # We can't easily get baseline progress from stats, so we'll estimate
            frames = analysis.get("frames_processed", 0)
            baseline_progress = min(1.0, frames / 150.0)  # 150 frames for baseline
            self.face_analysis.update_baseline(baseline_ready, baseline_progress)
            
            # The face capture stores the last analysis result
            # We need to access it directly
            if hasattr(face_capture, '_last_analysis_result') and face_capture._last_analysis_result:
                result = face_capture._last_analysis_result
                
                # Update state
                state_est = result.get("state_estimate")
                if state_est:
                    self.face_analysis.update_state(
                        state_est.get("state", "--"),
                        state_est.get("confidence", 0.0)
                    )
                
                # Update estimated expression
                expr_est = result.get("expression_estimate")
                if expr_est:
                    expr_value = expr_est.get("expression", "--")
                    expr_confidence = expr_est.get("confidence", 0.0)
                    # Show "--" when expression is "unavailable" (no face) or confidence is 0
                    if expr_value == "unavailable" or expr_confidence == 0.0:
                        expr_value = "--"
                        expr_confidence = 0.0
                    self.face_analysis.update_expression(
                        expr_value, expr_confidence
                    )
                else:
                    # No expression estimate (emotion disabled, classifier unavailable, 
                    # or no face detected before baseline ready)
                    self.face_analysis.update_expression("--", 0.0)
                
                # Update metrics
                norm_features = result.get("normalized_features", {})
                self.face_analysis.update_metrics(norm_features)
                
                # Update ResNet emotion
                resnet_emotion_est = result.get("resnet_emotion_estimate")
                if resnet_emotion_est:
                    emotion = resnet_emotion_est.get("emotion", "--")
                    confidence = resnet_emotion_est.get("confidence", 0.0)
                    if emotion == "unavailable" or confidence == 0.0:
                        emotion = "--"
                        confidence = 0.0
                    self.face_analysis.update_resnet_emotion(emotion, confidence)
                    self.logger.debug("resnet_gui_update", emotion=emotion, confidence=f"{confidence:.0%}")
                else:
                    self.face_analysis.update_resnet_emotion("--", 0.0)
                
        except Exception as e:
            logger.error("face_analysis_poll_failed", error=str(e))

    def _poll_audio_analysis(self) -> None:
        """Poll audio analysis data from the audio capture modality."""
        if not self.session_manager or self.session_manager.state.value != "recording":
            return
        
        audio_capture = self.session_manager.modalities.get("audio")
        if not audio_capture:
            return
        
        try:
            stats = audio_capture.get_stats()
            analysis = stats.get("analysis", {})
            
            if not analysis:
                # No analysis data yet
                self.audio_analysis.update_voice_activity(False)
                return
            
            # Update voice activity status
            last_ts = analysis.get("last_analysis_timestamp")
            if last_ts:
                current_time = time.monotonic()
                if current_time - last_ts < 1.0:
                    self.audio_analysis.update_voice_activity(True)
                else:
                    self.audio_analysis.update_voice_activity(False)
            
            # Get baseline status
            baseline_ready = analysis.get("frames_processed", 0) > 0
            frames = analysis.get("frames_processed", 0)
            baseline_progress = min(1.0, frames / 150.0)  # 150 frames for baseline
            self.audio_analysis.update_baseline(baseline_ready, baseline_progress)
            
            # The audio capture stores the last analysis result
            if hasattr(audio_capture, '_last_analysis_result') and audio_capture._last_analysis_result:
                result = audio_capture._last_analysis_result
                
                # Update state
                state_est = result.get("state_estimate")
                if state_est:
                    self.audio_analysis.update_state(
                        state_est.get("state", "--"),
                        state_est.get("confidence", 0.0)
                    )
                
                # Update metrics
                norm_features = result.get("normalized_features", {})
                self.audio_analysis.update_metrics(norm_features)
                
        except Exception as e:
            logger.error("audio_analysis_poll_failed", error=str(e))

    def _poll_keyboard_analysis(self) -> None:
        """Poll keyboard analysis data from the keyboard capture modality."""
        if not self.session_manager or self.session_manager.state.value != "recording":
            return

        if self.keyboard_analysis.session_text_recording_active:
            return
        
        keyboard_capture = self.session_manager.modalities.get("keyboard")
        if not keyboard_capture:
            return
        
        try:
            stats = keyboard_capture.get_stats()
            analysis = stats.get("analysis", {})
            
            if not analysis:
                # No analysis data yet
                self.keyboard_analysis.update_keyboard_detected(False)
                return
            
            # Update keyboard detected status
            last_ts = analysis.get("last_analysis_timestamp")
            if last_ts:
                current_time = time.monotonic()
                if current_time - last_ts < 1.0:
                    self.keyboard_analysis.update_keyboard_detected(True)
                else:
                    self.keyboard_analysis.update_keyboard_detected(False)
            
            # Get baseline status
            baseline_ready = analysis.get("frames_processed", 0) > 0
            frames = analysis.get("frames_processed", 0)
            baseline_progress = min(1.0, frames / 150.0)  # 150 frames for baseline
            self.keyboard_analysis.update_baseline(baseline_ready, baseline_progress)
            
            # The keyboard capture stores the last analysis result
            if hasattr(keyboard_capture, '_last_analysis_result') and keyboard_capture._last_analysis_result:
                result = keyboard_capture._last_analysis_result
                
                # Update state
                state_est = result.get("state_estimate")
                if state_est:
                    self.keyboard_analysis.update_state(
                        state_est.get("state", "--")
                    )
                
                # Update metrics
                norm_features = result.get("normalized_features", {})
                self.keyboard_analysis.update_metrics(norm_features)
                
        except Exception as e:
            logger.error("keyboard_analysis_poll_failed", error=str(e))

    def _on_start_session(self) -> None:
        """Handle start session button."""
        result = self.session_manager.start_session()
        if result:
            session_text_path = (
                Path(self.session_manager.config.storage.output_directory)
                / self.session_manager.session_id
                / "typed_text.txt"
            )
            self.keyboard_analysis.begin_session_text_recording(session_text_path)
            self._state_worker.start()
            self._duration_timer.start()
            self._face_analysis_timer.start()
            self._audio_analysis_timer.start()
            self._keyboard_analysis_timer.start()
            self.log_panel.append_log("INFO", "UI", "Session started")

    def _on_pause_session(self) -> None:
        """Handle pause session button."""
        result = self.session_manager.pause_session()
        if result:
            self.keyboard_analysis.set_session_text_recording_active(False)
            self.log_panel.append_log("INFO", "UI", "Session paused")

    def _on_resume_session(self) -> None:
        """Handle resume session button."""
        result = self.session_manager.resume_session()
        if result:
            self.keyboard_analysis.set_session_text_recording_active(True)
            self.log_panel.append_log("INFO", "UI", "Session resumed")

    def _on_stop_session(self) -> None:
        """Handle stop session button."""
        result = self.session_manager.stop_session()
        if result:
            self._state_worker.stop()
            self._duration_timer.stop()
            self._face_analysis_timer.stop()
            self._audio_analysis_timer.stop()
            self._keyboard_analysis_timer.stop()
            self.face_analysis.clear()
            self.audio_analysis.clear()
            self.keyboard_analysis.finish_session_text_recording()
            self.overall_human_state.clear()
            self.explanation_panel.update_explanation("")
            self.multimodal_contribution.clear()
            self.log_panel.append_log("INFO", "UI", "Session stopped")

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        if self.session_manager and self.session_manager.state.value in ("recording", "paused"):
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "A session is in progress. Stop the session and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.session_manager.stop_session()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def closeEvent(self, event) -> None:
        """Handle window close."""
        super().closeEvent(event)


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Set application font
    font = QFont("Segoe UI", 9)
    app.setFont(font)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())