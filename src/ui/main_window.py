"""
Main application window using PySide6.
"""

import sys
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


logger = get_logger(__name__)


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
            face_capture = FaceCapture(self.session_manager.config.face.__dict__)
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

        # Top panel: Session info and controls
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(12, 12, 12, 12)
        top_layout.setSpacing(12)

        # Session info panel
        self.session_info = SessionInfoPanel()
        top_layout.addWidget(self.session_info)

        # Session controls
        self.session_controls = SessionControls()
        self.session_controls.start_requested.connect(self._on_start_session)
        self.session_controls.pause_requested.connect(self._on_pause_session)
        self.session_controls.resume_requested.connect(self._on_resume_session)
        self.session_controls.stop_requested.connect(self._on_stop_session)
        top_layout.addWidget(self.session_controls)

        splitter.addWidget(top_widget)

        # Bottom panel: Log panel
        self.log_panel = LogPanel()
        splitter.addWidget(self.log_panel)

        # Set splitter sizes
        splitter.setSizes([400, 300])

        # Status bar at bottom
        self.statusBar().showMessage("Ready")
        self.statusBar().setStyleSheet("QStatusBar { background-color: #f0f0f0; }")

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

    def _on_start_session(self) -> None:
        """Handle start session button."""
        result = self.session_manager.start_session()
        if result:
            self._state_worker.start()
            self._duration_timer.start()
            self.log_panel.append_log("INFO", "UI", "Session started")

    def _on_pause_session(self) -> None:
        """Handle pause session button."""
        result = self.session_manager.pause_session()
        if result:
            self.log_panel.append_log("INFO", "UI", "Session paused")

    def _on_resume_session(self) -> None:
        """Handle resume session button."""
        result = self.session_manager.resume_session()
        if result:
            self.log_panel.append_log("INFO", "UI", "Session resumed")

    def _on_stop_session(self) -> None:
        """Handle stop session button."""
        result = self.session_manager.stop_session()
        if result:
            self._state_worker.stop()
            self._duration_timer.stop()
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