"""
Unit tests for the UI module.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt

from ui.main_window import (
    MainWindow,
    StatusIndicator,
    SessionInfoPanel,
    SessionControls,
    LogPanel,
)
from ui.keyboard_analysis_panel import KeyboardAnalysisPanel


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def window(qapp):
    """Create a MainWindow instance for testing."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    
    from ui.main_window import MainWindow
    
    # Mock the config loading to avoid file I/O
    import main_window
    original_load_config = main_window.load_config
    
    def mock_load_config(path):
        from core.config import get_default_config
        config = get_default_config()
        config.storage.output_directory = "/tmp/test_sessions"
        return config
    
    main_window.load_config = mock_load_config
    
    window = MainWindow()
    yield window
    window.close()
    
    main_window.load_config = original_load_config


class TestStatusIndicator:
    """Tests for StatusIndicator widget."""

    def test_creation(self, qapp):
        """Test StatusIndicator creation."""
        indicator = StatusIndicator("Test")
        assert indicator is not None
        assert indicator._status == "stopped"

    def test_set_status(self, qapp):
        """Test setting status."""
        indicator = StatusIndicator("Test")
        
        indicator.set_status("ready")
        assert indicator._status == "ready"
        
        indicator.set_status("running")
        assert indicator._status == "running"
        
        indicator.set_status("error")
        assert indicator._status == "error"


class TestSessionInfoPanel:
    """Tests for SessionInfoPanel widget."""

    def test_creation(self, qapp):
        """Test SessionInfoPanel creation."""
        panel = SessionInfoPanel()
        assert panel is not None

    def test_update_session_info(self, qapp):
        """Test updating session info."""
        panel = SessionInfoPanel()
        
        panel.update_session_info("test-session-123", 5000, ["audio", "keyboard"])
        
        assert "test-ses" in panel.session_id_label.text()
        assert "00:00:05" in panel.duration_label.text()
        assert "audio, keyboard" in panel.modalities_label.text()

    def test_reset(self, qapp):
        """Test resetting panel."""
        panel = SessionInfoPanel()
        
        panel.update_session_info("test-session-123", 5000, ["audio"])
        panel.reset()
        
        assert "Not started" in panel.session_id_label.text()
        assert "00:00:00" in panel.duration_label.text()
        assert "None" in panel.modalities_label.text()


class TestSessionControls:
    """Tests for SessionControls widget."""

    def test_creation(self, qapp):
        """Test SessionControls creation."""
        controls = SessionControls()
        assert controls is not None

    def test_initial_state(self, qapp):
        """Test initial button states."""
        controls = SessionControls()
        
        assert controls.btn_start.isEnabled() is True
        assert controls.btn_pause.isEnabled() is False
        assert controls.btn_resume.isEnabled() is False
        assert controls.btn_stop.isEnabled() is False

    def test_state_transitions(self, qapp):
        """Test button state transitions."""
        controls = SessionControls()
        
        # Ready state
        controls.set_state("ready")
        assert controls.btn_start.isEnabled() is True
        assert controls.btn_pause.isEnabled() is False
        assert controls.btn_resume.isEnabled() is False
        assert controls.btn_stop.isEnabled() is False
        
        # Recording state
        controls.set_state("recording")
        assert controls.btn_start.isEnabled() is False
        assert controls.btn_pause.isEnabled() is True
        assert controls.btn_resume.isEnabled() is False
        assert controls.btn_stop.isEnabled() is True
        
        # Paused state
        controls.set_state("paused")
        assert controls.btn_start.isEnabled() is False
        assert controls.btn_pause.isEnabled() is False
        assert controls.btn_resume.isEnabled() is True
        assert controls.btn_stop.isEnabled() is True


class TestLogPanel:
    """Tests for LogPanel widget."""

    def test_creation(self, qapp):
        """Test LogPanel creation."""
        log = LogPanel()
        assert log is not None
        assert log.isReadOnly() is True

    def test_append_log(self, qapp):
        """Test appending log entries."""
        log = LogPanel()
        
        log.append_log("INFO", "TestModule", "Test message")
        
        assert log.document().blockCount() == 1
        
        log.append_log("ERROR", "TestModule", "Error message")
        
        assert log.document().blockCount() == 2

    def test_clear_logs(self, qapp):
        """Test clearing logs."""
        log = LogPanel()
        
        log.append_log("INFO", "TestModule", "Test message")
        log.clear_logs()
        
        # QTextEdit.clear() leaves one empty block
        assert log.document().blockCount() == 1


class TestKeyboardAnalysisPanel:
    def test_rhythm_waits_for_enough_key_intervals(self, qapp, tmp_path):
        panel = KeyboardAnalysisPanel()
        panel.begin_session_text_recording(tmp_path / "session" / "typed_text.txt")

        QTest.keyClicks(panel.input_field, "abc", delay=30)
        assert panel.rhythm_label.text() == "Rhythm Regularity: collecting timing"

        QTest.keyClicks(panel.input_field, "d", delay=30)
        assert panel.rhythm_label.text().startswith("Rhythm Regularity: ")
        assert panel.rhythm_label.text() != "Rhythm Regularity: collecting timing"

    def test_session_text_is_saved_while_metrics_update(self, qapp, tmp_path):
        panel = KeyboardAnalysisPanel()
        text_path = tmp_path / "session" / "typed_text.txt"
        panel.begin_session_text_recording(text_path)

        QTest.keyClicks(panel.input_field, "abcd", delay=30)
        assert text_path.read_text(encoding="utf-8") == "abcd"
        QTest.keyClick(panel.input_field, Qt.Key.Key_Backspace)

        assert panel._feature_extractor._total_keys == 5
        assert panel.input_field.text() == "abc"
        assert text_path.read_text(encoding="utf-8") == "abc"
        assert panel.typing_speed_label.text() != "Typing Speed: --"
        assert panel.iki_label.text() != "Inter-Key Interval: --"
        assert panel.correction_rate_label.text() != "Correction Rate: --"
        assert panel.sample_progress_label.text() == "Sample Progress: 25%"
        assert panel.rhythm_label.text().startswith("Rhythm Regularity:")
        assert panel.state_label.text() == "State: typing_pattern_observed"

    def test_session_pause_resume_and_stop_preserve_text(self, qapp, tmp_path):
        panel = KeyboardAnalysisPanel()
        text_path = tmp_path / "session" / "typed_text.txt"
        panel.begin_session_text_recording(text_path)
        QTest.keyClicks(panel.input_field, "session", delay=10)

        panel.set_session_text_recording_active(False)
        assert not panel.input_field.isEnabled()

        panel.set_session_text_recording_active(True)
        assert panel.input_field.isEnabled()
        panel.finish_session_text_recording()

        assert not panel.input_field.isEnabled()
        assert panel.input_field.text() == "session"
        assert text_path.read_text(encoding="utf-8") == "session"
        assert "saved to typed_text.txt" in panel.input_status_label.text()


class TestMainWindow:
    """Tests for MainWindow."""

    def test_creation(self):
        """Test MainWindow creation."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        
        from ui.main_window import MainWindow
        from core.config import get_default_config
        
        # Mock config loading
        import ui.main_window as main_window_module
        original_load_config = main_window_module.load_config
        
        def mock_load_config(path):
            from core.config import get_default_config
            config = get_default_config()
            config.storage.output_directory = "/tmp/test_sessions"
            return config
        
        main_window_module.load_config = mock_load_config
        
        try:
            window = MainWindow()
            assert window is not None
            assert window.windowTitle() == "Multimodal Human State Capture"
        finally:
            main_window_module.load_config = original_load_config
            if 'window' in locals():
                window.close()

    def test_state_changed(self, qapp):
        """Test state change handling."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        
        from ui.main_window import MainWindow
        from core.config import get_default_config
        
        import ui.main_window as main_window_module
        original_load_config = main_window_module.load_config
        
        def mock_load_config(path):
            config = get_default_config()
            config.storage.output_directory = "/tmp/test_sessions"
            return config
        
        main_window_module.load_config = mock_load_config
        
        try:
            window = MainWindow()
            
            # Test state changes
            window._on_state_changed("recording")
            assert window.overall_status._status == "recording"
            
            window._on_state_changed("paused")
            assert window.overall_status._status == "paused"
            
            window._on_state_changed("stopped")
            assert window.overall_status._status == "stopped"
            
        finally:
            main_window_module.load_config = original_load_config
            window.close()


# Need imports
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestCameraPreviewWidget:
    """Tests for CameraPreviewWidget."""

    def test_creation(self, qapp):
        """Test CameraPreviewWidget creation and default state."""
        from ui.main_window import CameraPreviewWidget
        
        widget = CameraPreviewWidget()
        assert widget is not None
        assert widget.width() == 160
        assert widget.height() == 120
        assert widget.text() == "Camera is off"
        assert widget.pixmap().isNull()

    def test_update_frame(self, qapp):
        """Test updating preview with a valid frame."""
        from ui.main_window import CameraPreviewWidget
        
        widget = CameraPreviewWidget()
        widget.show()
        
        # Create a test frame (BGR format)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Blue channel
        frame[:, :, 1] = 128  # Green channel
        frame[:, :, 2] = 64   # Red channel
        
        widget.update_frame(frame)
        
        # Verify pixmap was set
        assert not widget.pixmap().isNull()
        assert widget.pixmap().width() <= 160
        assert widget.pixmap().height() <= 120

    def test_clear(self, qapp):
        """Test clearing preview restores placeholder."""
        from ui.main_window import CameraPreviewWidget
        
        widget = CameraPreviewWidget()
        widget.show()
        
        # Set a frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        widget.update_frame(frame)
        assert not widget.pixmap().isNull()
        
        # Clear
        widget.clear()
        
        # Verify placeholder is shown
        assert widget.text() == "Camera is off"
        assert widget.pixmap().isNull()

    def test_update_frame_handles_invalid_input(self, qapp):
        """Test update_frame handles invalid input gracefully."""
        from ui.main_window import CameraPreviewWidget
        
        widget = CameraPreviewWidget()
        widget.show()
        
        # Should not crash on None or empty array
        widget.update_frame(None)
        widget.update_frame(np.array([]))
        
        # Placeholder should still be shown
        assert widget.text() == "Camera is off"


class TestPreviewFrameEmitter:
    """Tests for PreviewFrameEmitter signal emitter."""

    def test_signal_emission(self, qapp):
        """Test frame_ready signal can be emitted and connected."""
        from ui.main_window import PreviewFrameEmitter
        
        emitter = PreviewFrameEmitter()
        received_frames = []
        
        def on_frame(frame):
            received_frames.append(frame)
        
        emitter.frame_ready.connect(on_frame)
        
        # Emit a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        emitter.frame_ready.emit(test_frame)
        
        assert len(received_frames) == 1
        assert np.array_equal(received_frames[0], test_frame)


class TestMainWindowPreviewIntegration:
    """Integration tests for camera preview in MainWindow."""

    def test_preview_widget_exists_after_init(self, qapp):
        """Test MainWindow creates camera preview widget."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        
        from ui.main_window import MainWindow
        from core.config import get_default_config
        import ui.main_window as main_window_module
        
        original_load_config = main_window_module.load_config
        
        def mock_load_config(path):
            config = get_default_config()
            config.storage.output_directory = "/tmp/test_sessions"
            return config
        
        main_window_module.load_config = mock_load_config
        
        try:
            window = MainWindow()
            assert hasattr(window, 'camera_preview')
            assert window.camera_preview is not None
            assert not window.camera_preview.isVisible()  # Hidden by default
        finally:
            main_window_module.load_config = original_load_config
            window.close()

    def test_preview_hidden_at_startup(self, qapp):
        """Test preview is hidden at application startup (camera off)."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        
        from ui.main_window import MainWindow
        from core.config import get_default_config
        import ui.main_window as main_window_module
        
        original_load_config = main_window_module.load_config
        
        def mock_load_config(path):
            config = get_default_config()
            config.storage.output_directory = "/tmp/test_sessions"
            return config
        
        main_window_module.load_config = mock_load_config
        
        try:
            window = MainWindow()
            assert not window.camera_preview.isVisible()
            assert window.camera_preview.text() == "Camera is off"
        finally:
            main_window_module.load_config = original_load_config
            window.close()

    def test_setup_camera_preview_shows_widget(self, qapp):
        """Test _setup_camera_preview makes preview visible."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        
        from ui.main_window import MainWindow
        from core.config import get_default_config
        import ui.main_window as main_window_module
        
        original_load_config = main_window_module.load_config
        
        def mock_load_config(path):
            config = get_default_config()
            config.storage.output_directory = "/tmp/test_sessions"
            return config
        
        main_window_module.load_config = mock_load_config
        
        try:
            window = MainWindow()
            window.show()  # Need to show for geometry calculations
            
            # Mock face capture with set_preview_callback
            mock_face_capture = MagicMock()
            window.session_manager._modalities["face"] = mock_face_capture
            
            window._setup_camera_preview()
            
            # Preview should be visible
            assert window.camera_preview.isVisible()
            # Face capture should have preview callback set
            mock_face_capture.set_preview_callback.assert_called_once()
        finally:
            main_window_module.load_config = original_load_config
            window.close()

    def test_cleanup_camera_preview_hides_widget(self, qapp):
        """Test _cleanup_camera_preview hides preview and clears callback."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        
        from ui.main_window import MainWindow
        from core.config import get_default_config
        import ui.main_window as main_window_module
        
        original_load_config = main_window_module.load_config
        
        def mock_load_config(path):
            config = get_default_config()
            config.storage.output_directory = "/tmp/test_sessions"
            return config
        
        main_window_module.load_config = mock_load_config
        
        try:
            window = MainWindow()
            window.show()
            
            # Mock face capture with set_preview_callback
            mock_face_capture = MagicMock()
            window.session_manager._modalities["face"] = mock_face_capture
            
            # Set up preview first
            window._setup_camera_preview()
            assert window.camera_preview.isVisible()
            
            # Clean up
            window._cleanup_camera_preview()
            
            # Preview should be hidden and cleared
            assert not window.camera_preview.isVisible()
            assert window.camera_preview.text() == "Camera is off"
            # Callback should be cleared (called with None)
            assert mock_face_capture.set_preview_callback.call_count == 2
            mock_face_capture.set_preview_callback.assert_called_with(None)
        finally:
            main_window_module.load_config = original_load_config
            window.close()

    def test_on_preview_frame_ready_updates_widget(self, qapp):
        """Test _on_preview_frame_ready slot updates preview widget."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        
        from ui.main_window import MainWindow
        from core.config import get_default_config
        import ui.main_window as main_window_module
        
        original_load_config = main_window_module.load_config
        
        def mock_load_config(path):
            config = get_default_config()
            config.storage.output_directory = "/tmp/test_sessions"
            return config
        
        main_window_module.load_config = mock_load_config
        
        try:
            window = MainWindow()
            window.show()
            
            # Create test frame
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            test_frame[:, :, 0] = 255  # Blue
            
            # Call slot directly
            window._on_preview_frame_ready(test_frame)
            
            # Preview should show frame
            assert not window.camera_preview.pixmap().isNull()
        finally:
            main_window_module.load_config = original_load_config
            window.close()

    def test_on_preview_frame_ready_handles_invalid_input(self, qapp):
        """Test _on_preview_frame_ready handles invalid input gracefully."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        
        from ui.main_window import MainWindow
        from core.config import get_default_config
        import ui.main_window as main_window_module
        
        original_load_config = main_window_module.load_config
        
        def mock_load_config(path):
            config = get_default_config()
            config.storage.output_directory = "/tmp/test_sessions"
            return config
        
        main_window_module.load_config = mock_load_config
        
        try:
            window = MainWindow()
            window.show()
            
            # Should not crash on invalid input
            window._on_preview_frame_ready(None)
            window._on_preview_frame_ready("not a frame")
            window._on_preview_frame_ready(np.array([]))
            
            # Placeholder should remain
            assert window.camera_preview.text() == "Camera is off"
        finally:
            main_window_module.load_config = original_load_config
            window.close()

    def test_resize_event_repositions_preview(self, qapp):
        """Test resizeEvent repositions preview widget."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        
        from ui.main_window import MainWindow
        from core.config import get_default_config
        from PySide6.QtGui import QResizeEvent
        from PySide6.QtCore import QSize
        import ui.main_window as main_window_module
        
        original_load_config = main_window_module.load_config
        
        def mock_load_config(path):
            config = get_default_config()
            config.storage.output_directory = "/tmp/test_sessions"
            return config
        
        main_window_module.load_config = mock_load_config
        
        try:
            window = MainWindow()
            window.show()
            
            # Mock face capture
            mock_face_capture = MagicMock()
            window.session_manager._modalities["face"] = mock_face_capture
            
            window._setup_camera_preview()
            initial_pos = window.camera_preview.pos()
            
            # Simulate resize event with proper QResizeEvent
            resize_event = QResizeEvent(QSize(1000, 800), QSize(800, 600))
            window.resizeEvent(resize_event)
            
            assert window.camera_preview.isVisible()
        finally:
            main_window_module.load_config = original_load_config
            window.close()