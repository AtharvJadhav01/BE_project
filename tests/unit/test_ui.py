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
from pathlib import Path
from unittest.mock import patch, MagicMock