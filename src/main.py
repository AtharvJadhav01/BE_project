"""
Main entry point for the Multimodal Human State Capture Application.
"""

import sys
import signal
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent))

from core.config import load_config
from core.logging import get_logger, setup_logging
from session.manager import SessionManager, get_session_manager
from keyboard import KeyboardCapture
from audio import AudioCapture
from face import FaceCapture

logger = get_logger(__name__)


class Application:
    """Main application class."""

    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path(__file__).parent.parent / "config" / "config.yaml"
        self.session_manager: SessionManager = None
        self._running = False

    def initialize(self) -> bool:
        """Initialize the application."""
        try:
            # Load configuration
            config = load_config(self.config_path)

            # Setup logging
            setup_logging(config.logging)

            logger.info("application_initializing", version="0.1.0")

            # Create session manager
            self.session_manager = SessionManager(
                config=config,
                application_version="0.1.0"
            )

            # Register real modalities
            self._register_modalities()

            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            logger.info("application_initialized")
            return True

        except Exception as e:
            logger.error("application_initialization_failed", error=str(e))
            return False

    def _register_modalities(self) -> None:
        """Register real modality implementations."""
        # Register keyboard capture
        if self.session_manager.config.keyboard.enabled:
            keyboard_capture = KeyboardCapture(self.session_manager.config.keyboard.__dict__)
            # Set unified event callback for session manager
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

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        logger.info("shutdown_signal_received", signal=signum)
        self._running = False
        if self.session_manager and self.session_manager.state.value in ("recording", "paused"):
            self.session_manager.stop_session()

    def run(self) -> int:
        """Run the main application loop."""
        if not self.initialize():
            return 1

        self._running = True
        logger.info("application_running")

        # For now, just wait for signal
        # In the future, this will be the main event loop
        try:
            while self._running:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("keyboard_interrupt_received")

        logger.info("application_shutting_down")
        return 0


def main():
    """Main application entry point."""
    app = Application()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())