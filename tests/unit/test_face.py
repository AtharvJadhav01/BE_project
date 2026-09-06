"""
Unit tests for face capture module.
"""

import pytest
import tempfile
import time
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.events import Event, EventType, ModalityType
from core.timestamps import TimestampGenerator
from face.camera import FaceCamera
from face.detector import FaceDetector, FaceDetectionResult
from face.processor import FaceProcessor, ProcessedFaceObservation
from face.storage import FaceStorage
from face.capture import FaceCapture


class TestFaceCamera:
    """Tests for FaceCamera class."""

    @pytest.fixture
    def timestamp_generator(self):
        """Create a timestamp generator."""
        gen = TimestampGenerator()
        gen.start_session()
        return gen

    @pytest.fixture
    def mock_callback(self):
        """Create a mock event callback."""
        return Mock()

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            "fps": 15,
            "resolution": [640, 480],
            "device": 0,
        }

    def test_camera_creation(self, timestamp_generator, mock_callback, config):
        """Test camera creation."""
        camera = FaceCamera(
            session_id="test-session",
            timestamp_generator=timestamp_generator,
            event_callback=mock_callback,
            config=config
        )
        assert camera.session_id == "test-session"
        assert not camera.is_running
        assert not camera.is_paused

    def test_check_camera_availability(self):
        """Test camera availability check."""
        # This will test with the actual system camera
        # We'll just verify the method exists and returns a boolean
        result = FaceCamera.check_camera_availability(0)
        assert isinstance(result, bool)

    def test_get_available_cameras(self):
        """Test getting available cameras."""
        cameras = FaceCamera.get_available_cameras()
        assert isinstance(cameras, list)
        for cam in cameras:
            assert "index" in cam
            assert "width" in cam
            assert "height" in cam


class TestFaceDetector:
    """Tests for FaceDetector class."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            "detection_model": "mediapipe",
            "landmark_model": "mediapipe",
            "refine_landmarks": True,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        }

    @pytest.fixture
    def detector(self, config):
        """Create a detector instance."""
        return FaceDetector(config)

    def test_detector_creation(self, detector):
        """Test detector creation."""
        assert detector is not None
        assert detector._face_mesh is None  # Not initialized until first use

    def test_process_frame_no_face(self, detector):
        """Test processing a frame with no face."""
        # Create a blank frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = detector.process_frame(frame)
        
        assert isinstance(result, FaceDetectionResult)
        assert result.face_detected is False
        assert result.detection_confidence == 0.0
        assert result.processing_time_ms > 0

    def test_close(self, detector):
        """Test closing the detector."""
        detector.close()
        assert detector._face_mesh is None


class TestFaceProcessor:
    """Tests for FaceProcessor class."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {}

    @pytest.fixture
    def processor(self, config):
        """Create a processor instance."""
        return FaceProcessor(config)

    @pytest.fixture
    def mock_detection_result(self):
        """Create a mock detection result."""
        return FaceDetectionResult(
            face_detected=True,
            detection_confidence=0.95,
            bounding_box={"x": 100, "y": 100, "width": 200, "height": 200},
            landmarks_3d=[{"x": 0.5, "y": 0.5, "z": 0.0}] * 468,
            landmarks_2d=[{"x": 320, "y": 240}] * 468,
            blendshapes={"mouthSmileLeft": 0.1, "mouthSmileRight": 0.1},
            iris={"left": {"x": 0.4, "y": 0.3, "z": 0.0}, "right": {"x": 0.6, "y": 0.3, "z": 0.0}},
            head_pose={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            processing_time_ms=10.5
        )

    def test_processor_start_stop(self, processor):
        """Test processor start/stop."""
        assert processor.start() is True
        assert processor._processing is True
        
        assert processor.stop() is True
        assert processor._processing is False

    def test_process_observation(self, processor, mock_detection_result):
        """Test processing an observation."""
        processor.start()
        
        # Set up mock storage callback
        storage_callback = Mock()
        processor.set_storage_callback(storage_callback)
        
        gen = TimestampGenerator()
        gen.start_session()
        timestamp_monotonic = gen.now_monotonic()
        timestamp_wallclock = gen.now_wallclock()
        
        # Process observation
        result = processor.process_observation(
            result=mock_detection_result,
            frame_index=1,
            timestamp_monotonic=timestamp_monotonic,
            timestamp_wallclock=timestamp_wallclock
        )
        assert result is True
        
        # Wait for processing
        time.sleep(0.1)
        
        # Verify callback was called
        storage_callback.assert_called_once()
        
        processor.stop()

    def test_face_state_tracking(self, processor, mock_detection_result):
        """Test face found/lost state tracking."""
        processor.start()
        
        # Set up mock unified event callback
        unified_callback = Mock()
        processor.set_unified_event_callback(unified_callback)
        
        gen = TimestampGenerator()
        gen.start_session()
        timestamp_monotonic = gen.now_monotonic()
        timestamp_wallclock = gen.now_wallclock()
        
        # First frame with face
        processor.process_observation(
            result=mock_detection_result,
            frame_index=1,
            timestamp_monotonic=timestamp_monotonic,
            timestamp_wallclock=timestamp_wallclock
        )
        time.sleep(0.1)
        
        # Should have emitted face_found and face_observation
        assert unified_callback.call_count >= 1
        
        # Second frame without face
        no_face_result = FaceDetectionResult(
            face_detected=False,
            detection_confidence=0.0,
            bounding_box=None,
            landmarks_3d=None,
            landmarks_2d=None,
            blendshapes=None,
            iris=None,
            head_pose=None,
            processing_time_ms=5.0
        )
        
        processor.process_observation(
            result=no_face_result,
            frame_index=2,
            timestamp_monotonic=timestamp_monotonic + 0.1,
            timestamp_wallclock=timestamp_wallclock
        )
        time.sleep(0.1)
        
        # Should have emitted face_lost and face_observation
        assert unified_callback.call_count >= 2
        
        processor.stop()

    def test_stats(self, processor, mock_detection_result):
        """Test processor statistics."""
        processor.start()
        
        gen = TimestampGenerator()
        gen.start_session()
        timestamp_monotonic = gen.now_monotonic()
        timestamp_wallclock = gen.now_wallclock()
        
        processor.process_observation(
            result=mock_detection_result,
            frame_index=1,
            timestamp_monotonic=timestamp_monotonic,
            timestamp_wallclock=timestamp_wallclock
        )
        time.sleep(0.1)
        
        stats = processor.get_stats()
        assert stats["observations_processed"] == 1
        assert stats["observations_dropped"] == 0
        
        processor.stop()


class TestFaceStorage:
    """Tests for FaceStorage class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {}

    @pytest.fixture
    def timestamp_generator(self):
        """Create a timestamp generator."""
        gen = TimestampGenerator()
        gen.start_session()
        return gen

    @pytest.fixture
    def storage(self, temp_dir, config, timestamp_generator):
        """Create storage instance."""
        return FaceStorage(temp_dir / "test-session", config, timestamp_generator)

    @pytest.fixture
    def sample_observation(self):
        """Create a sample observation."""
        return ProcessedFaceObservation(
            observation_id="face_000001",
            frame_index=1,
            timestamp_monotonic=0.0,
            timestamp_wallclock="2024-01-01T00:00:00.000Z",
            face_detected=True,
            detection_confidence=0.95,
            bounding_box={"x": 100, "y": 100, "width": 200, "height": 200},
            landmarks_3d=[{"x": 0.5, "y": 0.5, "z": 0.0}],
            landmarks_2d=[{"x": 320, "y": 240}],
            blendshapes={"mouthSmileLeft": 0.1},
            iris={"left": {"x": 0.4, "y": 0.3, "z": 0.0}, "right": {"x": 0.6, "y": 0.3, "z": 0.0}},
            head_pose={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            processing_time_ms=10.5
        )

    def test_storage_start_stop(self, storage):
        """Test storage start/stop."""
        assert storage.start() is True
        assert storage._writing is True
        
        assert storage.stop() is True
        assert storage._writing is False

    def test_write_observation(self, storage, sample_observation):
        """Test writing an observation."""
        storage.start()
        
        result = storage.write_observation(sample_observation)
        assert result is True
        
        # Wait for write
        time.sleep(0.1)
        
        # Check file was created
        file_path = storage._file_path
        assert file_path.exists()
        
        # Read and verify content
        import orjson
        with open(file_path, 'rb') as f:
            line = f.readline()
            data = orjson.loads(line)
            assert data["observation_id"] == "face_000001"
            assert data["face_detected"] is True
            assert data["detection_confidence"] == 0.95
        
        storage.stop()

    def test_multiple_observations(self, storage, sample_observation):
        """Test writing multiple observations."""
        storage.start()
        
        # Write multiple observations
        for i in range(3):
            obs = ProcessedFaceObservation(
                observation_id=f"face_{i+1:06d}",
                frame_index=i+1,
                timestamp_monotonic=float(i) * 0.1,
                timestamp_wallclock="2024-01-01T00:00:00.000Z",
                face_detected=True,
                detection_confidence=0.9,
                bounding_box={"x": 100, "y": 100, "width": 200, "height": 200},
                landmarks_3d=[{"x": 0.5, "y": 0.5, "z": 0.0}],
                landmarks_2d=[{"x": 320, "y": 240}],
                blendshapes={"mouthSmileLeft": 0.1},
                iris={"left": {"x": 0.4, "y": 0.3, "z": 0.0}, "right": {"x": 0.6, "y": 0.3, "z": 0.0}},
                head_pose={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                processing_time_ms=10.0
            )
            storage.write_observation(obs)
        
        time.sleep(0.2)
        
        # Verify all observations written
        import orjson
        with open(storage._file_path, 'rb') as f:
            lines = f.readlines()
            assert len(lines) == 3
        
        stats = storage.get_stats()
        assert stats["observations_written"] == 3
        
        storage.stop()


class TestFaceCapture:
    """Tests for FaceCapture (modality implementation)."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            "enabled": True,
            "detection_model": "mediapipe",
            "landmark_model": "mediapipe",
            "refine_landmarks": True,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "camera": {
                "enabled": True,
                "fps": 15,
                "resolution": [640, 480],
                "device": 0,
            },
            "storage": {
                "output_directory": "data/sessions"
            }
        }

    @pytest.fixture
    def capture(self, config):
        """Create face capture instance."""
        return FaceCapture(config)

    def test_capture_creation(self, capture):
        """Test capture creation."""
        assert capture.name == "face"
        assert capture.status.value == "stopped"

    def test_capture_start_stop(self, capture, temp_dir):
        """Test capture start/stop."""
        config = capture.config.copy()
        config["camera"]["enabled"] = True
        config["storage"]["output_directory"] = str(temp_dir)
        
        # Mock camera availability
        with patch('face.capture.FaceCamera.check_camera_availability') as mock_check:
            mock_check.return_value = True
            
            # Mock OpenCV VideoCapture
            with patch('cv2.VideoCapture') as mock_cv_cap:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
                mock_cap.get.return_value = 15.0
                mock_cv_cap.return_value = mock_cap
                
                # Mock MediaPipe FaceLandmarker
                with patch('mediapipe.tasks.python.vision.FaceLandmarker') as mock_face_landmarker:
                    mock_landmarker = MagicMock()
                    mock_landmarker.detect_async = MagicMock()
                    mock_landmarker.close = MagicMock()
                    mock_face_landmarker.create_from_options.return_value = mock_landmarker
                    
                    # Mock the result callback behavior
                    def mock_detect_async(mp_image, timestamp_ms):
                        # Simulate async callback
                        import time
                        time.sleep(0.001)
                    mock_landmarker.detect_async.side_effect = mock_detect_async
                    
                    # Mock unified event callback
                    unified_callback = Mock()
                    capture.set_unified_event_callback(unified_callback)
                    
                    # Start capture
                    result = capture.start("test-session", temp_dir)
                    assert result is True
                    assert capture.status.value == "running"
                    
                    # Stop capture
                    result = capture.stop()
                    assert result is True
                    assert capture.status.value == "stopped"

    def test_capture_pause_resume(self, capture, temp_dir):
        """Test capture pause/resume."""
        config = capture.config.copy()
        config["camera"]["enabled"] = True
        config["storage"]["output_directory"] = str(temp_dir)
        
        with patch('face.capture.FaceCamera.check_camera_availability') as mock_check:
            mock_check.return_value = True
            
            with patch('cv2.VideoCapture') as mock_cv_cap:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
                mock_cap.get.return_value = 15.0
                mock_cv_cap.return_value = mock_cap
                
                with patch('mediapipe.tasks.python.vision.FaceLandmarker') as mock_face_landmarker:
                    mock_landmarker = MagicMock()
                    mock_landmarker.detect_async = MagicMock()
                    mock_landmarker.close = MagicMock()
                    mock_face_landmarker.create_from_options.return_value = mock_landmarker
                    
                    def mock_detect_async(mp_image, timestamp_ms):
                        import time
                        time.sleep(0.001)
                    mock_landmarker.detect_async.side_effect = mock_detect_async
                    
                    unified_callback = Mock()
                    capture.set_unified_event_callback(unified_callback)
                    
                    capture.start("test-session", temp_dir)
                    
                    # Pause
                    result = capture.pause()
                    assert result is True
                    assert capture.status.value == "paused"
                    
                    # Resume
                    result = capture.resume()
                    assert result is True
                    assert capture.status.value == "running"
                    
                    capture.stop()

    def test_get_stats(self, capture, temp_dir):
        """Test getting capture statistics."""
        config = capture.config.copy()
        config["camera"]["enabled"] = True
        config["storage"]["output_directory"] = str(temp_dir)
        
        with patch('face.capture.FaceCamera.check_camera_availability') as mock_check:
            mock_check.return_value = True
            
            with patch('cv2.VideoCapture') as mock_cv_cap:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
                mock_cap.get.return_value = 15.0
                mock_cv_cap.return_value = mock_cap
                
                with patch('mediapipe.tasks.python.vision.FaceLandmarker') as mock_face_landmarker:
                    mock_landmarker = MagicMock()
                    mock_landmarker.detect_async = MagicMock()
                    mock_landmarker.close = MagicMock()
                    mock_face_landmarker.create_from_options.return_value = mock_landmarker
                    
                    def mock_detect_async(mp_image, timestamp_ms):
                        import time
                        time.sleep(0.001)
                    mock_landmarker.detect_async.side_effect = mock_detect_async
                    
                    unified_callback = Mock()
                    capture.set_unified_event_callback(unified_callback)
                    
                    capture.start("test-session", temp_dir)
                    
                    stats = capture.get_stats()
                    assert "status" in stats
                    assert "processor" in stats
                    assert "storage" in stats
                    assert "camera" in stats
                    
                    capture.stop()


class TestFaceIntegration:
    """Integration tests for face capture with session manager."""

    def test_face_with_session_manager(self, temp_dir):
        """Test face capture integrated with session manager."""
        from core.config import get_default_config
        from session.manager import SessionManager
        
        config = get_default_config()
        config.storage.output_directory = str(temp_dir)
        config.face.enabled = True
        config.camera.enabled = True
        config.keyboard.enabled = False
        config.audio.enabled = False
        
        with patch('face.capture.FaceCamera.check_camera_availability') as mock_check:
            mock_check.return_value = True
            
            with patch('cv2.VideoCapture') as mock_cv_cap:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
                mock_cap.get.return_value = 15.0
                mock_cv_cap.return_value = mock_cap
                
                with patch('mediapipe.tasks.python.vision.FaceLandmarker') as mock_face_landmarker:
                    mock_landmarker = MagicMock()
                    mock_landmarker.detect_async = MagicMock()
                    mock_landmarker.close = MagicMock()
                    mock_face_landmarker.create_from_options.return_value = mock_landmarker
                    
                    def mock_detect_async(mp_image, timestamp_ms):
                        import time
                        time.sleep(0.001)
                    mock_landmarker.detect_async.side_effect = mock_detect_async
                    
                    manager = SessionManager(config=config, application_version="test-1.0.0")
                    
                    # Register face capture
                    from face import FaceCapture
                    face_capture = FaceCapture(config.face.__dict__)
                    face_capture.set_unified_event_callback(
                        lambda event: manager._emit_event(event)
                    )
                    manager.register_modality(face_capture)
                    
                    # Start session
                    result = manager.start_session()
                    assert result is True
                    assert manager.state.value == "recording"
                    
                    # Verify face modality is running
                    assert "face" in manager.modalities
                    assert manager.modalities["face"].status.value == "running"
                    
                    # Wait a bit
                    time.sleep(0.2)
                    
                    # Stop session
                    result = manager.stop_session()
                    assert result is True
                    assert manager.state.value == "stopped"
                    
                    # Verify session directory was created
                    session_dir = temp_dir / manager.session_id
                    assert session_dir.exists()
                    
                    # Face features file should be created by storage
                    face_features_file = session_dir / "facial_features.jsonl"
                    assert face_features_file.exists()


# Need temp_dir fixture
@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)