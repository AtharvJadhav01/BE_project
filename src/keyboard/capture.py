"""
Keyboard modality implementation.

Implements the KeyboardModalityInterface using pynput for global keyboard capture.
"""

import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from core.events import Event, EventType, ModalityType, create_keyboard_event
from core.interfaces import KeyboardModalityInterface, ModalityStatus
from core.logging import get_logger
from core.timestamps import TimestampGenerator

from .listener import KeyboardListener
from .processor import KeyboardProcessor
from .storage import KeyboardStorage

# Analysis imports
from analysis.keyboard.features import KeyboardFeatureExtractor
from analysis.keyboard.state import KeyboardStateEstimator
from analysis.keyboard.baseline import KeyboardBaselineManager
from analysis.temporal import TemporalWindow
from analysis.results import FeatureVector, StateEstimate, AnalysisResult
import orjson


class KeyboardCapture(KeyboardModalityInterface):
    """
    Keyboard capture modality.
    
    Captures global keyboard events (key_down, key_up) with timestamps
    and stores them in the session directory.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = get_logger(__name__)
        
        # Components
        self._listener: Optional[KeyboardListener] = None
        self._processor: Optional[KeyboardProcessor] = None
        self._storage: Optional[KeyboardStorage] = None
        
        # Analysis components
        self._feature_extractor: Optional[KeyboardFeatureExtractor] = None
        self._baseline_manager: Optional[KeyboardBaselineManager] = None
        self._temporal_windows: Optional[Dict[str, TemporalWindow]] = None
        self._state_estimator: Optional[KeyboardStateEstimator] = None
        self._analysis_storage: Optional[Any] = None  # Will be analysis storage writer
        
        # Session state
        self._session_id: Optional[str] = None
        self._session_dir: Optional[Path] = None
        self._timestamp_generator: Optional[TimestampGenerator] = None
        self._unified_event_callback: Optional[callable] = None
        
        # Analysis state
        self._last_analysis_result: Optional[Dict[str, Any]] = None
        self._analysis_chunk_count = 0
        self._last_analysis_timestamp: Optional[float] = None
        
        # Thread safety
        self._lock = threading.Lock()

    def set_unified_event_callback(self, callback: callable) -> None:
        """Set callback for unified event pipeline."""
        self._unified_event_callback = callback

    def _on_storage_write(self, event) -> None:
        """Callback for writing events to storage."""
        if self._storage:
            self._storage.write_event(event)

    def _on_keyboard_event(self, event: Event) -> None:
        """Callback for keyboard events from listener."""
        # Forward to processor
        if self._processor:
            self._processor.process_event(event)
        
        # Forward to unified event pipeline
        if self._unified_event_callback:
            self._unified_event_callback(event)

    def _initialize_analysis_components(self) -> None:
        """Initialize keyboard analysis components."""
        try:
            # Get keyboard analysis config
            keyboard_config = self.config.get("analysis", {}).get("keyboard", {})
            analysis_config = self.config.get("analysis", {})
            
            # Initialize feature extractor
            self._feature_extractor = KeyboardFeatureExtractor(
                "keyboard_features", 
                keyboard_config
            )
            
            # Initialize baseline manager
            self._baseline_manager = KeyboardBaselineManager(
                warmup_duration_sec=analysis_config.get("baseline_warmup_sec", 10.0),
                min_samples=analysis_config.get("baseline_min_samples", 50),
                ema_alpha=analysis_config.get("ema_alpha", 0.1),
                min_std=analysis_config.get("min_std", 1e-6)
            )
            
            # Initialize temporal windows for key keyboard features
            key_features = [
                "typing_speed_kpm", "typing_speed_kps",
                "iki_mean", "iki_std", "iki_cv",
                "key_duration_mean", "key_duration_std",
                "correction_count", "correction_rate", "correction_burst_count",
                "pause_count", "pause_duration_mean", "long_pause_count",
                "typing_rhythm_regularity", "burst_typing_score", "hesitation_score",
                "total_keys", "printable_keys", "modifier_usage"
            ]
            
            window_size = analysis_config.get("temporal_window_sec", 3.0)
            self._temporal_windows = {
                feature: TemporalWindow(window_duration_sec=window_size)
                for feature in key_features
            }
            
            # Initialize state estimator
            self._state_estimator = KeyboardStateEstimator(
                "keyboard_state", 
                keyboard_config
            )
            
            # Initialize analysis storage
            self._setup_analysis_storage()
            
            self.logger.info("keyboard_analysis_components_initialized")
            
        except Exception as e:
            self.logger.error("keyboard_analysis_init_failed", error=str(e))
            raise

    def _setup_analysis_storage(self) -> None:
        """Set up analysis storage for keyboard features and states."""
        try:
            # Create analysis directory
            analysis_dir = self._session_dir / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # Open files for appending
            self._analysis_features_path = analysis_dir / "keyboard_features.jsonl"
            self._analysis_states_path = analysis_dir / "keyboard_states.jsonl"
            
            self._analysis_features_file = open(self._analysis_features_path, "ab")
            self._analysis_states_file = open(self._analysis_states_path, "ab")
            
            self.logger.info("keyboard_analysis_storage_setup", 
                           features_path=str(self._analysis_features_path),
                           states_path=str(self._analysis_states_path))
                           
        except Exception as e:
            self.logger.error("keyboard_analysis_storage_setup_failed", error=str(e))
            raise

    def _analysis_start(self) -> bool:
        """Start analysis components."""
        try:
            self._analysis_ready = True
            self._analysis_chunk_count = 0
            self._last_analysis_timestamp = None
            self.logger.info("keyboard_analysis_started")
            return True
        except Exception as e:
            self.logger.error("keyboard_analysis_start_failed", error=str(e))
            return False

    def _analysis_stop(self) -> bool:
        """Stop analysis components and close files."""
        try:
            # Close analysis files
            if hasattr(self, '_analysis_features_file') and self._analysis_features_file:
                self._analysis_features_file.close()
            if hasattr(self, '_analysis_states_file') and self._analysis_states_file:
                self._analysis_states_file.close()
            
            self.logger.info("keyboard_analysis_stopped", 
                           chunks_processed=self._analysis_chunk_count)
            return True
        except Exception as e:
            self.logger.error("keyboard_analysis_stop_failed", error=str(e))
            return False

    def _on_keyboard_event(self, event: Event) -> None:
        """Callback for keyboard events from listener."""
        # Forward to processor
        if self._processor:
            self._processor.process_event(event)
        
        # Forward to unified event pipeline
        if self._unified_event_callback:
            self._unified_event_callback(event)

    def _on_analysis_event(self, event: Event) -> None:
        """Callback for running keyboard analysis on events."""
        if not self._feature_extractor or not self._baseline_manager or not self._state_estimator:
            return
        
        try:
            data = event.data
            event_type = data.get("event_type", "")
            
            # Only analyze key events
            if event_type not in ("key_down", "key_up"):
                return
            
            timestamp_monotonic = event.timestamp.monotonic
            timestamp_wallclock = event.timestamp.wallclock
            
            # Run keyboard analysis
            analysis_result = self._run_keyboard_analysis(
                data, timestamp_monotonic, timestamp_wallclock
            )
            
            if analysis_result is not None:
                self._store_analysis_results(analysis_result)
                self._analysis_chunk_count += 1
                self._last_analysis_timestamp = timestamp_monotonic
                self._last_analysis_result = analysis_result
                
        except Exception as e:
            self.logger.error("keyboard_analysis_callback_failed", error=str(e))

    def _run_keyboard_analysis(self, event_data: Dict[str, Any],
                              timestamp_monotonic: float,
                              timestamp_wallclock: str) -> Optional[Dict[str, Any]]:
        """
        Run keyboard analysis pipeline on event data.
        
        Args:
            event_data: Raw keyboard event data
            timestamp_monotonic: Monotonic timestamp
            timestamp_wallclock: Wallclock timestamp
            
        Returns:
            Dictionary containing analysis results or None if analysis failed
        """
        try:
            # Extract features from event
            raw_features = self._extract_raw_features_from_event(event_data)
            
            if not raw_features:
                return None
            
            # Update temporal windows
            self._update_temporal_windows(raw_features, timestamp_monotonic)
            
            # Get temporal window statistics
            window_stats = self._get_temporal_window_stats()
            
            # Update baseline with window statistics
            self._update_baseline(window_stats, timestamp_monotonic)
            
            # Normalize features using baseline
            normalized_features = self._normalize_features(window_stats)
            
            # Estimate keyboard state
            state_estimate = self._estimate_keyboard_state(normalized_features)
            
            # Create analysis result
            analysis_result = {
                "timestamp_monotonic": timestamp_monotonic,
                "timestamp_wallclock": timestamp_wallclock,
                "event_id": 0,  # Will be filled by caller if needed
                "raw_features": raw_features,
                "window_stats": window_stats,
                "normalized_features": normalized_features,
                "state_estimate": state_estimate,
                "baseline_ready": self._is_baseline_ready(),
                "analysis_timestamp": time.monotonic()
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error("keyboard_analysis_pipeline_failed", error=str(e))
            return None

    def _extract_raw_features_from_event(self, event_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract raw keyboard features from event data."""
        try:
            # Use feature extractor to get features
            if self._feature_extractor:
                features = self._feature_extractor.extract(
                    event_data, 
                    timestamp=time.monotonic(),
                    wallclock=self._timestamp_generator.now_wallclock() if self._timestamp_generator else ""
                )
                
                if features is not None:
                    return features.features
            
            return {}
        except Exception as e:
            self.logger.error("keyboard_feature_extraction_failed", error=str(e))
            return {}

    def _update_temporal_windows(self, features: Dict[str, float], timestamp: float) -> None:
        """Update temporal windows with feature values."""
        if self._temporal_windows is None:
            return
            
        for feature_name, value in features.items():
            if feature_name in self._temporal_windows:
                self._temporal_windows[feature_name].add(value, timestamp)

    def _get_temporal_window_stats(self) -> Dict[str, float]:
        """Get statistics from all temporal windows."""
        stats = {}
        if self._temporal_windows is not None:
            for feature_name, window in self._temporal_windows.items():
                window_stat = window.get_stats()
                if window_stat.count > 0:  # Only include if we have data
                    stats[feature_name] = window_stat.mean
        return stats

    def _update_baseline(self, features: Dict[str, float], timestamp: float) -> None:
        """Update baseline with feature values."""
        if self._baseline_manager is not None:
            for feature_name, value in features.items():
                self._baseline_manager.update(feature_name, value, timestamp)

    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features using baseline manager."""
        if self._baseline_manager is None:
            return features
            
        normalized = {}
        for feature_name, value in features.items():
            normalized_value = self._baseline_manager.normalize(feature_name, value)
            if normalized_value is not None:
                normalized[feature_name] = normalized_value
            else:
                # If baseline not ready, use raw value (will be handled in state estimation)
                normalized[feature_name] = value
                
        return normalized

    def _estimate_keyboard_state(self, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Estimate keyboard state from normalized features."""
        if self._state_estimator is None or self._baseline_manager is None:
            return None
            
        try:
            # Check if baseline is ready
            if not self._is_baseline_ready():
                return {
                    "state": "uncertain",
                    "confidence": 0.0,
                    "uncertainty": 1.0,
                    "contributing_features": {}
                }
            
            # Get state estimate from estimator
            state_result = self._state_estimator.estimate(features, self._baseline_manager)
            
            if state_result is not None:
                return {
                    "state": state_result.state,
                    "confidence": state_result.confidence,
                    "uncertainty": state_result.uncertainty,
                    "contributing_features": state_result.contributing_features
                }
            
            return None
        except Exception as e:
            self.logger.error("keyboard_state_estimation_failed", error=str(e))
            return None

    def _is_baseline_ready(self) -> bool:
        """Check if keyboard baseline is ready."""
        if self._baseline_manager is None:
            return False
        return self._baseline_manager.is_keyboard_baseline_ready()

    def _store_analysis_results(self, analysis_result: Dict[str, Any], event_id: int = 0) -> None:
        """Store analysis results to analysis files."""
        try:
            # Store keyboard features
            feature_line = {
                "timestamp_monotonic": analysis_result["timestamp_monotonic"],
                "timestamp_wallclock": analysis_result["timestamp_wallclock"],
                "event_id": event_id,
                "features": analysis_result["normalized_features"],
                "raw_features": analysis_result["raw_features"],
                "window_stats": analysis_result["window_stats"],
                "baseline_ready": analysis_result["baseline_ready"]
            }
            
            # Write to features file
            line = orjson.dumps(feature_line) + b"\n"
            self._analysis_features_file.write(line)
            self._analysis_features_file.flush()
            
            # Store keyboard state
            state_line = {
                "timestamp_monotonic": analysis_result["timestamp_monotonic"],
                "timestamp_wallclock": analysis_result["timestamp_wallclock"],
                "event_id": event_id,
                "state": analysis_result["state_estimate"]["state"] if analysis_result["state_estimate"] else "unknown",
                "confidence": analysis_result["state_estimate"]["confidence"] if analysis_result["state_estimate"] else 0.0,
                "uncertainty": analysis_result["state_estimate"]["uncertainty"] if analysis_result["state_estimate"] else 1.0,
                "contributing_features": analysis_result["state_estimate"]["contributing_features"] if analysis_result["state_estimate"] else {},
                "baseline_ready": analysis_result["baseline_ready"]
            }
            
            # Write to states file
            line = orjson.dumps(state_line) + b"\n"
            self._analysis_states_file.write(line)
            self._analysis_states_file.flush()
            
        except Exception as e:
            self.logger.error("keyboard_analysis_storage_failed", error=str(e))

    def _on_keyboard_event(self, event: Event) -> None:
        """Callback for keyboard events from listener."""
        # Forward to processor
        if self._processor:
            self._processor.process_event(event)
        
        # Forward to unified event pipeline
        if self._unified_event_callback:
            self._unified_event_callback(event)

    def start(self, session_id: str, session_dir: Optional[Path] = None) -> bool:
        """
        Start keyboard capture for a session.
        
        Args:
            session_id: Unique session identifier
            session_dir: Optional session directory path
            
        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            if self._status == ModalityStatus.RUNNING:
                self.logger.warning("keyboard_already_running", session_id=session_id)
                return True
            
            self._session_id = session_id
            
            # Use provided session_dir or construct from config
            if session_dir is not None:
                self._session_dir = session_dir
            else:
                self._session_dir = Path(self.config.get("storage", {}).get("output_directory", "data/sessions")) / session_id
            
            self._timestamp_generator = TimestampGenerator()
            self._analysis_chunk_count = 0
            self._last_analysis_timestamp = None
            
            try:
                # Check device availability
                if not KeyboardListener.check_device_availability(self.config.get("device")):
                    self.logger.error("keyboard_no_device_available")
                    self._set_error(RuntimeError("No keyboard device available"), "No keyboard device available")
                    return False
                
                # Initialize components
                self._processor = KeyboardProcessor(self.config)
                self._processor.set_storage_callback(self._on_storage_write)
                self._processor.set_unified_event_callback(self._on_keyboard_event)
                self._processor.set_analysis_callback(self._on_analysis_event)
                
                self._storage = KeyboardStorage(self._session_dir, self.config)
                
                self._listener = KeyboardListener(
                    session_id=session_id,
                    timestamp_generator=self._timestamp_generator,
                    event_callback=self._processor.process_event,
                    config=self.config
                )
                self._listener.set_storage_callback(self._processor.process_event)
                
                # Initialize analysis components
                self._initialize_analysis_components()
                
                # Start components in order
                if not self._storage.start():
                    raise RuntimeError("Failed to start keyboard storage")
                
                if not self._processor.start():
                    raise RuntimeError("Failed to start keyboard processor")
                
                if not self._analysis_start():
                    raise RuntimeError("Failed to start keyboard analysis")
                
                if not self._listener.start():
                    raise RuntimeError("Failed to start keyboard listener")
                
                self._set_status(ModalityStatus.RUNNING)
                self.logger.info("keyboard_capture_started", session_id=session_id)
                return True
                
            except Exception as e:
                self._set_error(e, f"Failed to start keyboard capture: {e}")
                self._cleanup()
                return False

    def stop(self) -> bool:
        """
        Stop keyboard capture.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        with self._lock:
            if self._status == ModalityStatus.STOPPED:
                return True
            
            self._set_status(ModalityStatus.STOPPING)
            
            success = True
            
            # Stop in reverse order
            if self._listener:
                if not self._listener.stop():
                    success = False
            
            if self._processor:
                if not self._processor.stop():
                    success = False
            
            # Stop analysis components
            if not self._analysis_stop():
                success = False
            
            if self._storage:
                if not self._storage.stop():
                    success = False
            
            self._cleanup()
            
            if success:
                self._set_status(ModalityStatus.STOPPED)
                self.logger.info("keyboard_capture_stopped", session_id=self._session_id)
            else:
                self._set_status(ModalityStatus.ERROR)
                self.logger.error("keyboard_capture_stop_failed", session_id=self._session_id)
            
            return success

    def pause(self) -> bool:
        """
        Pause keyboard capture.
        
        Returns:
            True if paused successfully, False otherwise
        """
        with self._lock:
            if self._status != ModalityStatus.RUNNING:
                return False
            
            if self._listener:
                result = self._listener.pause()
                if result:
                    self._set_status(ModalityStatus.PAUSED)
                    self.logger.info("keyboard_capture_paused", session_id=self._session_id)
                return result
            return False

    def resume(self) -> bool:
        """
        Resume keyboard capture.
        
        Returns:
            True if resumed successfully, False otherwise
        """
        with self._lock:
            if self._status != ModalityStatus.PAUSED:
                return False
            
            if self._listener:
                result = self._listener.resume()
                if result:
                    self._set_status(ModalityStatus.RUNNING)
                    self.logger.info("keyboard_capture_resumed", session_id=self._session_id)
                return result
            return False

    def _cleanup(self) -> None:
        """Clean up resources."""
        self._listener = None
        self._processor = None
        self._storage = None
        self._session_id = None
        self._session_dir = None
        self._timestamp_generator = None

    def get_stats(self) -> Dict[str, Any]:
        """Get capture statistics."""
        stats = {
            "status": self._status.value,
            "session_id": self._session_id,
        }
        
        if self._listener:
            stats["listener"] = {
                "running": self._listener.is_running,
                "paused": self._listener.is_paused,
            }
        
        if self._processor:
            stats["processor"] = self._processor.get_stats()
        
        if self._storage:
            stats["storage"] = self._storage.get_stats()
        
        # Add analysis stats
        stats["analysis"] = {
            "chunks_processed": getattr(self, '_analysis_chunk_count', 0),
            "last_analysis_timestamp": getattr(self, '_last_analysis_timestamp', None),
            "features_file": getattr(self, '_analysis_features_path', None),
            "states_file": getattr(self, '_analysis_states_path', None)
        }
        
        return stats