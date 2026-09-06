"""
Session storage for managing session metadata and coordinating modality storage.

Provides a unified interface for session data persistence.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import orjson

from core.logging import get_logger
from core.timestamps import TimestampGenerator

if TYPE_CHECKING:
    from session.metadata import SessionMetadata, ModalityStatusInfo


class SessionStore:
    """
    Manages session metadata and coordinates modality-specific storage.
    
    Handles session.json and coordinates with modality-specific storage
    (audio, keyboard, face) to maintain a consistent session data structure.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Storage configuration
        self.output_directory = Path(config.get("storage", {}).get("output_directory", "data/sessions"))
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Active session state
        self._active_session: Optional["SessionMetadata"] = None
        self._session_dir: Optional[Path] = None
        self._timestamp_generator: Optional[TimestampGenerator] = None
        
        # Modality storage references
        self._audio_storage = None
        self._keyboard_storage = None
        self._face_storage = None

    def set_modality_storages(
        self,
        audio_storage: Optional[Any] = None,
        keyboard_storage: Optional[Any] = None,
        face_storage: Optional[Any] = None
    ) -> None:
        """Set modality storage references."""
        self._audio_storage = audio_storage
        self._keyboard_storage = keyboard_storage
        self._face_storage = face_storage

    def create_session(
        self,
        session_id: str,
        timestamp_generator: TimestampGenerator,
        application_version: str,
        enabled_modalities: List[str],
        config_snapshot: Dict[str, Any],
        start_time_monotonic: float,
        start_time_wallclock: str
    ) -> "SessionMetadata":
        """
        Create a new session and initialize its metadata.
        
        Args:
            session_id: Unique session identifier
            timestamp_generator: Timestamp generator for the session
            application_version: Application version string
            enabled_modalities: List of enabled modality names
            config_snapshot: Configuration snapshot for this session
            start_time_monotonic: Session start time (monotonic, seconds)
            start_time_wallclock: Session start time (wallclock, ISO 8601)
            
        Returns:
            SessionMetadata object for the new session
        """
        with self._lock:
            # Import here to avoid circular imports
            from session.metadata import SessionMetadata
            
            # Create session metadata
            metadata = SessionMetadata.create_new(
                application_version=application_version,
                enabled_modalities=enabled_modalities,
                config_snapshot=config_snapshot,
                start_time_monotonic=start_time_monotonic,
                start_time_wallclock=start_time_wallclock,
            )
            
            # Override session_id if provided
            if session_id:
                metadata.session_id = session_id
            
            # Set active session
            self._active_session = metadata
            self._timestamp_generator = timestamp_generator
            
            # Create session directory
            self._session_dir = self.output_directory / metadata.session_id
            self._session_dir.mkdir(parents=True, exist_ok=True)
            (self._session_dir / "audio").mkdir(exist_ok=True)
            
            # Write initial session.json
            self._write_session_json()
            
            self.logger.info("session_created", 
                           session_id=metadata.session_id,
                           enabled_modalities=enabled_modalities)
            
            return metadata

    def get_active_session(self) -> Optional["SessionMetadata"]:
        """Get the currently active session metadata."""
        with self._lock:
            return self._active_session

    def get_session_dir(self) -> Optional[Path]:
        """Get the current session directory."""
        with self._lock:
            return self._session_dir

    def get_timestamp_generator(self) -> Optional[TimestampGenerator]:
        """Get the timestamp generator for the active session."""
        with self._lock:
            return self._timestamp_generator

    def update_modality_status(
        self,
        modality_name: str,
        status: str,
        error_message: Optional[str] = None
    ) -> None:
        """Update the status of a modality in the active session."""
        with self._lock:
            if self._active_session:
                self._active_session.update_modality_status(
                    modality_name, status, error_message
                )
                self._write_session_json()

    def mark_session_completed(
        self,
        end_time_monotonic: float,
        end_time_wallclock: str
    ) -> None:
        """Mark the active session as completed."""
        with self._lock:
            if self._active_session:
                self._active_session.mark_completed(
                    end_time_monotonic, end_time_wallclock
                )
                self._write_session_json()
                self.logger.info("session_completed", 
                               session_id=self._active_session.session_id,
                               duration_ms=self._active_session.duration_ms)

    def mark_session_failed(
        self,
        end_time_monotonic: float,
        end_time_wallclock: str,
        error_count: int
    ) -> None:
        """Mark the active session as failed."""
        with self._lock:
            if self._active_session:
                self._active_session.mark_failed(
                    end_time_monotonic, end_time_wallclock, error_count
                )
                self._write_session_json()
                self.logger.error("session_failed", 
                                session_id=self._active_session.session_id,
                                error_count=error_count)

    def _write_session_json(self) -> None:
        """Write session metadata to session.json."""
        if not self._active_session or not self._session_dir:
            return
        
        try:
            session_file = self._session_dir / "session.json"
            data = self._active_session.to_dict()
            
            # Write atomically using a temporary file
            temp_file = session_file.with_suffix(".tmp")
            with open(temp_file, "wb") as f:
                f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
            
            # Atomic rename
            temp_file.replace(session_file)
            
        except Exception as e:
            self.logger.error("session_json_write_failed", 
                            session_id=self._active_session.session_id if self._active_session else None,
                            error=str(e))

    def save_session_json(self, session_dir: Path, metadata: "SessionMetadata") -> bool:
        """
        Save session metadata to a specific directory.
        
        Args:
            session_dir: Directory to save session.json
            metadata: Session metadata to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session_file = session_dir / "session.json"
            data = metadata.to_dict()
            
            # Write atomically using a temporary file
            temp_file = session_file.with_suffix(".tmp")
            with open(temp_file, "wb") as f:
                f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
            
            # Atomic rename
            temp_file.replace(session_file)
            
            return True
        except Exception as e:
            self.logger.error("session_json_write_failed", 
                            session_id=metadata.session_id,
                            error=str(e))
            return False

    def load_session_json(self, session_dir: Path) -> Optional["SessionMetadata"]:
        """
        Load session metadata from a session directory.
        
        Args:
            session_dir: Directory containing session.json
            
        Returns:
            SessionMetadata if successful, None otherwise
        """
        try:
            # Import here to avoid circular imports
            from session.metadata import SessionMetadata
            
            session_file = session_dir / "session.json"
            if not session_file.exists():
                return None
            
            with open(session_file, "rb") as f:
                data = orjson.loads(f.read())
            
            return SessionMetadata.from_dict(data)
        except Exception as e:
            self.logger.error("session_json_load_failed", 
                            session_dir=str(session_dir),
                            error=str(e))
            return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available sessions.
        
        Returns:
            List of session summaries (session_id, start_time, duration, etc.)
        """
        sessions = []
        try:
            if not self.output_directory.exists():
                return sessions
            
            for session_dir in self.output_directory.iterdir():
                if not session_dir.is_dir():
                    continue
                
                metadata = self.load_session_json(session_dir)
                if metadata:
                    sessions.append({
                        "session_id": metadata.session_id,
                        "start_time_wallclock": metadata.start_time_wallclock,
                        "end_time_wallclock": metadata.end_time_wallclock,
                        "duration_ms": metadata.duration_ms,
                        "status": metadata.status,
                        "enabled_modalities": metadata.enabled_modalities,
                        "error_count": metadata.error_count,
                    })
        except Exception as e:
            self.logger.error("list_sessions_failed", error=str(e))
        
        return sorted(sessions, key=lambda s: s.get("start_time_wallclock", ""), reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its data.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted, False if not found or error
        """
        try:
            session_dir = self.output_directory / session_id
            if not session_dir.exists():
                return False
            
            import shutil
            shutil.rmtree(session_dir)
            self.logger.info("session_deleted", session_id=session_id)
            return True
        except Exception as e:
            self.logger.error("session_delete_failed", 
                            session_id=session_id, error=str(e))
            return False

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a session without loading all data.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session summary or None if not found
        """
        session_dir = self.output_directory / session_id
        metadata = self.load_session_json(session_dir)
        
        if not metadata:
            return None
        
        summary = {
            "session_id": metadata.session_id,
            "start_time_wallclock": metadata.start_time_wallclock,
            "end_time_wallclock": metadata.end_time_wallclock,
            "duration_ms": metadata.duration_ms,
            "status": metadata.status,
            "enabled_modalities": metadata.enabled_modalities,
            "modality_status": {
                name: {"status": info.status, "error": info.error_message}
                for name, info in metadata.modality_status.items()
            },
            "error_count": metadata.error_count,
        }
        
        # Add file sizes if available
        try:
            session_dir = self.output_directory / session_id
            if (session_dir / "keystrokes.jsonl").exists():
                metadata_file = session_dir / "keystrokes.jsonl"
                metadata["keystrokes_size_bytes"] = metadata_file.stat().st_size
            
            if (session_dir / "facial_features.jsonl").exists():
                metadata_file = session_dir / "facial_features.jsonl"
                metadata["facial_features_size_bytes"] = metadata_file.stat().st_size
            
            audio_dir = session_dir / "audio"
            if audio_dir.exists():
                total_audio = sum(f.stat().st_size for f in audio_dir.glob("*.wav"))
                metadata["audio_size_bytes"] = total_audio
        except Exception:
            pass
        
        return summary

    def clear_active_session(self) -> None:
        """Clear the active session state."""
        with self._lock:
            self._active_session = None
            self._session_dir = None
            self._timestamp_generator = None


class SessionLoader:
    """
    Loads session data from disk.
    
    Provides methods to load session metadata, events, and modality-specific data.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        self.output_directory = Path(config.get("storage", {}).get("output_directory", "data/sessions"))

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load complete session data.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            Dictionary with session metadata, events, and modality data, or None
        """
        session_dir = self.output_directory / session_id
        if not session_dir.exists():
            return None
        
        # Load metadata
        metadata_file = session_dir / "session.json"
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, "rb") as f:
                metadata = orjson.loads(f.read())
        except Exception as e:
            self.logger.error("load_metadata_failed", session_id=session_id, error=str(e))
            return None
        
        # Load events
        events = self._load_events(session_dir)
        
        # Load keystrokes
        keystrokes = self._load_keystrokes(session_dir)
        
        # Load facial features
        facial_features = self._load_facial_features(session_dir)
        
        # Load audio index
        audio_index = self._load_audio_index(session_dir)
        
        return {
            "metadata": metadata,
            "events": events,
            "keystrokes": keystrokes,
            "facial_features": facial_features,
            "audio_index": audio_index,
        }

    def _load_events(self, session_dir: Path) -> List[Dict[str, Any]]:
        """Load events from events.jsonl."""
        events_file = session_dir / "events.jsonl"
        if not events_file.exists():
            return []
        
        events = []
        try:
            with open(events_file, "rb") as f:
                for line in f:
                    if line.strip():
                        events.append(orjson.loads(line))
        except Exception as e:
            self.logger.error("load_events_failed", session_dir=str(session_dir), error=str(e))
        
        return events

    def _load_keystrokes(self, session_dir: Path) -> List[Dict[str, Any]]:
        """Load keystrokes from keystrokes.jsonl."""
        keystrokes_file = session_dir / "keystrokes.jsonl"
        if not keystrokes_file.exists():
            return []
        
        keystrokes = []
        try:
            with open(keystrokes_file, "rb") as f:
                for line in f:
                    if line.strip():
                        keystrokes.append(orjson.loads(line))
        except Exception as e:
            self.logger.error("load_keystrokes_failed", session_dir=str(session_dir), error=str(e))
        
        return keystrokes

    def _load_facial_features(self, session_dir: Path) -> List[Dict[str, Any]]:
        """Load facial features from facial_features.jsonl."""
        features_file = session_dir / "facial_features.jsonl"
        if not features_file.exists():
            return []
        
        features = []
        try:
            with open(features_file, "rb") as f:
                for line in f:
                    if line.strip():
                        features.append(orjson.loads(line))
        except Exception as e:
            self.logger.error("load_facial_features_failed", session_dir=str(session_dir), error=str(e))
        
        return features

    def _load_audio_index(self, session_dir: Path) -> List[Dict[str, Any]]:
        """Load audio index from audio/audio_index.jsonl."""
        audio_index_file = session_dir / "audio" / "audio_index.jsonl"
        if not audio_index_file.exists():
            return []
        
        index = []
        try:
            with open(audio_index_file, "rb") as f:
                for line in f:
                    if line.strip():
                        index.append(orjson.loads(line))
        except Exception as e:
            self.logger.error("load_audio_index_failed", session_dir=str(session_dir), error=str(e))
        
        return index

    def load_audio_chunk(self, session_id: str, chunk_index: int) -> Optional[bytes]:
        """
        Load audio chunk data from WAV file.
        
        Args:
            session_id: Session ID
            chunk_index: Chunk index (1-based)
            
        Returns:
            Audio data as bytes, or None if not found
        """
        session_dir = self.output_directory / session_id
        audio_dir = session_dir / "audio"
        filename = f"chunk_{chunk_index:06d}.wav"
        filepath = audio_dir / filename
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, "rb") as f:
                return f.read()
        except Exception as e:
            self.logger.error("load_audio_chunk_failed", 
                            session_id=session_id, chunk_index=chunk_index, error=str(e))
            return None

    def get_session_events_generator(self, session_id: str, modality: Optional[str] = None):
        """
        Generator that yields events from a session, optionally filtered by modality.
        
        Args:
            session_id: Session ID
            modality: Optional modality filter ('voice', 'keyboard', 'face', 'system')
            
        Yields:
            Event dictionaries
        """
        session_dir = self.output_directory / session_id
        events_file = session_dir / "events.jsonl"
        
        if not events_file.exists():
            return
        
        try:
            with open(events_file, "rb") as f:
                for line in f:
                    if line.strip():
                        event = orjson.loads(line)
                        if modality is None or event.get("modality") == modality:
                            yield event
        except Exception as e:
            self.logger.error("event_generator_error", session_id=session_id, error=str(e))

    def export_session_csv(self, session_id: str, output_dir: Path) -> bool:
        """
        Export session data to CSV files for analysis.
        
        Args:
            session_id: Session ID to export
            output_dir: Directory to write CSV files
            
        Returns:
            True if successful
        """
        import csv
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            session_data = self.load_session(session_id)
            if not session_data:
                return False
            
            # Export events
            events_file = output_dir / f"{session_id}_events.csv"
            with open(events_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["event_id", "timestamp_monotonic", "timestamp_wallclock", 
                               "modality", "event_type", "data"])
                for event in session_data["events"]:
                    writer.writerow([
                        event.get("event_id", ""),
                        event.get("timestamp", {}).get("timestamp_monotonic", ""),
                        event.get("timestamp", {}).get("timestamp_wallclock", ""),
                        event.get("modality", ""),
                        event.get("event_type", ""),
                        json.dumps(event.get("data", {}))
                    ])
            
            # Export keystrokes
            keystrokes_file = output_dir / f"{session_id}_keystrokes.csv"
            with open(keystrokes_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["event_id", "timestamp_monotonic", "timestamp_wallclock",
                               "event_type", "key", "key_code", "modifiers", 
                               "duration_ms", "is_printable", "is_sensitive_context"])
                for ks in session_data["keystrokes"]:
                    writer.writerow([
                        ks.get("event_id", ""),
                        ks.get("timestamp_monotonic", ""),
                        ks.get("timestamp_wallclock", ""),
                        ks.get("event_type", ""),
                        ks.get("key", ""),
                        ks.get("key_code", ""),
                        "|".join(ks.get("modifiers", [])),
                        ks.get("duration_ms", ""),
                        ks.get("is_printable", ""),
                        ks.get("is_sensitive_context", "")
                    ])
            
            # Export facial features
            features_file = output_dir / f"{session_id}_facial_features.csv"
            with open(features_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["observation_id", "timestamp_monotonic", "timestamp_wallclock",
                               "frame_index", "face_detected", "detection_confidence",
                               "processing_time_ms"])
                for ff in session_data["facial_features"]:
                    writer.writerow([
                        ff.get("observation_id", ""),
                        ff.get("timestamp_monotonic", ""),
                        ff.get("timestamp_wallclock", ""),
                        ff.get("frame_index", ""),
                        ff.get("face_detected", ""),
                        ff.get("detection_confidence", ""),
                        ff.get("processing_time_ms", "")
                    ])
            
            return True
        except Exception as e:
            self.logger.error("export_csv_failed", session_id=session_id, error=str(e))
            return False