"""
Session metadata model.

Defines the structure of session metadata that is stored in session.json.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import uuid


@dataclass
class ModalityStatusInfo:
    """Status information for a single modality in a session."""
    name: str
    status: str  # "completed", "error", "not_started"
    enabled: bool
    error_message: Optional[str] = None
    error_count: int = 0


@dataclass
class SessionMetadata:
    """
    Complete session metadata stored in session.json.

    Contains all information needed to reconstruct the session context.
    """
    session_id: str
    application_version: str
    start_time_wallclock: str  # ISO 8601 UTC
    end_time_wallclock: Optional[str] = None  # ISO 8601 UTC
    start_time_monotonic: float = 0.0
    end_time_monotonic: Optional[float] = None
    duration_ms: Optional[int] = None
    enabled_modalities: list[str] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    status: str = "ready"  # ready, recording, paused, stopped, completed, failed
    modality_status: dict[str, ModalityStatusInfo] = field(default_factory=dict)
    error_count: int = 0
    events_file: str = "events.jsonl"
    audio_index_file: str = "audio/audio_index.jsonl"
    schema_version: str = "1.0"

    @classmethod
    def create_new(
        cls,
        application_version: str,
        enabled_modalities: list[str],
        config_snapshot: dict[str, Any],
        start_time_monotonic: float,
        start_time_wallclock: str,
    ) -> "SessionMetadata":
        """Create a new session metadata instance."""
        return cls(
            session_id=str(uuid.uuid4()),
            application_version=application_version,
            start_time_wallclock=start_time_wallclock,
            start_time_monotonic=start_time_monotonic,
            enabled_modalities=enabled_modalities,
            config_snapshot=config_snapshot,
            status="recording",
            modality_status={
                name: ModalityStatusInfo(
                    name=name,
                    status="not_started",
                    enabled=name in enabled_modalities,
                )
                for name in enabled_modalities
            },
        )

    def mark_completed(self, end_time_monotonic: float, end_time_wallclock: str) -> None:
        """Mark session as completed."""
        self.end_time_monotonic = end_time_monotonic
        self.end_time_wallclock = end_time_wallclock
        # Handle both absolute and elapsed time for end_time_monotonic
        if end_time_monotonic < self.start_time_monotonic:
            # end_time_monotonic is elapsed time since session start
            self.duration_ms = int(end_time_monotonic * 1000)
        else:
            # end_time_monotonic is absolute monotonic time
            self.duration_ms = int((end_time_monotonic - self.start_time_monotonic) * 1000)
        self.status = "completed"
        for modality in self.modality_status.values():
            if modality.status == "not_started":
                modality.status = "completed"

    def mark_failed(self, end_time_monotonic: float, end_time_wallclock: str, error_count: int) -> None:
        """Mark session as failed."""
        self.end_time_monotonic = end_time_monotonic
        self.end_time_wallclock = end_time_wallclock
        # Handle both absolute and elapsed time for end_time_monotonic
        if end_time_monotonic < self.start_time_monotonic:
            # end_time_monotonic is elapsed time since session start
            self.duration_ms = int(end_time_monotonic * 1000)
        else:
            # end_time_monotonic is absolute monotonic time
            self.duration_ms = int((end_time_monotonic - self.start_time_monotonic) * 1000)
        self.status = "failed"
        self.error_count = error_count

    def update_modality_status(
        self,
        modality_name: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Update status for a specific modality."""
        if modality_name in self.modality_status:
            self.modality_status[modality_name].status = status
            if error_message:
                self.modality_status[modality_name].error_message = error_message
                self.modality_status[modality_name].error_count += 1
                self.error_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "application_version": self.application_version,
            "start_time_wallclock": self.start_time_wallclock,
            "end_time_wallclock": self.end_time_wallclock,
            "start_time_monotonic": self.start_time_monotonic,
            "end_time_monotonic": self.end_time_monotonic,
            "duration_ms": self.duration_ms,
            "enabled_modalities": self.enabled_modalities,
            "config_snapshot": self.config_snapshot,
            "status": self.status,
            "modality_status": {
                name: {
                    "name": info.name,
                    "status": info.status,
                    "enabled": info.enabled,
                    "error_message": info.error_message,
                    "error_count": info.error_count,
                }
                for name, info in self.modality_status.items()
            },
            "error_count": self.error_count,
            "events_file": self.events_file,
            "audio_index_file": self.audio_index_file,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMetadata":
        """Create SessionMetadata from dictionary."""
        modality_status = {}
        for name, info in data.get("modality_status", {}).items():
            modality_status[name] = ModalityStatusInfo(
                name=info["name"],
                status=info["status"],
                enabled=info["enabled"],
                error_message=info.get("error_message"),
                error_count=info.get("error_count", 0),
            )

        return cls(
            session_id=data["session_id"],
            application_version=data["application_version"],
            start_time_wallclock=data["start_time_wallclock"],
            end_time_wallclock=data.get("end_time_wallclock"),
            start_time_monotonic=data["start_time_monotonic"],
            end_time_monotonic=data.get("end_time_monotonic"),
            duration_ms=data.get("duration_ms"),
            enabled_modalities=data["enabled_modalities"],
            config_snapshot=data["config_snapshot"],
            status=data["status"],
            modality_status=modality_status,
            error_count=data.get("error_count", 0),
            events_file=data.get("events_file", "events.jsonl"),
            audio_index_file=data.get("audio_index_file", "audio/audio_index.jsonl"),
            schema_version=data.get("schema_version", "1.0"),
        )