"""
Timestamp generation and synchronization utilities.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Timestamp:
    """Unified timestamp with both monotonic and wall-clock time."""
    monotonic: float  # seconds since session start (monotonic clock)
    wallclock: str    # ISO 8601 UTC timestamp

    @property
    def monotonic_ms(self) -> float:
        """Monotonic timestamp in milliseconds."""
        return self.monotonic * 1000

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp_monotonic": self.monotonic,
            "timestamp_wallclock": self.wallclock,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Timestamp":
        """Create Timestamp from dictionary."""
        return cls(
            monotonic=data["timestamp_monotonic"],
            wallclock=data["timestamp_wallclock"],
        )


class TimestampGenerator:
    """
    Generates synchronized timestamps using both monotonic and wall-clock time.

    Uses time.monotonic() for elapsed time measurement (immune to clock adjustments)
    and time.time() for wall-clock timestamps (for human-readable time).
    """

    def __init__(self):
        self._session_start_monotonic: Optional[float] = None
        self._session_start_wallclock: Optional[float] = None

    def start_session(self) -> None:
        """Mark the start of a session for timestamp synchronization."""
        self._session_start_monotonic = time.monotonic()
        self._session_start_wallclock = time.time()

    def reset(self) -> None:
        """Reset the timestamp generator."""
        self._session_start_monotonic = None
        self._session_start_wallclock = None

    def now(self) -> Timestamp:
        """
        Get current synchronized timestamp.

        Returns:
            Timestamp with both monotonic and wall-clock time.
        """
        if self._session_start_monotonic is None:
            self.start_session()

        now_monotonic = time.monotonic()
        now_wallclock = time.time()

        elapsed_monotonic = now_monotonic - self._session_start_monotonic
        # Calculate wallclock based on session start + elapsed monotonic
        # This ensures consistency between monotonic and wallclock
        calculated_wallclock = self._session_start_wallclock + elapsed_monotonic

        return Timestamp(
            monotonic=elapsed_monotonic,
            wallclock=datetime.fromtimestamp(calculated_wallclock, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        )

    def now_monotonic(self) -> float:
        """Get current monotonic time in seconds since session start."""
        if self._session_start_monotonic is None:
            self.start_session()
        return time.monotonic() - self._session_start_monotonic

    def now_wallclock(self) -> str:
        """Get current wall-clock time as ISO 8601 UTC string."""
        if self._session_start_wallclock is None:
            self.start_session()
        now_wallclock = time.time()
        return datetime.fromtimestamp(now_wallclock, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def monotonic_to_wallclock(self, monotonic_time: float) -> str:
        """
        Convert monotonic time (seconds since session start) to wall-clock ISO string.

        Args:
            monotonic_time: Time in seconds since session start

        Returns:
            ISO 8601 UTC timestamp string
        """
        if self._session_start_wallclock is None:
            raise ValueError("Session not started")
        wallclock = self._session_start_wallclock + monotonic_time
        return datetime.fromtimestamp(wallclock, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def wallclock_to_monotonic(self, wallclock_str: str) -> float:
        """
        Convert wall-clock ISO string to monotonic time (seconds since session start).

        Args:
            wallclock_str: ISO 8601 UTC timestamp string

        Returns:
            Time in seconds since session start
        """
        if self._session_start_wallclock is None:
            raise ValueError("Session not started")
        wallclock = datetime.fromisoformat(wallclock_str.replace("Z", "+00:00")).timestamp()
        return wallclock - self._session_start_wallclock

    @property
    def session_start_wallclock(self) -> Optional[str]:
        """Get session start wall-clock time as ISO string."""
        if self._session_start_wallclock is None:
            return None
        return datetime.fromtimestamp(self._session_start_wallclock, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    @property
    def session_start_monotonic(self) -> Optional[float]:
        """Get session start monotonic time."""
        return self._session_start_monotonic


# Global timestamp generator instance
_timestamp_generator = TimestampGenerator()


def get_timestamp_generator() -> TimestampGenerator:
    """Get the global timestamp generator instance."""
    return _timestamp_generator


def now() -> Timestamp:
    """Get current timestamp using the global generator."""
    return _timestamp_generator.now()