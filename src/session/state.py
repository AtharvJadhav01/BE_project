"""
Session state management.

Defines the state machine for session lifecycle.
"""

from enum import Enum
from typing import Optional


class SessionState(Enum):
    """Session lifecycle states."""
    READY = "ready"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


# Valid state transitions
VALID_TRANSITIONS = {
    SessionState.READY: [SessionState.RECORDING],
    SessionState.RECORDING: [SessionState.PAUSED, SessionState.STOPPED, SessionState.COMPLETED, SessionState.FAILED],
    SessionState.PAUSED: [SessionState.RECORDING, SessionState.STOPPED, SessionState.FAILED],
    SessionState.STOPPED: [SessionState.COMPLETED, SessionState.FAILED],
    SessionState.COMPLETED: [],
    SessionState.FAILED: [],
}


class SessionStateMachine:
    """
    Manages session state transitions.

    Ensures only valid state transitions are allowed.
    """

    def __init__(self):
        self._state = SessionState.READY
        self._previous_state: Optional[SessionState] = None

    @property
    def state(self) -> SessionState:
        """Get current session state."""
        return self._state

    @property
    def previous_state(self) -> Optional[SessionState]:
        """Get previous session state."""
        return self._previous_state

    def can_transition(self, new_state: SessionState) -> bool:
        """Check if transition to new state is valid."""
        return new_state in VALID_TRANSITIONS.get(self._state, [])

    def transition(self, new_state: SessionState) -> bool:
        """
        Attempt to transition to a new state.

        Args:
            new_state: Target state

        Returns:
            True if transition was successful, False otherwise
        """
        if not self.can_transition(new_state):
            return False
        self._previous_state = self._state
        self._state = new_state
        return True

    def start_recording(self) -> bool:
        """Transition to RECORDING state."""
        return self.transition(SessionState.RECORDING)

    def pause(self) -> bool:
        """Transition to PAUSED state."""
        return self.transition(SessionState.PAUSED)

    def resume(self) -> bool:
        """Transition to RECORDING state (resume from pause)."""
        if self._state != SessionState.PAUSED:
            return False
        return self.transition(SessionState.RECORDING)

    def stop(self) -> bool:
        """Transition to STOPPED state."""
        return self.transition(SessionState.STOPPED)

    def complete(self) -> bool:
        """Transition to COMPLETED state."""
        return self.transition(SessionState.COMPLETED)

    def fail(self) -> bool:
        """Transition to FAILED state."""
        return self.transition(SessionState.FAILED)

    def reset(self) -> None:
        """Reset to initial READY state."""
        self._previous_state = self._state
        self._state = SessionState.READY