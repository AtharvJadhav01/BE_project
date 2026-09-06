"""
Session loader module for loading session data from disk.

Re-exports SessionLoader from session_store for backward compatibility.
"""

from .session_store import SessionLoader

__all__ = ["SessionLoader"]