"""
Keyboard event listener using pynput.

Captures global keyboard events (key_down, key_up) with timestamps.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set
from pynput import keyboard

from core.events import Event, EventType, ModalityType, create_keyboard_event
from core.interfaces import KeyboardModalityInterface, ModalityStatus
from core.logging import get_logger
from core.timestamps import TimestampGenerator


@dataclass
class KeyEventData:
    """Raw key event data from pynput."""
    key: keyboard.Key
    event_type: str  # "press" or "release"
    timestamp: float  # monotonic time


class KeyboardListener:
    """
    Global keyboard listener using pynput.
    
    Runs in a separate thread and captures key press/release events.
    """

    def __init__(
        self,
        session_id: str,
        timestamp_generator: TimestampGenerator,
        event_callback: Callable[[Event], None],
        config: Dict[str, Any]
    ):
        self.session_id = session_id
        self.timestamp_generator = timestamp_generator
        self.event_callback = event_callback
        self.config = config
        self.logger = get_logger(__name__)
        
        # State
        self._listener: Optional[keyboard.Listener] = None
        self._running = False
        self._paused = False
        self._lock = threading.Lock()
        
        # Key tracking for duration calculation
        self._pressed_keys: Dict[str, float] = {}  # key -> press_time
        self._key_sequence = 0
        
        # Sensitive context detection
        self._sensitive_keywords = {
            "password", "login", "signin", "credential", "secret",
            "token", "pin", "cvv", "ssn", "credit", "card", "auth",
            "2fa", "mfa", "passwd", "passcode"
        }
        self._current_window_title: Optional[str] = None
        self._current_process_name: Optional[str] = None
        
        # Platform-specific key code mapping
        self._key_code_map = self._build_key_code_map()
        
        # Storage callback
        self._storage_callback: Optional[callable] = None

    def set_storage_callback(self, callback: callable) -> None:
        """Set callback for storing keyboard events."""
        self._storage_callback = callback

    def _build_key_code_map(self) -> Dict[str, int]:
        """Build mapping from key names to virtual key codes."""
        # Common virtual key codes (Windows VK_* compatible)
        return {
            "a": 65, "b": 66, "c": 67, "d": 68, "e": 69, "f": 70, "g": 71,
            "h": 72, "i": 73, "j": 74, "k": 75, "l": 76, "m": 77, "n": 78,
            "o": 79, "p": 80, "q": 81, "r": 82, "s": 83, "t": 84, "u": 85,
            "v": 86, "w": 87, "x": 88, "y": 89, "z": 90,
            "0": 48, "1": 49, "2": 50, "3": 51, "4": 52, "5": 53, "6": 54, "7": 55, "8": 56, "9": 57,
            "f1": 112, "f2": 113, "f3": 114, "f4": 115, "f5": 116, "f6": 117,
            "f7": 118, "f8": 119, "f9": 120, "f10": 121, "f11": 122, "f12": 123,
            "enter": 13, "tab": 9, "backspace": 8, "escape": 27, "space": 32,
            "shift": 16, "ctrl": 17, "alt": 18, "cmd": 91, "win": 91,
            "up": 38, "down": 40, "left": 37, "right": 39,
            "home": 36, "end": 35, "page_up": 33, "page_down": 34,
            "insert": 45, "delete": 46,
            "num_0": 96, "num_1": 97, "num_2": 98, "num_3": 99, "num_4": 100,
            "num_5": 101, "num_6": 102, "num_7": 103, "num_8": 104, "num_9": 105,
            "num_add": 107, "num_subtract": 109, "num_multiply": 106, "num_divide": 111,
            "num_decimal": 110, "num_enter": 13,
        }

    def _normalize_key(self, key: keyboard.Key) -> tuple[str, bool, int]:
        """
        Normalize pynput key to standard format.
        
        Returns:
            tuple: (key_name, is_printable, key_code)
        """
        key_name = ""
        is_printable = False
        key_code = 0
        
        if hasattr(key, 'char') and key.char is not None:
            # Printable character
            key_name = key.char
            is_printable = True
            key_code = ord(key.char.upper()) if key.char.isalpha() else ord(key.char)
        elif hasattr(key, 'name'):
            # Special key
            key_name = key.name
            is_printable = False
            key_code = self._key_code_map.get(key_name.lower(), 0)
            
            # Normalize modifier key names
            if key_name in ("shift", "shift_l", "shift_r"):
                key_name = "Shift_L" if "l" in key_name else "Shift_R"
            elif key_name in ("ctrl", "ctrl_l", "ctrl_r", "control", "control_l", "control_r"):
                key_name = "Control_L" if "l" in key_name else "Control_R"
            elif key_name in ("alt", "alt_l", "alt_r"):
                key_name = "Alt_L" if "l" in key_name else "Alt_R"
            elif key_name in ("cmd", "cmd_l", "cmd_r", "meta", "meta_l", "meta_r", "win", "win_l", "win_r"):
                key_name = "Meta_L" if "l" in key_name else "Meta_R"
            elif key_name == "space":
                key_name = "Space"
                is_printable = True
            elif key_name == "enter":
                key_name = "Enter"
            elif key_name == "tab":
                key_name = "Tab"
            elif key_name == "backspace":
                key_name = "Backspace"
            elif key_name == "escape":
                key_name = "Escape"
            elif key_name.startswith("f") and key_name[1:].isdigit():
                key_name = key_name.upper()
            elif key_name.startswith("num_") or key_name.startswith("numpad"):
                key_name = key_name.replace("numpad", "Num_").capitalize()
            else:
                key_name = key_name.capitalize()
        else:
            key_name = str(key)
            key_code = 0
        
        return key_name, is_printable, key_code

    def _get_active_modifiers(self) -> Set[str]:
        """Get currently active modifier keys."""
        modifiers = set()
        # Check pressed keys for modifiers
        for key in self._pressed_keys:
            key_lower = key.lower()
            if key_lower in ("shift", "shift_l", "shift_r"):
                modifiers.add("shift")
            elif key_lower in ("ctrl", "ctrl_l", "ctrl_r", "control", "control_l", "control_r"):
                modifiers.add("ctrl")
            elif key_lower in ("alt", "alt_l", "alt_r"):
                modifiers.add("alt")
            elif key_lower in ("cmd", "cmd_l", "cmd_r", "meta", "meta_l", "meta_r", "win", "win_l", "win_r"):
                modifiers.add("cmd" if "cmd" in key_lower or "meta" in key_lower else "win")
        return modifiers

    def _is_sensitive_context(self) -> bool:
        """Check if current context is sensitive (password field, etc.)."""
        if not self.config.get("exclude_sensitive", True):
            return False
        
        # Check window title for sensitive keywords
        if self._current_window_title:
            title_lower = self._current_window_title.lower()
            for keyword in self._sensitive_keywords:
                if keyword in title_lower:
                    return True
        
        return False

    def _update_window_context(self) -> None:
        """Update current window title and process name."""
        try:
            import platform
            if platform.system() == "Windows":
                self._update_window_context_windows()
            elif platform.system() == "Darwin":
                self._update_window_context_macos()
            else:
                self._update_window_context_linux()
        except Exception as e:
            self.logger.debug("window_context_update_failed", error=str(e))

    def _update_window_context_windows(self) -> None:
        """Update window context on Windows using ctypes."""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            
            hwnd = user32.GetForegroundWindow()
            if hwnd:
                # Get window title
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buffer = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buffer, length + 1)
                    self._current_window_title = buffer.value
                
                # Get process name
                pid = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value:
                    handle = kernel32.OpenProcess(0x0400 | 0x0010, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ
                    if handle:
                        buffer = ctypes.create_unicode_buffer(260)
                        if kernel32.GetModuleFileNameExW(handle, None, buffer, 260):
                            import os
                            self._current_process_name = os.path.basename(buffer.value)
                        kernel32.CloseHandle(handle)
        except Exception:
            pass

    def _update_window_context_macos(self) -> None:
        """Update window context on macOS."""
        try:
            import subprocess
            # Get frontmost app
            script = '''
            tell application "System Events"
                set frontApp to first application process whose frontmost is true
                return name of frontApp
            end tell
            '''
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                self._current_process_name = result.stdout.strip()
                self._current_window_title = self._current_process_name
        except Exception:
            pass

    def _update_window_context_linux(self) -> None:
        """Update window context on Linux (X11)."""
        try:
            import subprocess
            # Try xdotool first
            result = subprocess.run(
                ["xdotool", "getwindowfocus", "getwindowname"],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                self._current_window_title = result.stdout.strip()
            
            # Get process name
            result = subprocess.run(
                ["xdotool", "getwindowfocus", "getwindowpid"],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                pid = result.stdout.strip()
                try:
                    with open(f"/proc/{pid}/comm", "r") as f:
                        self._current_process_name = f.read().strip()
                except Exception:
                    pass
        except Exception:
            pass

    def _on_press(self, key: keyboard.Key) -> None:
        """Handle key press event."""
        if not self._running or self._paused:
            return
        
        with self._lock:
            key_name, is_printable, key_code = self._normalize_key(key)
            press_time = self.timestamp_generator.now_monotonic()
            
            # Track pressed key for duration calculation
            self._pressed_keys[key_name] = press_time
            
            # Update window context periodically
            if self.config.get("capture_context", True):
                self._update_window_context()
            
            # Check sensitive context
            is_sensitive = self._is_sensitive_context()
            
            # Build event data
            event_data = {
                "key": "[REDACTED]" if is_sensitive else key_name,
                "key_code": key_code,
                "modifiers": list(self._get_active_modifiers()),
                "is_printable": is_printable,
                "is_sensitive_context": is_sensitive,
            }
            
            if self.config.get("capture_context", True) and not is_sensitive:
                event_data["window_title"] = self._current_window_title
                event_data["process_name"] = self._current_process_name
            
            # Create and emit event
            event = create_keyboard_event(
                session_id=self.session_id,
                event_type=EventType.KEY_DOWN,
                data=event_data,
                timestamp_generator=self.timestamp_generator,
            )
            self.event_callback(event)
            
            # Call storage callback with raw data for storage
            if self._storage_callback:
                storage_data = {
                    "key": event_data["key"],
                    "key_code": event_data["key_code"],
                    "modifiers": event_data["modifiers"],
                    "is_printable": event_data["is_printable"],
                    "is_sensitive_context": event_data["is_sensitive_context"],
                    "window_title": event_data.get("window_title"),
                    "process_name": event_data.get("process_name"),
                }
                self._storage_callback(0, key, press_time, 0, 0)  # placeholder for chunk_index, data, start_time, duration, rms_db
            
            self.logger.debug("key_pressed", key=key_name if not is_sensitive else "[REDACTED]")

    def _on_release(self, key: keyboard.Key) -> None:
        """Handle key release event."""
        if not self._running or self._paused:
            return
        
        with self._lock:
            key_name, is_printable, key_code = self._normalize_key(key)
            release_time = self.timestamp_generator.now_monotonic()
            
            # Calculate duration if we have the press time
            press_time = self._pressed_keys.pop(key_name, None)
            duration_ms = None
            if press_time is not None:
                duration_ms = (release_time - press_time) * 1000
            
            # Check sensitive context
            is_sensitive = self._is_sensitive_context()
            
            # Build event data
            event_data = {
                "key": "[REDACTED]" if is_sensitive else key_name,
                "key_code": key_code,
                "modifiers": list(self._get_active_modifiers()),
                "duration_ms": duration_ms,
                "is_printable": is_printable,
            }
            
            # Create and emit event
            event = create_keyboard_event(
                session_id=self.session_id,
                event_type=EventType.KEY_UP,
                data=event_data,
                timestamp_generator=self.timestamp_generator,
            )
            self.event_callback(event)
            
            # Call storage callback with raw data for storage
            if self._storage_callback:
                storage_data = {
                    "key": event_data["key"],
                    "key_code": event_data["key_code"],
                    "modifiers": event_data["modifiers"],
                    "duration_ms": event_data.get("duration_ms"),
                    "is_printable": event_data["is_printable"],
                    "is_sensitive_context": event_data["is_sensitive_context"],
                    "window_title": event_data.get("window_title"),
                    "process_name": event_data.get("process_name"),
                }
                self._storage_callback(0, key, press_time, duration_ms or 0, 0)
            
            self.logger.debug("key_released", key=key_name if not is_sensitive else "[REDACTED]")

    def start(self) -> bool:
        """Start the keyboard listener."""
        if self._running:
            return True
        
        try:
            self._listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
                suppress=False  # Don't suppress keys
            )
            self._listener.start()
            self._running = True
            self._paused = False
            self.logger.info("keyboard_listener_started", session_id=self.session_id)
            return True
        except Exception as e:
            self.logger.error("keyboard_listener_start_failed", error=str(e))
            return False

    def stop(self) -> bool:
        """Stop the keyboard listener."""
        if not self._running:
            return True
        
        try:
            if self._listener:
                self._listener.stop()
                self._listener = None
            self._running = False
            self._pressed_keys.clear()
            self.logger.info("keyboard_listener_stopped", session_id=self.session_id)
            return True
        except Exception as e:
            self.logger.error("keyboard_listener_stop_failed", error=str(e))
            return False

    def pause(self) -> bool:
        """Pause the keyboard listener."""
        if not self._running or self._paused:
            return False
        self._paused = True
        self.logger.info("keyboard_listener_paused", session_id=self.session_id)
        return True

    def resume(self) -> bool:
        """Resume the keyboard listener."""
        if not self._running or not self._paused:
            return False
        self._paused = False
        self.logger.info("keyboard_listener_resumed", session_id=self.session_id)
        return True

    @property
    def is_running(self) -> bool:
        return self._running and not self._paused

    @property
    def is_paused(self) -> bool:
        return self._paused

    @staticmethod
    def check_device_availability(device: Optional[int] = None) -> bool:
        """Check if a keyboard device is available."""
        try:
            import pynput
            # pynput doesn't have a direct device check, but we can try to create a listener
            # If it fails, no keyboard device is available
            listener = pynput.keyboard.Listener(on_press=lambda k: None, on_release=lambda k: None)
            listener.start()
            listener.stop()
            return True
        except Exception:
            return False

    @staticmethod
    def get_available_devices(max_devices: int = 10) -> list:
        """Get list of available keyboard devices (always returns at least one if any)."""
        try:
            import pynput
            listener = pynput.keyboard.Listener(on_press=lambda k: None, on_release=lambda k: None)
            listener.start()
            listener.stop()
            return [{"index": 0, "name": "Default Keyboard"}]
        except Exception:
            return []