"""
Unit tests for verifying that required libraries are installed and importable.
"""

import pytest

def test_sounddevice_import():
    """Test that sounddevice can be imported."""
    import sounddevice
    assert sounddevice is not None

def test_soundfile_import():
    """Test that soundfile can be imported."""
    import soundfile
    assert soundfile is not None

def test_pynput_import():
    """Test that pynput can be imported."""
    import pynput
    assert pynput is not None

def test_opencv_import():
    """Test that opencv-python can be imported."""
    import cv2
    assert cv2 is not None

def test_mediapipe_import():
    """Test that mediapipe can be imported."""
    import mediapipe
    assert mediapipe is not None

def test_pyside6_import():
    """Test that PySide6 can be imported."""
    import PySide6
    assert PySide6 is not None

def test_numpy_import():
    """Test that numpy can be imported."""
    import numpy
    assert numpy is not None

def test_scipy_import():
    """Test that scipy can be imported."""
    import scipy
    assert scipy is not None

def test_yaml_import():
    """Test that pyyaml can be imported."""
    import yaml
    assert yaml is not None

def test_structlog_import():
    """Test that structlog can be imported."""
    import structlog
    assert structlog is not None

def test_orjson_import():
    """Test that orjson can be imported."""
    import orjson
    assert orjson is not None

def test_pytest_import():
    """Test that pytest can be imported."""
    import pytest
    assert pytest is not None