"""
Unit tests for project structure and basic functionality.
"""

import pytest
import os
from pathlib import Path

def test_project_structure():
    """Test that the basic project structure exists."""
    # Check that key directories exist
    assert os.path.isdir("src"), "src directory should exist"
    assert os.path.isdir("src/core"), "src/core directory should exist"
    assert os.path.isdir("src/session"), "src/session directory should exist"
    assert os.path.isdir("src/audio"), "src/audio directory should exist"
    assert os.path.isdir("src/keyboard"), "src/keyboard directory should exist"
    assert os.path.isdir("src/face"), "src/face directory should exist"
    assert os.path.isdir("src/processing"), "src/processing directory should exist"
    assert os.path.isdir("src/storage"), "src/storage directory should exist"
    assert os.path.isdir("src/ui"), "src/ui directory should exist"
    
    assert os.path.isdir("tests"), "tests directory should exist"
    assert os.path.isdir("tests/unit"), "tests/unit directory should exist"
    assert os.path.isdir("tests/integration"), "tests/integration directory should exist"
    assert os.path.isdir("tests/failure"), "tests/failure directory should exist"
    
    assert os.path.isdir("data/sessions"), "data/sessions directory should exist"
    assert os.path.isdir("config"), "config directory should exist"
    assert os.path.isdir("docs"), "docs directory should exist"

def test_key_files_exist():
    """Test that key files exist."""
    assert os.path.isfile("pyproject.toml"), "pyproject.toml should exist"
    assert os.path.isfile("requirements.txt"), "requirements.txt should exist"
    assert os.path.isfile("config/config.yaml"), "config/config.yaml should exist"
    assert os.path.isfile("src/main.py"), "src/main.py should exist"
    assert os.path.isfile("README.md"), "README.md should exist"
    assert os.path.isfile("SPEC.md"), "SPEC.md should exist"

def test_python_package_structure():
    """Test that Python packages have __init__.py files."""
    assert os.path.isfile("src/__init__.py"), "src/__init__.py should exist"
    assert os.path.isfile("src/core/__init__.py"), "src/core/__init__.py should exist"
    assert os.path.isfile("src/session/__init__.py"), "src/session/__init__.py should exist"
    assert os.path.isfile("src/audio/__init__.py"), "src/audio/__init__.py should exist"
    assert os.path.isfile("src/keyboard/__init__.py"), "src/keyboard/__init__.py should exist"
    assert os.path.isfile("src/face/__init__.py"), "src/face/__init__.py should exist"
    assert os.path.isfile("src/processing/__init__.py"), "src/processing/__init__.py should exist"
    assert os.path.isfile("src/storage/__init__.py"), "src/storage/__init__.py should exist"
    assert os.path.isfile("src/ui/__init__.py"), "src/ui/__init__.py should exist"