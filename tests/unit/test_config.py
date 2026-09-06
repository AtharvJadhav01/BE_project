"""
Unit tests for configuration loading.
"""

import pytest
import yaml
import os

def test_config_loads():
    """Test that the configuration file can be loaded."""
    config_path = "config/config.yaml"
    assert os.path.isfile(config_path), f"Config file {config_path} should exist"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check that required sections exist
    assert "audio" in config, "Audio section should exist in config"
    assert "keyboard" in config, "Keyboard section should exist in config"
    assert "camera" in config, "Camera section should exist in config"
    assert "face" in config, "Face section should exist in config"
    assert "storage" in config, "Storage section should exist in config"
    assert "logging" in config, "Logging section should exist in config"
    assert "ui" in config, "UI section should exist in config"
    
    # Check some specific values
    assert config["audio"]["enabled"] == True, "Audio should be enabled by default"
    assert config["audio"]["sample_rate"] == 16000, "Sample rate should be 16000"
    assert config["keyboard"]["enabled"] == True, "Keyboard should be enabled by default"
    assert config["camera"]["enabled"] == True, "Camera should be enabled by default"
    assert config["face"]["enabled"] == True, "Face processing should be enabled by default"