"""
Audio analysis module for real-time pitch and prosodic feature extraction.

This module provides components for analyzing audio from microphone capture
to extract meaningful acoustic features and estimate vocal states.
"""

from analysis.audio.features import AudioFeatureExtractor
from analysis.audio.state import AudioStateEstimator
from analysis.audio.baseline import AudioBaselineManager

__all__ = [
    "AudioFeatureExtractor",
    "AudioStateEstimator", 
    "AudioBaselineManager"
]