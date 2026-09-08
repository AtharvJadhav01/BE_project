"""
Multimodal fusion module for real-time human-state analysis.

Combines face, audio, and keyboard analysis into a unified human-state estimate.
"""

from analysis.multimodal.fusion import HumanStateEstimator as MultimodalFusion, HumanStateEstimator
from analysis.multimodal.alignment import TemporalAligner, ModalitySnapshot
from analysis.multimodal.fusion_engine import FusionEngine

__all__ = [
    "MultimodalFusion",
    "HumanStateEstimator",
    "TemporalAligner",
    "ModalitySnapshot",
    "FusionEngine",
]