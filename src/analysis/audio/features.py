"""
Audio feature extraction for real-time pitch and prosodic analysis.

Converts raw audio chunks into meaningful acoustic features using
aubio for pitch detection and numpy/scipy for other features.
"""

import math
import time
import warnings
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import numpy as np

from analysis.interfaces import FeatureExtractor, FeatureVector, DataQuality
from analysis.results import FeatureVector as FeatureVectorModel
from analysis.features import BaseFeatureExtractor, compute_statistical_features


@dataclass
class AudioFeatures:
    """Container for extracted audio features."""
    # Pitch features
    pitch_hz: float = 0.0
    pitch_valid: bool = False
    pitch_confidence: float = 0.0
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    pitch_min: float = 0.0
    pitch_max: float = 0.0
    pitch_range: float = 0.0
    
    # Energy features
    rms_energy: float = 0.0
    rms_energy_db: float = -120.0
    energy_mean: float = 0.0
    energy_std: float = 0.0
    
    # Zero crossing rate
    zcr: float = 0.0
    zcr_mean: float = 0.0
    zcr_std: float = 0.0
    
    # Voice activity
    voice_activity: float = 0.0
    voiced_ratio: float = 0.0
    
    # Spectral features
    spectral_centroid: float = 0.0
    spectral_rolloff: float = 0.0
    spectral_flux: float = 0.0


class AudioFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts acoustic features from audio chunks.
    
    Uses aubio for pitch detection (if available) and numpy/scipy
    for energy, ZCR, spectral features, and voice activity detection.
    """
    
    # Default pitch range (human voice)
    DEFAULT_PITCH_MIN_HZ = 80.0
    DEFAULT_PITCH_MAX_HZ = 400.0
    
    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        
        # Audio settings
        self.sample_rate = config.get("sample_rate", 16000)
        self.channels = config.get("channels", 1)
        
        # Pitch detection settings
        self.pitch_min_hz = config.get("pitch_min_hz", self.DEFAULT_PITCH_MIN_HZ)
        self.pitch_max_hz = config.get("pitch_max_hz", self.DEFAULT_PITCH_MAX_HZ)
        self.pitch_confidence_threshold = config.get("pitch_confidence_threshold", 0.5)
        self.pitch_method = config.get("pitch_method", "yin")  # aubio method
        
        # Analysis window settings
        self.window_size_sec = config.get("window_size_sec", 0.025)  # 25ms
        self.hop_size_sec = config.get("hop_size_sec", 0.01)  # 10ms
        
        # Voice activity detection
        self.vad_energy_threshold_db = config.get("vad_energy_threshold_db", -40.0)
        self.vad_zcr_threshold = config.get("vad_zcr_threshold", 0.1)
        
        # Temporal windows for pitch/energy tracking
        self._pitch_history: List[float] = []
        self._pitch_timestamps: List[float] = []
        self._pitch_window_sec = config.get("pitch_history_sec", 1.0)
        self._energy_history: List[float] = []
        self._energy_timestamps: List[float] = []
        self._energy_window_sec = config.get("energy_history_sec", 2.0)
        
        # Previous chunk for spectral flux
        self._prev_magnitude: Optional[np.ndarray] = None
        
        # Initialize pitch detector if aubio is available
        self._aubio_pitch = None
        self._init_pitch_detector()
        
        # Frame size calculations
        self.frame_size = int(self.sample_rate * self.window_size_sec)
        self.hop_size = int(self.sample_rate * self.hop_size_sec)
        
    def _init_pitch_detector(self) -> None:
        """Initialize aubio pitch detector if available."""
        try:
            import aubio
            pitch_obj = aubio.pitch(
                self.pitch_method,
                self.frame_size,
                self.hop_size,
                self.sample_rate
            )
            pitch_obj.set_unit("Hz")
            pitch_obj.set_tolerance(0.8)
            pitch_obj.set_silence(-40.0)
            self._aubio_pitch = pitch_obj
        except ImportError:
            # aubio not available - will use fallback
            self._aubio_pitch = None
        except Exception:
            # Other initialization errors
            self._aubio_pitch = None
    
    def _extract_features(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract audio features from audio chunk.
        
        Args:
            raw_data: Dictionary containing:
                     - data: numpy array of audio samples
                     - sample_rate: int
                     - timestamp: float (monotonic)
                     - wallclock: str (ISO 8601)
        
        Returns:
            Dictionary of audio features
        """
        if not raw_data or "data" not in raw_data:
            return self._get_default_features()
        
        audio_data = raw_data.get("data")
        if audio_data is None or len(audio_data) == 0:
            return self._get_default_features()
        
        # Ensure mono
        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data[:, 0]  # Take first channel
        elif audio_data.ndim > 1:
            audio_data = audio_data.flatten()
        
        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        features = AudioFeatures()
        current_time = time.monotonic()
        
        # Extract features
        self._extract_pitch(audio_data, features, current_time)
        self._extract_energy(audio_data, features, current_time)
        self._extract_zcr(audio_data, features)
        self._extract_voice_activity(audio_data, features)
        self._extract_spectral_features(audio_data, features, current_time)
        
        # Update temporal history
        self._update_temporal_history(features, current_time)
        
        # Convert to dictionary
        return self._features_to_dict(features)
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values for silence/no audio."""
        features = AudioFeatures()
        return self._features_to_dict(features)
    
    def _features_to_dict(self, features: AudioFeatures) -> Dict[str, float]:
        """Convert AudioFeatures dataclass to dictionary."""
        return {
            # Pitch
            "pitch_hz": features.pitch_hz,
            "pitch_valid": features.pitch_valid,
            "pitch_confidence": features.pitch_confidence,
            "pitch_mean": features.pitch_mean,
            "pitch_std": features.pitch_std,
            "pitch_min": features.pitch_min,
            "pitch_max": features.pitch_max,
            "pitch_range": features.pitch_range,
            
            # Energy
            "rms_energy": features.rms_energy,
            "rms_energy_db": features.rms_energy_db,
            "energy_mean": features.energy_mean,
            "energy_std": features.energy_std,
            
            # ZCR
            "zcr": features.zcr,
            "zcr_mean": features.zcr_mean,
            "zcr_std": features.zcr_std,
            
            # Voice activity
            "voice_activity": features.voice_activity,
            "voiced_ratio": features.voiced_ratio,
            
            # Spectral
            "spectral_centroid": features.spectral_centroid,
            "spectral_rolloff": features.spectral_rolloff,
            "spectral_flux": features.spectral_flux,
        }
    
    def _extract_pitch(self, audio_data: np.ndarray, features: AudioFeatures, current_time: float) -> None:
        """Extract pitch using aubio or fallback method."""
        try:
            # Use aubio if available
            if self._aubio_pitch is not None:
                import aubio
                # aubio expects float32
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                pitch_obj = self._aubio_pitch
                pitch = pitch_obj(audio_data)
                pitch_hz = float(pitch[0])
                confidence = float(pitch_obj.get_confidence())
                
                # Validate pitch range
                if (self.pitch_min_hz <= pitch_hz <= self.pitch_max_hz and 
                    confidence >= self.pitch_confidence_threshold):
                    features.pitch_hz = pitch_hz
                    features.pitch_valid = True
                    features.pitch_confidence = confidence
                else:
                    features.pitch_hz = 0.0
                    features.pitch_valid = False
                    features.pitch_confidence = 0.0
            else:
                # Fallback: autocorrelation-based pitch detection
                pitch_hz, confidence = self._estimate_pitch_autocorr(audio_data)
                if (self.pitch_min_hz <= pitch_hz <= self.pitch_max_hz and 
                    confidence >= self.pitch_confidence_threshold):
                    features.pitch_hz = pitch_hz
                    features.pitch_valid = True
                    features.pitch_confidence = confidence
                else:
                    features.pitch_hz = 0.0
                    features.pitch_valid = False
                    features.pitch_confidence = 0.0
        except Exception:
            features.pitch_hz = 0.0
            features.pitch_valid = False
            features.pitch_confidence = 0.0
    
    def _estimate_pitch_autocorr(self, audio_data: np.ndarray) -> tuple[float, float]:
        """Estimate pitch using autocorrelation (fallback method)."""
        # Simple autocorrelation-based pitch estimation
        # Normalize
        x = audio_data - np.mean(audio_data)
        if np.max(np.abs(x)) == 0:
            return 0.0, 0.0
        
        x = x / np.max(np.abs(x))
        
        # Autocorrelation
        corr = np.correlate(x, x, mode='full')
        corr = corr[len(corr)//2:]
        
        # Find first zero crossing
        zero_crossings = np.where(np.diff(np.signbit(corr)))[0]
        if len(zero_crossings) == 0:
            return 0.0, 0.0
        
        start = zero_crossings[0]
        
        # Find peak after first zero crossing
        if start + 1 >= len(corr):
            return 0.0, 0.0
        
        peak_idx = np.argmax(corr[start:]) + start
        if peak_idx == 0:
            return 0.0, 0.0
        
        # Estimate pitch
        pitch_hz = self.sample_rate / peak_idx
        confidence = corr[peak_idx] / corr[0] if corr[0] > 0 else 0.0
        
        return pitch_hz, float(confidence)
    
    def _extract_energy(self, audio_data: np.ndarray, features: AudioFeatures, current_time: float) -> None:
        """Extract RMS energy and energy statistics."""
        # RMS energy
        rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
        features.rms_energy = float(rms)
        
        if rms > 0:
            features.rms_energy_db = 20 * math.log10(rms)
        else:
            features.rms_energy_db = -120.0
        
        # Add to history for temporal stats
        self._energy_history.append(features.rms_energy_db)
        self._energy_timestamps.append(current_time)
    
    def _extract_zcr(self, audio_data: np.ndarray, features: AudioFeatures) -> None:
        """Extract zero crossing rate."""
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
        features.zcr = float(zero_crossings) / len(audio_data)
    
    def _extract_voice_activity(self, audio_data: np.ndarray, features: AudioFeatures) -> None:
        """Simple voice activity detection based on energy and ZCR."""
        energy_db = features.rms_energy_db
        zcr = features.zcr
        
        # Simple VAD: energy above threshold AND reasonable ZCR
        is_voiced = (energy_db > self.vad_energy_threshold_db and 
                     zcr < self.vad_zcr_threshold and
                     energy_db > -80.0)  # Not silence
        
        features.voice_activity = 1.0 if is_voiced else 0.0
        
        # Track voiced ratio
        if features.pitch_valid:
            features.voiced_ratio = 1.0
        else:
            features.voiced_ratio = 0.0
    
    def _extract_spectral_features(self, audio_data: np.ndarray, features: AudioFeatures, current_time: float) -> None:
        """Extract spectral features (centroid, rolloff, flux)."""
        try:
            # Apply window
            window = np.hanning(len(audio_data))
            windowed = audio_data * window
            
            # FFT
            fft = np.fft.rfft(windowed)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(windowed), 1.0 / self.sample_rate)
            
            if np.sum(magnitude) > 0:
                # Spectral centroid
                features.spectral_centroid = float(np.sum(freqs * magnitude) / np.sum(magnitude))
                
                # Spectral rolloff (85% of energy)
                cumsum = np.cumsum(magnitude)
                total_energy = cumsum[-1]
                rolloff_idx = np.searchsorted(cumsum, 0.85 * total_energy)
                if rolloff_idx < len(freqs):
                    features.spectral_rolloff = float(freqs[rolloff_idx])
                
                # Spectral flux (change from previous frame)
                if self._prev_magnitude is not None and len(self._prev_magnitude) == len(magnitude):
                    flux = np.sqrt(np.sum((magnitude - self._prev_magnitude) ** 2))
                    features.spectral_flux = float(flux)
                
                self._prev_magnitude = magnitude.copy()
        except Exception:
            pass
    
    def _update_temporal_history(self, features: AudioFeatures, current_time: float) -> None:
        """Update temporal history for pitch and energy statistics."""
        # Update pitch history (only valid pitches)
        if features.pitch_valid:
            self._pitch_history.append(features.pitch_hz)
            self._pitch_timestamps.append(current_time)
        
        # Remove old entries
        cutoff = current_time - self._pitch_window_sec
        while self._pitch_timestamps and self._pitch_timestamps[0] < cutoff:
            self._pitch_timestamps.pop(0)
            self._pitch_history.pop(0)
        
        # Energy history already updated in _extract_energy
        cutoff = current_time - self._energy_window_sec
        while self._energy_timestamps and self._energy_timestamps[0] < cutoff:
            self._energy_timestamps.pop(0)
            self._energy_history.pop(0)
        
        # Compute temporal statistics
        if self._pitch_history:
            features.pitch_mean = float(np.mean(self._pitch_history))
            features.pitch_std = float(np.std(self._pitch_history)) if len(self._pitch_history) > 1 else 0.0
            features.pitch_min = float(np.min(self._pitch_history))
            features.pitch_max = float(np.max(self._pitch_history))
            features.pitch_range = features.pitch_max - features.pitch_min
        
        if self._energy_history:
            features.energy_mean = float(np.mean(self._energy_history))
            features.energy_std = float(np.std(self._energy_history)) if len(self._energy_history) > 1 else 0.0
    
    def _assess_quality(self, features: dict[str, float]) -> DataQuality:
        """Assess data quality based on feature values."""
        if not features:
            return DataQuality.INSUFFICIENT
        
        # Check for NaN or infinite values
        for v in features.values():
            if math.isnan(v) or v == float('inf') or v == float('-inf'):
                return DataQuality.POOR
        
        # Check for silence (all energy-related features near zero)
        rms_energy = features.get("rms_energy", 0.0)
        rms_energy_db = features.get("rms_energy_db", -120.0)
        voice_activity = features.get("voice_activity", 0.0)
        
        if rms_energy < 1e-6 and rms_energy_db < -80.0 and voice_activity == 0.0:
            return DataQuality.INSUFFICIENT
        
        return DataQuality.GOOD

    def get_feature_names(self) -> list[str]:
        """Return list of feature names this extractor produces."""
        return [
            # Pitch
            "pitch_hz", "pitch_valid", "pitch_confidence",
            "pitch_mean", "pitch_std", "pitch_min", "pitch_max", "pitch_range",
            
            # Energy
            "rms_energy", "rms_energy_db", "energy_mean", "energy_std",
            
            # ZCR
            "zcr", "zcr_mean", "zcr_std",
            
            # Voice activity
            "voice_activity", "voiced_ratio",
            
            # Spectral
            "spectral_centroid", "spectral_rolloff", "spectral_flux",
        ]
    
    def reset(self) -> None:
        """Reset internal state."""
        super().reset()
        self._pitch_history.clear()
        self._pitch_timestamps.clear()
        self._energy_history.clear()
        self._energy_timestamps.clear()
        self._prev_magnitude = None