"""
Face state estimation for interpreting facial features into meaningful states.

This module provides rule-based facial state estimation that converts
normalized facial features into interpretable states such as neutral,
positive expression, negative expression, surprise, and tension.
"""

import math
import time
from typing import Dict, List, Optional, Any
from analysis.interfaces import StateEstimator, StateEstimate, BaselineManager
from analysis.results import StateEstimate as StateEstimateModel


class FaceStateEstimator(StateEstimator):
    """
    Rule-based facial state estimator.
    
    Converts normalized facial features into interpretable states using
    observable facial action units rather than claiming to detect internal
    emotional states.
    """
    
    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        
        # Configuration thresholds for state detection
        self.thresholds = {
            # Eye thresholds
            "eye_openness_high": config.get("eye_openness_high", 0.7),      # Wide open eyes
            "eye_openness_low": config.get("eye_openness_low", 0.3),       # Squinted eyes
            "blink_rate_high": config.get("blink_rate_high", 25.0),        # Blinks per minute
            "blink_rate_low": config.get("blink_rate_low", 8.0),          # Blinks per minute
            
            # Mouth thresholds
            "mouth_openness_high": config.get("mouth_openness_high", 0.6), # Open mouth
            "mouth_width_high": config.get("mouth_width_high", 0.4),       # Smile width
            "mouth_width_low": config.get("mouth_width_low", -0.2),        # Frown width
            "lip_press_high": config.get("lip_press_high", 0.5),          # Lip pressing
            
            # Eyebrow thresholds
            "brow_inner_up_high": config.get("brow_inner_up_high", 0.4),   # Inner brow raise (surprise)
            "brow_down_high": config.get("brow_down_high", 0.4),           # Brow lower (anger/focus)
            "brow_outer_up_high": config.get("brow_outer_up_high", 0.3),   # Outer brow raise
            
            # Movement thresholds
            "facial_movement_high": config.get("facial_movement_high", 0.02), # High movement
            "facial_symmetry_low": config.get("facial_symmetry_low", 0.7),   # Low symmetry
            "expressivity_high": config.get("expressivity_high", 0.6),       # High expressivity
            
            # Head pose thresholds
            "head_pitch_high": config.get("head_pitch_high", 15.0),        # Looking up (degrees)
            "head_pitch_low": config.get("head_pitch_low", -15.0),         # Looking down (degrees)
            "head_yaw_high": config.get("head_yaw_high", 20.0),            # Looking right (degrees)
            "head_yaw_low": config.get("head_yaw_low", -20.0),             # Looking left (degrees)
        }
        
        # State definitions - ordered by priority (higher priority states checked first)
        self._state_definitions = [
            # High priority states
            ("surprise_like_indicator", self._is_surprise_like),
            ("negative_expression_indicator", self._is_negative_expression),
            ("positive_expression_indicator", self._is_positive_expression),
            ("elevated_facial_tension_indicator", self._is_elevated_tension),
            # Lower priority states
            ("neutral", self._is_neutral)
        ]
    
    def estimate(self, features: Dict[str, float], 
                baseline_manager: BaselineManager) -> Optional[StateEstimate]:
        """
        Estimate facial state from normalized features.
        
        Args:
            features: Dictionary of normalized facial feature values (z-scores)
            baseline_manager: Baseline manager for this modality
            
        Returns:
            StateEstimate if estimation successful, None otherwise
        """
        if not features:
            return None
            
        # Check if baseline is ready for face features
        if not baseline_manager.is_face_baseline_ready():
            return StateEstimateModel(
                state="uncertain",
                confidence=0.0,
                timestamp=time.monotonic(),
                wallclock=self._get_wallclock_timestamp(),
                contributing_features={},
                uncertainty=1.0,  # High uncertainty when baseline not ready
                modality="face"
            )
        
        # Evaluate each state in priority order
        for state_name, evaluation_func in self._state_definitions:
            try:
                result = evaluation_func(features)
                if result is not None:
                    state, confidence, contributing_features, uncertainty = result
                    
                    return StateEstimateModel(
                        state=state,
                        confidence=confidence,
                        timestamp=time.monotonic(),
                        wallclock=self._get_wallclock_timestamp(),
                        contributing_features=contributing_features,
                        uncertainty=uncertainty,
                        modality="face",
                        baseline_status=baseline_manager.get_status("face_features") or 
                                    baseline_manager.get_status("eye_openness_left") or
                                    "unknown",
                        raw_features=features.copy()
                    )
            except Exception:
                # Continue to next state if evaluation fails
                continue
        
        # Fallback to neutral if no state matched
        return StateEstimateModel(
            state="neutral",
            confidence=0.5,
            timestamp=time.monotonic(),
            wallclock=self._get_wallclock_timestamp(),
            contributing_features={},
            uncertainty=0.5,
            modality="face"
        )
    
    def _is_surprise_like(self, features: Dict[str, float]) -> Optional[tuple]:
        """Check for surprise-like indicators: brows up, eyes wide, mouth open."""
        try:
            # Surprise indicators:
            # - Inner brows raised (browInnerUp)
            # - Eyes wide open (high eye openness)
            # - Jaw open (mouth openness)
            # - Symmetric expression
            
            brow_up = features.get("brow_inner_up", 0.0)
            eye_openness = features.get("eye_openness_avg", 0.0)
            mouth_open = features.get("mouth_openness", 0.0)
            symmetry = features.get("facial_symmetry", 1.0)
            
            # Check if indicators exceed thresholds
            if (brow_up > self.thresholds["brow_inner_up_high"] and
                eye_openness > self.thresholds["eye_openness_high"] and
                mouth_open > self.thresholds["mouth_openness_high"] and
                symmetry > self.thresholds["facial_symmetry_low"]):
                
                # Calculate confidence based on strength of indicators
                brow_conf = min(1.0, brow_up / self.thresholds["brow_inner_up_high"])
                eye_conf = min(1.0, eye_openness / self.thresholds["eye_openness_high"])
                mouth_conf = min(1.0, mouth_open / self.thresholds["mouth_openness_high"])
                sym_conf = symmetry  # Already 0-1
                
                confidence = (brow_conf + eye_conf + mouth_conf + sym_conf) / 4.0
                confidence = min(0.95, max(0.1, confidence))  # Clamp to reasonable range
                
                contributing = {
                    "brow_inner_up": brow_up,
                    "eye_openness_avg": eye_openness,
                    "mouth_openness": mouth_open,
                    "facial_symmetry": symmetry
                }
                
                uncertainty = 1.0 - confidence
                
                return ("surprise_like_indicator", confidence, contributing, uncertainty)
        except Exception:
            pass
        return None
    
    def _is_negative_expression(self, features: Dict[str, float]) -> Optional[tuple]:
        """Check for negative expression indicators: brows down, lips pressed."""
        try:
            # Negative expression indicators:
            # - Brows lowered and drawn together (browDown)
            # - Lips pressed or compressed
            # - Possible lower forehead tension
            
            brow_down = (features.get("brow_down_left", 0.0) + 
                        features.get("brow_down_right", 0.0)) / 2.0
            lip_press = features.get("lip_press", 0.0)
            brow_inner_up = features.get("brow_inner_up", 0.0)  # Should be low for negative
            
            # Check if indicators exceed thresholds
            if (brow_down > self.thresholds["brow_down_high"] and
                lip_press > self.thresholds["lip_press_high"] and
                brow_inner_up < self.thresholds["brow_inner_up_high"] * 0.5):  # Not surprised
                
                # Calculate confidence
                brow_conf = min(1.0, brow_down / self.thresholds["brow_down_high"])
                lip_conf = min(1.0, lip_press / self.thresholds["lip_press_high"])
                
                confidence = (brow_conf + lip_conf) / 2.0
                confidence = min(0.95, max(0.1, confidence))
                
                contributing = {
                    "brow_down_left": features.get("brow_down_left", 0.0),
                    "brow_down_right": features.get("brow_down_right", 0.0),
                    "lip_press": lip_press,
                    "brow_inner_up": brow_inner_up
                }
                
                uncertainty = 1.0 - confidence
                
                return ("negative_expression_indicator", confidence, contributing, uncertainty)
        except Exception:
            pass
        return None
    
    def _is_positive_expression(self, features: Dict[str, float]) -> Optional[tuple]:
        """Check for positive expression indicators: smile, cheek raise."""
        try:
            # Positive expression indicators:
            # - Mouth corners raised (mouth width positive)
            # - Possible eye crinkling
            # - Symmetric or slightly asymmetric
            
            mouth_width = features.get("mouth_width", 0.0)
            eye_openness = features.get("eye_openness_avg", 0.0)
            smile_asymmetry = features.get("smile_asymmetry", 0.0)
            
            # Positive smile (not frown)
            if (mouth_width > self.thresholds["mouth_width_high"] and
                eye_openness > 0.3 and  # Eyes reasonably open
                smile_asymmetry < 0.3):  # Not extremely asymmetric
                
                # Calculate confidence
                mouth_conf = min(1.0, mouth_width / self.thresholds["mouth_width_high"])
                eye_conf = min(1.0, eye_openness / 0.8)  # Normalize eye openness
                sym_conf = max(0.0, 1.0 - (smile_asymmetry / 0.5))  # Lower asymmetry = higher conf
                
                confidence = (mouth_conf + eye_conf + sym_conf) / 3.0
                confidence = min(0.95, max(0.1, confidence))
                
                contributing = {
                    "mouth_width": mouth_width,
                    "eye_openness_avg": eye_openness,
                    "smile_asymmetry": smile_asymmetry
                }
                
                uncertainty = 1.0 - confidence
                
                return ("positive_expression_indicator", confidence, contributing, uncertainty)
        except Exception:
            pass
        return None
    
    def _is_elevated_tension(self, features: Dict[str, float]) -> Optional[tuple]:
        """Check for elevated facial tension indicators: high movement, asymmetry, lip press."""
        try:
            # Elevated tension indicators:
            # - High facial movement velocity
            # - Low facial symmetry (asymmetrical tension)
            # - Lip pressing or jaw clenching
            # - Brow tension
            
            movement = features.get("facial_movement_velocity", 0.0)
            symmetry = features.get("facial_symmetry", 1.0)
            lip_press = features.get("lip_press", 0.0)
            brow_down = (features.get("brow_down_left", 0.0) + 
                        features.get("brow_down_right", 0.0)) / 2.0
            jaw_open = features.get("mouth_openness", 0.0)  # Jaw tension can be open or closed
            
            # Check if indicators exceed thresholds
            if ((movement > self.thresholds["facial_movement_high"] or
                 symmetry < self.thresholds["facial_symmetry_low"] or
                 lip_press > self.thresholds["lip_press_high"] or
                 brow_down > self.thresholds["brow_down_high"]) and
                jaw_open < 0.8):  # Not extreme surprise/yawn
                
                # Calculate confidence based on which indicators are present
                indicators = []
                
                if movement > self.thresholds["facial_movement_high"]:
                    indicators.append(min(1.0, movement / self.thresholds["facial_movement_high"]))
                
                if symmetry < self.thresholds["facial_symmetry_low"]:
                    sym_deficit = 1.0 - symmetry
                    indicators.append(min(1.0, sym_deficit / (1.0 - self.thresholds["facial_symmetry_low"])))
                
                if lip_press > self.thresholds["lip_press_high"]:
                    indicators.append(min(1.0, lip_press / self.thresholds["lip_press_high"]))
                
                if brow_down > self.thresholds["brow_down_high"]:
                    indicators.append(min(1.0, brow_down / self.thresholds["brow_down_high"]))
                
                if indicators:
                    confidence = sum(indicators) / len(indicators)
                    confidence = min(0.95, max(0.1, confidence))
                    
                    contributing = {
                        "facial_movement_velocity": movement,
                        "facial_symmetry": symmetry,
                        "lip_press": lip_press,
                        "brow_down_left": features.get("brow_down_left", 0.0),
                        "brow_down_right": features.get("brow_down_right", 0.0)
                    }
                    
                    uncertainty = 1.0 - confidence
                    
                    return ("elevated_facial_tension_indicator", confidence, contributing, uncertainty)
        except Exception:
            pass
        return None
    
    def _is_neutral(self, features: Dict[str, float]) -> Optional[tuple]:
        """Check for neutral state: relaxed, balanced features."""
        try:
            # Neutral indicators:
            # - Eyes at moderate openness
            # - Mouth relaxed (neutral width)
            # - Brows relaxed
            # - Symmetric
            # - Low movement
            
            eye_openness = features.get("eye_openness_avg", 0.0)
            mouth_width = features.get("mouth_width", 0.0)
            brow_inner_up = features.get("brow_inner_up", 0.0)
            brow_down = (features.get("brow_down_left", 0.0) + 
                        features.get("brow_down_right", 0.0)) / 2.0
            symmetry = features.get("facial_symmetry", 1.0)
            movement = features.get("facial_movement_velocity", 0.0)
            lip_press = features.get("lip_press", 0.0)
            
            # Check if features are within neutral ranges
            eye_neutral = 0.3 <= eye_openness <= 0.7
            mouth_neutral = -0.2 <= mouth_width <= 0.2
            brow_relaxed = brow_inner_up < 0.3 and brow_down < 0.3
            symmetric = symmetry > 0.7
            low_movement = movement < 0.01
            relaxed_lips = lip_press < 0.2
            
            # Count how many neutral indicators are met
            neutral_indicators = [
                eye_neutral,
                mouth_neutral,
                brow_relaxed,
                symmetric,
                low_movement,
                relaxed_lips
            ]
            
            neutral_count = sum(neutral_indicators)
            total_indicators = len(neutral_indicators)
            
            if neutral_count >= total_indicators * 0.6:  # At least 60% neutral indicators
                # Calculate confidence based on how neutral the features are
                eye_conf = 1.0 - min(1.0, abs(eye_openness - 0.5) / 0.5)  # Distance from ideal 0.5
                mouth_conf = 1.0 - min(1.0, abs(mouth_width) / 0.5)      # Distance from 0
                brow_conf = 1.0 - min(1.0, (brow_inner_up + brow_down) / 0.6)
                sym_conf = symmetry  # Already 0-1
                move_conf = max(0.0, 1.0 - movement / 0.02)  # Lower movement = higher conf
                lip_conf = 1.0 - min(1.0, lip_press / 0.3)
                
                confidences = [eye_conf, mouth_conf, brow_conf, sym_conf, move_conf, lip_conf]
                confidence = sum(confidences) / len(confidences)
                confidence = min(0.95, max(0.3, confidence))  # Neutral should have decent confidence
                
                contributing = {
                    "eye_openness_avg": eye_openness,
                    "mouth_width": mouth_width,
                    "brow_inner_up": brow_inner_up,
                    "brow_down_left": features.get("brow_down_left", 0.0),
                    "brow_down_right": features.get("brow_down_right", 0.0),
                    "facial_symmetry": symmetry,
                    "facial_movement_velocity": movement
                }
                
                uncertainty = 1.0 - confidence
                
                return ("neutral", confidence, contributing, uncertainty)
        except Exception:
            pass
        return None
    
    def _get_wallclock_timestamp(self) -> str:
        """Get current wall-clock timestamp in ISO format."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    
    def get_possible_states(self) -> List[str]:
        """Return list of possible state labels."""
        return [
            "surprise_like_indicator",
            "negative_expression_indicator", 
            "positive_expression_indicator",
            "elevated_facial_tension_indicator",
            "neutral",
            "uncertain"
        ]
    
    def reset(self) -> None:
        """Reset internal state."""
        pass  # No internal state to reset in this implementation