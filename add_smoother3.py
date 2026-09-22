#!/usr/bin/env python
import sys
sys.path.insert(0, 'src')

# Read the file
with open('src/face/capture.py', 'r') as f:
    content = f.read()

# Find the expression_estimate section in _run_face_analysis and add smoother integration
# We need to add smoother.add_prediction() and get_smoothed()

# Locate the expression estimate creation and add smoother integration
# The expression_estimate is created in _run_face_analysis method

# We need to add after the expression_estimate is set
# And before the analysis_result is created

# Let's find the specific section
target = '''expression_estimate = None
            if (self._emotion_enabled and 
                self._emotion_classifier is not None and 
                detection_result.face_detected):'''

replacement = '''expression_estimate = None
            if (self._emotion_enabled and 
                self._emotion_classifier is not None and 
                detection_result.face_detected):
                
                # Add prediction to smoother
                try:
                    raw_probs = self._emotion_classifier.get_metadata().get("target_names", EMOTION_LABELS)
                    # Get the raw prediction first
                    result = self._emotion_classifier.predict(raw_features)
                    probas = result["probabilities"]
                    # Convert to numpy array in correct order
                    probs_array = np.array([probas[label] for label in raw_probs], dtype=np.float64)
                    confidence = result["confidence"]
                    self._emotion_smoother.add_prediction(probs_array, confidence)
                    
                    # Get smoothed result
                    smoothed_expr, smoothed_conf = self._emotion_smoother.get_smoothed()
                    
                    if smoothed_expr is not None:
                        expression_estimate = {
                            "expression": smoothed_expr,
                            "confidence": smoothed_conf,
                            "probabilities": result["probabilities"],
                            "timestamp": timestamp_monotonic,
                            "wallclock": timestamp_wallclock,
                        }
                    else:
                        # No valid smoothed prediction - check if we should show unavailable
                        if not detection_result.face_detected:
                            expression_estimate = {
                                "expression": "unavailable",
                                "confidence": 0.0,
                                "probabilities": {},
                                "timestamp": timestamp_monotonic,
                                "wallclock": timestamp_wallclock,
                                "reason": "no_face_detected"
                            }
                        else:
                            # Classifier not confident enough, show unavailable
                            expression_estimate = {
                                "expression": "unavailable",
                                "confidence": 0.0,
                                "probabilities": {},
                                "timestamp": timestamp_monotonic,
                                "wallclock": timestamp_wallclock,
                                "reason": "low_confidence"
                            }
                except Exception as e:
                    self.logger.warning("emotion_smoothing_failed", error=str(e))
                    # Fall back to raw prediction without smoothing
                    expression_estimate = self._emotion_classifier.predict(raw_features)'''

if target in content:
    content = content.replace(target, replacement)
    with open('src/face/capture.py', 'w') as f:
        f.write(content)
    print('EmotionSmoother integration added successfully')
else:
    print('Target pattern not found')
    # Try to find a simpler location
    idx = content.find("expression_estimate = None")
    if idx != -1:
        print(f"Found at {idx}, showing context:")
        print(content[idx:idx+300])