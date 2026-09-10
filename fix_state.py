with open('src/analysis/face/state.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the last occurrence of 'def set_enabled'
idx = content.rfind('def set_enabled(self, enabled: bool) -> None:')
if idx >= 0:
    # Find the end of this method (next blank line or end of file)
    idx_end = content.find('\n\n', idx)
    if idx_end < 0:
        idx_end = len(content)
    
    # Replace the method with the method + get_possible_states
    old = content[idx:idx_end]
    new = '''def set_enabled(self, enabled: bool) -> None:
        """Enable or disable smoothing."""
        self.enabled = enabled
        if not enabled:
            self.reset()

    def get_possible_states(self) -> List[str]:
        """Return list of possible emotion labels."""
        if self._labels:
            return list(self._labels)
        return [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprised",
            "neutral",
            "uncertain"
        ]'''

    content = content[:idx] + content[idx_end:]
    content = content[:idx] + '''def set_enabled(self, enabled: bool) -> None:
        """Enable or disable smoothing."""
        self.enabled = enabled
        if not enabled:
            self.reset()

    def get_possible_states(self) -> List[str]:
        """Return list of possible emotion labels."""
        if self._labels:
            return list(self._labels)
        return [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprised",
            "neutral",
            "uncertain"
        ]'''

# We need to replace from the start of set_enabled to the end of the file
idx_start = content.rfind('def set_enabled(self, enabled: bool) -> None:')
# Find the end of the file or next method/class
# The method ends without a blank line after it, so we need to add our method after it
content = content[:content.rfind('def set_enabled(self, enabled: bool) -> None:')] + '''def set_enabled(self, enabled: bool) -> None:
        """Enable or disable smoothing."""
        self.enabled = enabled
        if not enabled:
            self.reset()

    def get_possible_states(self) -> List[str]:
        """Return list of possible emotion labels."""
        if self._labels:
            return list(self._labels)
        return [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprised",
            "neutral",
            "uncertain"
        ]'''

with open('src/analysis/face/state.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done!')