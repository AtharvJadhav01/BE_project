with open(r'C:\Users\theat\OneDrive\Desktop\BE_Project\my_app\app_work\tests\unit\test_audio_analysis.py', 'rb') as f:
    content = f.read()

# Fix the for loop structure:
# - for loop at 12 spaces (4 more than method body at 8)
# - if/else at 16 spaces (4 more than for at 12)
# - continue at 12 spaces (for loop body level, after if-else)
# - empty line after for loop at 8 spaces (method body level)
# - next method at 4 spaces

old = b'            else:\n                assert v == 0.0, f"Expected 0.0 for {k}, got {v}"\n                \n                continue\n            \n    def test_extract_sine_wave'
new = b'            else:\n                assert v == 0.0, f"Expected 0.0 for {k}, got {v}"\n            continue\n        \n        \n    def test_extract_sine_wave'

content = content.replace(old, new)

with open(r'C:\Users\theat\OneDrive\Desktop\BE_Project\my_app\app_work\tests\unit\test_audio_analysis.py', 'wb') as f:
    f.write(content)
print('Fixed')