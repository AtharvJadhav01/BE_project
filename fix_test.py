with open(r'C:\Users\theat\OneDrive\Desktop\BE_Project\my_app\app_work\tests\unit\test_audio_analysis.py', 'rb') as f:
    content = f.read()

# Fix the indentation issue - add a pass statement after the for loop's if-else
old = b'                assert v == 0.0, f"Expected 0.0 for {k}, got {v}"\n        \n    def test_extract_sine_wave'
new = b'                assert v == 0.0, f"Expected 0.0 for {k}, got {v}"\n            pass\n        \n    def test_extract_sine_wave'

content = content.replace(old, new)

with open(r'C:\Users\theat\OneDrive\Desktop\BE_Project\my_app\app_work\tests\unit\test_audio_analysis.py', 'wb') as f:
    f.write(content)
print('Fixed')