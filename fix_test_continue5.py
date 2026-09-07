with open(r'C:\Users\theat\OneDrive\Desktop\BE_Project\my_app\app_work\tests\unit\test_audio_analysis.py', 'rb') as f:
    content = f.read()

# Fix the continue statement - move it to for loop body level (12 spaces)
old = b'        continue\n        \n    def test_extract_sine_wave'
new = b'            continue\n        \n    def test_extract_sine_wave'

content = content.replace(old, new)

with open(r'C:\Users\theat\OneDrive\Desktop\BE_Project\my_app\app_work\tests\unit\test_audio_analysis.py', 'wb') as f:
    f.write(content)
print('Fixed')